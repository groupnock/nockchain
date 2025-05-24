# Writing the fully patched and translated Rust code to a file for user download
code = r'''use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::collections::HashMap;

use tokio::runtime::Runtime;
use tokio::sync::{Semaphore, oneshot};
use tokio::task::JoinSet;

use rayon::prelude::*;
use rayon::iter::IntoParallelIterator;

use tempfile::tempdir;
use tracing::{instrument, warn, error, info, debug};
use thiserror::Error;
use libc::{sched_param, sched_setscheduler, SCHED_RR};

use crate::NockchainCli;
use kernels::miner::KERNEL;
use nockapp::kernel::checkpoint::JamPaths;
use nockapp::kernel::form::Kernel;
use nockapp::nockapp::driver::{IODriverFn, NockAppHandle, PokeResult};
use nockapp::nockapp::wire::Wire;
use nockapp::nockapp::NockAppError;
use nockapp::noun::slab::NounSlab;
use nockapp::noun::{AtomExt, NounExt};
use nockvm::noun::{Atom, D, T};
use nockvm_macros::tas;

/// Wire message types for mining
pub enum MiningWire {
    Mined,
    Candidate,
    SetPubKey,
    Enable,
}

impl MiningWire {
    fn verb(&self) -> &'static str {
        match self {
            MiningWire::Mined     => "mined",
            MiningWire::Candidate => "candidate",
            MiningWire::SetPubKey => "setpubkey",
            MiningWire::Enable    => "enable",
        }
    }
}

impl Wire for MiningWire {
    const VERSION: u64 = 1;
    const SOURCE: &'static str = "miner";

    fn to_wire(&self) -> nockapp::wire::WireRepr {
        let tags = vec![self.verb().into()];
        nockapp::wire::WireRepr::new(Self::SOURCE, Self::VERSION, tags)
    }
}

/// Configuration for mining keys, parsed from "share,m:key1,key2,..."
#[derive(Debug, Clone)]
pub struct MiningKeyConfig {
    pub share: u64,
    pub m: u64,
    pub keys: Vec<String>,
}

impl FromStr for MiningKeyConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err("Invalid format. Expected 'share,m:key1,key2,...'".into());
        }
        let sm: Vec<&str> = parts[0].split(',').collect();
        if sm.len() != 2 {
            return Err("Invalid share,m format".into());
        }
        let share = sm[0].parse::<u64>().map_err(|e| e.to_string())?;
        let m     = sm[1].parse::<u64>().map_err(|e| e.to_string())?;
        let keys  = parts[1].split(',').map(String::from).collect();
        Ok(Self { share, m, keys })
    }
}

/// A fixed-size memory pool with periodic cleanup and shrink capability
pub struct MemoryPool {
    blocks: Vec<Vec<u8>>,
    active_blocks: AtomicUsize,
    max_blocks: usize,
    cleanup_interval: Duration,
    last_cleanup: Instant,
    shrink_threshold: f64,
    shutting_down: AtomicBool,
    warning_threshold: f64,
    last_warning: Instant,
    warning_cooldown: Duration,
}

impl MemoryPool {
    pub fn new(block_size: usize, max_blocks: usize, cleanup_interval: Duration, shrink_threshold: f64) -> Self {
        let mut blocks = Vec::with_capacity(max_blocks);
        for _ in 0..max_blocks {
            blocks.push(vec![0; block_size]);
        }
        Self {
            blocks,
            active_blocks: AtomicUsize::new(0),
            max_blocks,
            cleanup_interval,
            last_cleanup: Instant::now(),
            shrink_threshold,
            shutting_down: AtomicBool::new(false),
            warning_threshold: 0.9,
            last_warning: Instant::now(),
            warning_cooldown: Duration::from_secs(60),
        }
    }

    pub fn get_block(&mut self) -> Option<Vec<u8>> {
        if self.shutting_down.load(Ordering::Relaxed) {
            return None;
        }
        let active = self.active_blocks.load(Ordering::Relaxed);
        if active >= self.max_blocks {
            return None;
        }
        let usage = active as f64 / self.max_blocks as f64;
        if usage >= self.warning_threshold && self.last_warning.elapsed() >= self.warning_cooldown {
            warn!("Memory pool usage is high: {:.1}%", usage * 100.0);
            self.last_warning = Instant::now();
        }
        if self.last_cleanup.elapsed() >= self.cleanup_interval {
            self.cleanup();
        }
        self.active_blocks.fetch_add(1, Ordering::Relaxed);
        self.blocks.pop()
    }

    pub fn return_block(&mut self, mut block: Vec<u8>) {
        if self.shutting_down.load(Ordering::Relaxed) {
            return;
        }
        block.clear();
        self.blocks.push(block);
        self.active_blocks.fetch_sub(1, Ordering::Relaxed);
    }

    fn cleanup(&mut self) {
        let active = self.active_blocks.load(Ordering::Relaxed);
        let target = (self.max_blocks as f64 * self.shrink_threshold) as usize;
        if active < target {
            self.blocks.truncate(active);
            debug!("Memory pool shrunk to {}", self.blocks.len());
        }
        self.last_cleanup = Instant::now();
    }

    pub fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::Relaxed);
    }
}

/// Errors that can occur in mining logic
#[derive(Debug, Error)]
pub enum MiningError {
    #[error("Memory pool error: {0}")]
    MemoryPoolError(String),
    #[error("Kernel error: {0}")]
    KernelError(String),
    #[error("Runtime error: {0}")]
    RuntimeError(String),
    #[error("Shutdown error: {0}")]
    ShutdownError(String),
    #[error("Resource error: {0}")]
    ResourceError(String),
}

/// Collects and reports mining performance metrics
pub struct PerformanceMonitor {
    start_time: Instant,
    proof_count: AtomicU64,
    error_count: AtomicU64,
    success_count: AtomicU64,
    resource_usage: Arc<Mutex<HashMap<String, u64>>>,
    report_interval: Duration,
    last_report: Instant,
    hash_time_ns: AtomicU64,
    hash_count: AtomicU64,
    alloc_count: AtomicU64,
    free_count: AtomicU64,
    thread_usage_pct: AtomicU64,
    thread_total: AtomicU64,
    error_types: Arc<Mutex<HashMap<String, u64>>>,
}

impl PerformanceMonitor {
    pub fn new(interval_secs: u64) -> Self {
        Self {
            start_time: Instant::now(),
            proof_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            resource_usage: Arc::new(Mutex::new(HashMap::new())),
            report_interval: Duration::from_secs(interval_secs),
            last_report: Instant::now(),
            hash_time_ns: AtomicU64::new(0),
            hash_count: AtomicU64::new(0),
            alloc_count: AtomicU64::new(0),
            free_count: AtomicU64::new(0),
            thread_usage_pct: AtomicU64::new(0),
            thread_total: AtomicU64::new(0),
            error_types: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn record_error(&self, kind: &str) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
        let mut et = self.error_types.lock().unwrap();
        *et.entry(kind.to_string()).or_insert(0) += 1;
    }

    pub fn record_resource(&self, name: &str, qty: u64) {
        let mut ru = self.resource_usage.lock().unwrap();
        *ru.entry(name.to_string()).or_insert(0) += qty;
    }

    pub fn record_hash(&self, duration: Duration) {
        self.hash_time_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.hash_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_alloc(&self) {
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_free(&self) {
        self.free_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_threads(&self, active: usize, total: usize) {
        self.thread_total.store(total as u64, Ordering::Relaxed);
        let pct = ((active as f64 / total as f64) * 100.0) as u64;
        self.thread_usage_pct.store(pct, Ordering::Relaxed);
    }

    pub fn should_report(&self) -> bool {
        self.last_report.elapsed() >= self.report_interval
    }

    pub fn report(&self) -> String {
        let uptime = self.start_time.elapsed();
        let proofs = self.proof_count.load(Ordering::Relaxed);
        let errs = self.error_count.load(Ordering::Relaxed);
        let succ = self.success_count.load(Ordering::Relaxed);
        let avg_hash = if self.hash_count.load(Ordering::Relaxed) > 0 {
            self.hash_time_ns.load(Ordering::Relaxed) as f64
                / self.hash_count.load(Ordering::Relaxed) as f64
        } else { 0.0 };
        let allocs = self.alloc_count.load(Ordering::Relaxed);
        let frees  = self.free_count.load(Ordering::Relaxed);
        let leaks  = allocs.saturating_sub(frees);
        let usage  = self.thread_usage_pct.load(Ordering::Relaxed);

        let mut s = format!(
            "Uptime: {:?}\\nProofs: {}\\nSuccess: {}\\nErrors: {}\\nAvg hash time: {:.2} ns\\nMem allocs: {}\\nMem frees: {}\\nMem leak: {}\\nThread use: {}%\\n",
            uptime, proofs, succ, errs, avg_hash, allocs, frees, leaks, usage
        );

        s.push_str("Error types:\\n");
        for (k,v) in self.error_types.lock().unwrap().iter() {
            s.push_str(&format!("- {}: {}\\n", k, v));
        }
        s.push_str("Resource usage:\\n");
        for (k,v) in self.resource_usage.lock().unwrap().iter() {
            s.push_str(&format!("- {}: {}\\n", k, v));
        }

        s
    }
}

/// Manages mining threads, resources, retries, shutdown
pub struct MiningManager {
    pub threads: usize,
    pub runtime: Runtime,
    pub monitor: Arc<PerformanceMonitor>,
    pub retries: usize,
    pub retry_delay: Duration,
    pub pool_size_mb: usize,
    pub shutting_down: Arc<AtomicBool>,
    pub pool_cleanup: Duration,
    pub shrink_thresh: f64,
    pub tasks: Arc<Mutex<HashMap<String, Instant>>>,
    pub limits: Arc<Mutex<HashMap<String,u64>>>,
    pub task_timeout: Duration,
}

impl MiningManager {
    pub fn new(cfg: Option<&NockchainCli>) -> Self {
        let threads = cfg.and_then(|c| Some(c.mining_threads)).filter(|&n| n>0).unwrap_or(15);
        let rt = Runtime::new().expect("Tokio runtime");
        let mon = Arc::new(PerformanceMonitor::new(
            cfg.map(|c| c.mining_report_interval).unwrap_or(60)
        ));
        let retries = cfg.map(|c| c.mining_max_retries).unwrap_or(3);
        let retry_delay = Duration::from_secs(cfg.map(|c| c.mining_retry_delay).unwrap_or(1));
        let pool_size_mb = cfg.map(|c| c.mining_memory_pool).unwrap_or(8192);
        let pool_cl = Duration::from_secs(cfg.map(|c| c.mining_cleanup_interval).unwrap_or(300));
        let shrink = cfg.map(|c| c.mining_memory_shrink_threshold).unwrap_or(0.5);
        let to = Duration::from_secs(30);

        let mut lim = HashMap::new();
        lim.insert("mem_blocks".into(), pool_size_mb as u64);
        lim.insert("hashes".into(), 10000);
        lim.insert("proofs".into(), 1000);

        Self {
            threads, runtime: rt, monitor: mon,
            retries, retry_delay, pool_size_mb,
            shutting_down: Arc::new(AtomicBool::new(false)),
            pool_cleanup: pool_cl, shrink_thresh: shrink,
            tasks: Arc::new(Mutex::new(HashMap::new())),
            limits: Arc::new(Mutex::new(lim)),
            task_timeout: to,
        }
    }

    pub fn track(&self, id: String) {
        self.tasks.lock().unwrap().insert(id, Instant::now());
    }
    pub fn timed_out(&self,id:&str)->bool {
        if let Some(start)=self.tasks.lock().unwrap().get(id) {
            start.elapsed()>self.task_timeout
        } else { false }
    }
    pub fn remove(&self,id:&str){ self.tasks.lock().unwrap().remove(id); }
    pub fn limit_ok(&self,name:&str,amt:u64)->bool {
        if let Some(&l)=self.limits.lock().unwrap().get(name) {
            let u=self.monitor.resource_usage.lock().unwrap().get(name).cloned().unwrap_or(0);
            u+amt<=l
        } else { true }
    }
    pub fn shutdown(&self){ self.shutting_down.store(true,Ordering::Relaxed) }
}

/// Builds the I/O driver for nockapp
pub fn create_mining_driver(
    cfg: Option<Vec<MiningKeyConfig>>,
    enable: bool,
    init_tx: Option<oneshot::Sender<()>>,
    cli: Option<&NockchainCli>,
)->IODriverFn {
    Box::new(move |mut handle| Box::pin(async move {
        if cfg.is_none() {
            enable_mining(&handle,false).await?;
            if let Some(t)=init_tx { let _=t.send(());}
            return Ok(());
        }
        let keys=cfg.unwrap();
        if keys.len()==1 && keys[0].share==1&&keys[0].m==1 {
            set_mining_key(&handle,keys[0].keys[0].clone()).await?;
        } else {
            set_mining_key_advanced(&handle,keys).await?;
        }
        enable_mining(&handle,enable).await?;
        if let Some(t)=init_tx { let _=t.send(());}
        if !enable {return Ok(());}

        let mut next:Option<NounSlab>=None;
        let mut set=JoinSet::new();
        loop {
            tokio::select! {
                eff=handle.next_effect()=> {
                    let e=eff?;
                    let c=unsafe{e.root().as_cell()?};
                    if c.head().eq_bytes("mine") {
                        let mut s=NounSlab::new();
                        s.copy_into(c.tail());
                        if !set.is_empty(){ next=Some(s); }
                        else {
                            let (h1,h2)=handle.dup();
                            handle=h1;
                            set.spawn(mining_attempt(s,h2,cli));
                        }
                    }
                }
                _= set.join_next(),if !set.is_empty()=> {
                    if let Some(s)=next.take() {
                        let (h1,h2)=handle.dup();
                        handle=h1;
                        set.spawn(mining_attempt(s,h2,cli));
                    }
                }
            }
        }
    }))
}

/// Executes one mining attempt
pub async fn mining_attempt(
    candidate: NounSlab,
    mut handle: NockAppHandle,
    cli: Option<&NockchainCli>
) {
    let mgr=MiningManager::new(cli);
    let pool=Arc::new(Mutex::new(MemoryPool::new(
        mgr.pool_size_mb*1024*1024,
        mgr.threads,
        mgr.pool_cleanup,
        mgr.shrink_thresh,
    )));
    let dir=tempdir().unwrap();
    let hot=zkvm_jetpack::hot::produce_prover_hot_state();
    let jam=JamPaths::new(dir.path());
    let kernel=Kernel::load_with_hot_state_huge(
        dir.path().to_path_buf(), jam.clone(), KERNEL, &hot,false
    ).await.unwrap();

    (0..mgr.threads).into_par_iter().for_each(|_| {
        if mgr.shutting_down.load(Ordering::Relaxed) { return; }
        #[cfg(target_os="linux")]
        unsafe{
            let p=sched_param{sched_priority:99};
            let _=sched_setscheduler(0,SCHED_RR,&p);
        }
        let mut r=0;
        while r<mgr.retries && !mgr.shutting_down.load(Ordering::Relaxed) {
            let block={ let mut p=pool.lock().unwrap();
                p.get_block().unwrap_or_default()
            };
            mgr.monitor.record_alloc();
            mgr.monitor.record_resource("mem_blocks",1);
            let effects=futures::executor::block_on(
                kernel.poke(MiningWire::Candidate.to_wire(),candidate.clone())
            ).unwrap();
            for eff in effects.to_vec() {
                let c=unsafe{eff.root().as_cell().unwrap()};
                if c.head().eq_bytes("command") {
                    futures::executor::block_on(async{
                        handle.poke(MiningWire::Mined.to_wire(),eff).await.ok();
                    });
                    mgr.monitor.record_success();
                }
            }
            { let mut p=pool.lock().unwrap(); p.return_block(block); }
            mgr.monitor.record_free();
            mgr.monitor.record_resource("mem_blocks",1);
            r+=1;
        }
    });

    if mgr.monitor.should_report() {
        info!("{}",mgr.monitor.report());
    }
    pool.lock().unwrap().shutdown();
    mgr.shutdown();
}

#[instrument(skip(handle))]
async fn set_mining_key(
    handle:&NockAppHandle, pubkey:String
)->Result<PokeResult, NockAppError>{
    let mut slab=NounSlab::new();
    let cmd=Atom::from_value(&mut slab,"command").unwrap();
    let sk=Atom::from_value(&mut slab,"set-mining-key").unwrap();
    let pk=Atom::from_value(&mut slab,&pubkey).unwrap();
    let poke=T(&mut slab,&[D(cmd.as_bytes()),sk.as_noun(),pk.as_noun()]);
    slab.set_root(poke);
    handle.poke(MiningWire::SetPubKey.to_wire(),slab).await
}

#[instrument(skip(handle))]
async fn set_mining_key_advanced(
    handle:&NockAppHandle, configs:Vec<MiningKeyConfig>
)->Result<PokeResult, NockAppError>{
    let mut slab=NounSlab::new();
    let cmd=Atom::from_value(&mut slab,"command").unwrap();
    let adv=Atom::from_value(&mut slab,"set-mining-key-advanced").unwrap();
    let mut list=D(0);
    for cfg in configs {
        let mut kl=D(0);
        for k in cfg.keys {
            let a=Atom::from_value(&mut slab,&k).unwrap();
            kl=T(&mut slab,&[a.as_noun(),kl]);
        }
        let tup=T(&mut slab,&[D(cfg.share),D(cfg.m),kl]);
        list=T(&mut slab,&[tup,list]);
    }
    let poke=T(&mut slab,&[D(cmd.as_bytes()),adv.as_noun(),list]);
    slab.set_root(poke);
    handle.poke(MiningWire::SetPubKey.to_wire(),slab).await
}

#[instrument(skip(handle))]
async fn enable_mining(
    handle:&NockAppHandle, enable:bool
)->Result<PokeResult, NockAppError>{
    let mut slab=NounSlab::new();
    let cmd=Atom::from_value(&mut slab,"command").unwrap();
    let en=Atom::from_value(&mut slab,"enable-mining").unwrap();
    let flag=D(if enable{0}else{1});
    let poke=T(&mut slab,&[D(cmd.as_bytes()),en.as_noun(),flag]);
    slab.set_root(poke);
    handle.poke(MiningWire::Enable.to_wire(),slab).await
}'''
with open('/mnt/data/mining.rs', 'w') as f:
    f.write(code)


