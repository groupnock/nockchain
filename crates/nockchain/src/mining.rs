use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use rayon::prelude::*;
use std::time::Instant;
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;
use std::sync::atomic::AtomicBool;

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
use tempfile::tempdir;
use tracing::{instrument, warn, error, info, debug};

pub enum MiningWire {
    Mined,
    Candidate,
    SetPubKey,
    Enable,
}

impl MiningWire {
    pub fn verb(&self) -> &'static str {
        match self {
            MiningWire::Mined => "mined",
            MiningWire::SetPubKey => "setpubkey",
            MiningWire::Candidate => "candidate",
            MiningWire::Enable => "enable",
        }
    }
}

impl Wire for MiningWire {
    const VERSION: u64 = 1;
    const SOURCE: &'static str = "miner";

    fn to_wire(&self) -> nockapp::wire::WireRepr {
        let tags = vec![self.verb().into()];
        nockapp::wire::WireRepr::new(MiningWire::SOURCE, MiningWire::VERSION, tags)
    }
}

#[derive(Debug, Clone)]
pub struct MiningKeyConfig {
    pub share: u64,
    pub m: u64,
    pub keys: Vec<String>,
}

impl FromStr for MiningKeyConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Expected format: "share,m:key1,key2,key3"
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err("Invalid format. Expected 'share,m:key1,key2,key3'".to_string());
        }

        let share_m: Vec<&str> = parts[0].split(',').collect();
        if share_m.len() != 2 {
            return Err("Invalid share,m format".to_string());
        }

        let share = share_m[0].parse::<u64>().map_err(|e| e.to_string())?;
        let m = share_m[1].parse::<u64>().map_err(|e| e.to_string())?;
        let keys: Vec<String> = parts[1].split(',').map(String::from).collect();

        Ok(MiningKeyConfig { share, m, keys })
    }
}

pub struct MemoryPool {
    blocks: Vec<Vec<u8>>,
    block_size: usize,
    max_blocks: usize,
    active_blocks: AtomicUsize,
    semaphore: Arc<Semaphore>,
    last_cleanup: Instant,
    cleanup_interval: Duration,
    is_shutting_down: AtomicBool,
    shrink_threshold: f64,
    memory_warning_threshold: f64,
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
            block_size,
            max_blocks,
            active_blocks: AtomicUsize::new(0),
            semaphore: Arc::new(Semaphore::new(max_blocks)),
            last_cleanup: Instant::now(),
            cleanup_interval,
            is_shutting_down: AtomicBool::new(false),
            shrink_threshold,
            memory_warning_threshold: 0.9, // 90% 内存使用率触发警告
            last_warning: Instant::now(),
            warning_cooldown: Duration::from_secs(60), // 1分钟冷却时间
        }
    }

    pub fn get_block(&mut self) -> Option<&mut Vec<u8>> {
        if self.is_shutting_down.load(Ordering::Relaxed) {
            return None;
        }

        let active = self.active_blocks.load(Ordering::Relaxed);
        if active >= self.max_blocks {
            return None;
        }

        // 检查内存使用率
        let memory_usage = active as f64 / self.max_blocks as f64;
        if memory_usage >= self.memory_warning_threshold {
            let now = Instant::now();
            if now.duration_since(self.last_warning) >= self.warning_cooldown {
                warn!("内存池使用率过高: {:.1}%", memory_usage * 100.0);
                self.last_warning = now;
            }
        }

        if self.last_cleanup.elapsed() >= self.cleanup_interval {
            self.cleanup();
        }

        self.active_blocks.fetch_add(1, Ordering::Relaxed);
        self.blocks.iter_mut().next()
    }

    pub fn return_block(&mut self, block: Vec<u8>) {
        if self.is_shutting_down.load(Ordering::Relaxed) {
            return;
        }

        if self.blocks.len() < self.max_blocks {
            self.blocks.push(block);
            self.active_blocks.fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn cleanup(&mut self) {
        let active = self.active_blocks.load(Ordering::Relaxed);
        if active < (self.max_blocks as f64 * self.shrink_threshold) as usize {
            self.blocks.truncate(active);
            self.blocks.shrink_to_fit();
            debug!("内存池已清理，当前大小: {}", self.blocks.len());
        }
        self.last_cleanup = Instant::now();
    }

    pub fn shutdown(&self) {
        self.is_shutting_down.store(true, Ordering::Relaxed);
    }

    pub fn get_memory_usage(&self) -> f64 {
        self.active_blocks.load(Ordering::Relaxed) as f64 / self.max_blocks as f64
    }
}

#[derive(Debug)]
pub enum MiningError {
    MemoryPoolError(String),
    KernelError(String),
    RuntimeError(String),
    ShutdownError(String),
    ResourceError(String),
}

impl std::fmt::Display for MiningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MiningError::MemoryPoolError(msg) => write!(f, "内存池错误: {}", msg),
            MiningError::KernelError(msg) => write!(f, "内核错误: {}", msg),
            MiningError::RuntimeError(msg) => write!(f, "运行时错误: {}", msg),
            MiningError::ShutdownError(msg) => write!(f, "关闭错误: {}", msg),
            MiningError::ResourceError(msg) => write!(f, "资源错误: {}", msg),
        }
    }
}

impl std::error::Error for MiningError {}

pub struct PerformanceMonitor {
    start_time: Instant,
    proof_count: AtomicU64,
    metrics: Arc<Mutex<HashMap<String, f64>>>,
    last_report_time: Instant,
    report_interval: Duration,
    memory_usage: AtomicU64,
    error_count: AtomicU64,
    success_count: AtomicU64,
    metrics_enabled: bool,
    error_logging_enabled: bool,
    hash_compute_time: AtomicU64,
    hash_compute_count: AtomicU64,
    memory_alloc_count: AtomicU64,
    memory_free_count: AtomicU64,
    thread_utilization: AtomicU64,
    thread_count: AtomicU64,
    last_thread_check: Instant,
    thread_check_interval: Duration,
    error_types: Arc<Mutex<HashMap<String, u64>>>,
    resource_usage: Arc<Mutex<HashMap<String, u64>>>,
}

impl PerformanceMonitor {
    pub fn new(report_interval_secs: u64, metrics_enabled: bool, error_logging_enabled: bool) -> Self {
        Self {
            start_time: Instant::now(),
            proof_count: AtomicU64::new(0),
            metrics: Arc::new(Mutex::new(HashMap::new())),
            last_report_time: Instant::now(),
            report_interval: Duration::from_secs(report_interval_secs),
            memory_usage: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            metrics_enabled,
            error_logging_enabled,
            hash_compute_time: AtomicU64::new(0),
            hash_compute_count: AtomicU64::new(0),
            memory_alloc_count: AtomicU64::new(0),
            memory_free_count: AtomicU64::new(0),
            thread_utilization: AtomicU64::new(0),
            thread_count: AtomicU64::new(0),
            last_thread_check: Instant::now(),
            thread_check_interval: Duration::from_secs(5),
            error_types: Arc::new(Mutex::new(HashMap::new())),
            resource_usage: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn record_error_type(&self, error_type: &str) {
        if !self.error_logging_enabled {
            return;
        }
        let mut error_types = self.error_types.lock().unwrap();
        *error_types.entry(error_type.to_string()).or_insert(0) += 1;
    }

    pub fn record_resource_usage(&self, resource: &str, amount: u64) {
        if !self.metrics_enabled {
            return;
        }
        let mut resource_usage = self.resource_usage.lock().unwrap();
        *resource_usage.entry(resource.to_string()).or_insert(0) += amount;
    }

    pub fn get_performance_report(&self) -> String {
        let duration = self.start_time.elapsed();
        let proofs = self.proof_count.load(Ordering::Relaxed);
        let proofs_per_second = proofs as f64 / duration.as_secs_f64();
        let memory_usage_mb = self.memory_usage.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0);
        let errors = self.error_count.load(Ordering::Relaxed);
        let successes = self.success_count.load(Ordering::Relaxed);
        let total_attempts = errors + successes;
        let success_rate = if total_attempts > 0 {
            (successes as f64 / total_attempts as f64) * 100.0
        } else {
            0.0
        };

        let hash_compute_time = self.hash_compute_time.load(Ordering::Relaxed);
        let hash_compute_count = self.hash_compute_count.load(Ordering::Relaxed);
        let avg_hash_time = if hash_compute_count > 0 {
            hash_compute_time as f64 / hash_compute_count as f64
        } else {
            0.0
        };

        let memory_alloc_count = self.memory_alloc_count.load(Ordering::Relaxed);
        let memory_free_count = self.memory_free_count.load(Ordering::Relaxed);
        let memory_leak = memory_alloc_count.saturating_sub(memory_free_count);

        let thread_utilization = self.thread_utilization.load(Ordering::Relaxed);
        let thread_count = self.thread_count.load(Ordering::Relaxed);

        let error_types = self.error_types.lock().unwrap();
        let resource_usage = self.resource_usage.lock().unwrap();
        let metrics = self.metrics.lock().unwrap();

        let mut report = format!(
            "运行时间: {:?}\n总证明数: {}\n每秒证明数: {:.2}\n内存使用: {:.2} MB\n成功率: {:.2}%\n错误数: {}\n成功数: {}\n平均哈希计算时间: {:.2} ns\n内存分配次数: {}\n内存释放次数: {}\n内存泄漏: {}\n线程数: {}\n线程利用率: {:.1}%\n",
            duration, 
            proofs, 
            proofs_per_second,
            memory_usage_mb,
            success_rate,
            errors,
            successes,
            avg_hash_time,
            memory_alloc_count,
            memory_free_count,
            memory_leak,
            thread_count,
            thread_utilization as f64 / 100.0
        );

        report.push_str("\n错误类型统计:\n");
        for (error_type, count) in error_types.iter() {
            report.push_str(&format!("{}: {}\n", error_type, count));
        }

        report.push_str("\n资源使用统计:\n");
        for (resource, amount) in resource_usage.iter() {
            report.push_str(&format!("{}: {}\n", resource, amount));
        }

        report.push_str("\n其他指标:\n");
        for (name, value) in metrics.iter() {
            report.push_str(&format!("{}: {:.2}\n", name, value));
        }

        report
    }

    pub fn get_resource_usage(&self, resource: &str) -> u64 {
        let resource_usage = self.resource_usage.lock().unwrap();
        *resource_usage.get(resource).unwrap_or(&0)
    }

    pub fn should_report(&self) -> bool {
        self.last_report_time.elapsed() >= self.report_interval
    }
}

pub struct MiningManager {
    num_threads: usize,
    proof_counter: Arc<AtomicU64>,
    start_time: Instant,
    thread_pool: rayon::ThreadPool,
    performance_monitor: Arc<PerformanceMonitor>,
    max_retries: usize,
    retry_delay: Duration,
    memory_pool_size: usize,
    runtime: Runtime,
    is_shutting_down: Arc<AtomicBool>,
    cleanup_interval: Duration,
    shrink_threshold: f64,
    task_tracker: Arc<Mutex<HashMap<String, Instant>>>,
    resource_limits: Arc<Mutex<HashMap<String, u64>>>,
    task_timeout: Duration,
}

impl MiningManager {
    pub fn new(config: Option<&NockchainCli>) -> Self {
        let num_threads = if let Some(cfg) = config {
            if cfg.mining_threads > 0 {
                cfg.mining_threads
            } else {
                // 对于16核CPU，使用15个线程
                15
            }
        } else {
            15
        };
        
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .stack_size(8 * 1024 * 1024)  // 增加每个线程的栈大小到8MB
            .build()
            .unwrap();

        let performance_monitor = Arc::new(PerformanceMonitor::new(
            config.map(|c| c.mining_report_interval).unwrap_or(60),
            config.map(|c| c.mining_metrics_enabled).unwrap_or(true),
            config.map(|c| c.mining_error_logging).unwrap_or(true)
        ));

        let max_retries = config.map(|c| c.mining_max_retries).unwrap_or(3);
        let retry_delay = Duration::from_secs(
            config.map(|c| c.mining_retry_delay).unwrap_or(1)
        );
        // 为16核CPU优化内存池大小
        let memory_pool_size = config.map(|c| c.mining_memory_pool).unwrap_or(8192);  // 默认8GB
        let cleanup_interval = Duration::from_secs(
            config.map(|c| c.mining_cleanup_interval).unwrap_or(300)
        );
        let shrink_threshold = config.map(|c| c.mining_memory_shrink_threshold).unwrap_or(0.5);
        let task_timeout = Duration::from_secs(30);

        let mut resource_limits = HashMap::new();
        resource_limits.insert("memory_blocks".to_string(), memory_pool_size as u64);
        resource_limits.insert("hash_computations".to_string(), 10000);  // 增加哈希计算限制
        resource_limits.insert("successful_proofs".to_string(), 1000);   // 增加成功证明限制

        Self {
            num_threads,
            proof_counter: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            thread_pool,
            performance_monitor,
            max_retries,
            retry_delay,
            memory_pool_size,
            runtime: Runtime::new().expect("Failed to create Tokio runtime"),
            is_shutting_down: Arc::new(AtomicBool::new(false)),
            cleanup_interval,
            shrink_threshold,
            task_tracker: Arc::new(Mutex::new(HashMap::new())),
            resource_limits: Arc::new(Mutex::new(resource_limits)),
            task_timeout,
        }
    }

    pub fn track_task(&self, task_id: String) {
        let mut tracker = self.task_tracker.lock().unwrap();
        tracker.insert(task_id, Instant::now());
    }

    pub fn check_task_timeout(&self, task_id: &str) -> bool {
        let tracker = self.task_tracker.lock().unwrap();
        if let Some(start_time) = tracker.get(task_id) {
            start_time.elapsed() > self.task_timeout
        } else {
            false
        }
    }

    pub fn remove_task(&self, task_id: &str) {
        let mut tracker = self.task_tracker.lock().unwrap();
        tracker.remove(task_id);
    }

    pub fn check_resource_limit(&self, resource: &str, amount: u64) -> bool {
        let limits = self.resource_limits.lock().unwrap();
        if let Some(limit) = limits.get(resource) {
            let usage = self.performance_monitor.get_resource_usage(resource);
            usage + amount <= *limit
        } else {
            true
        }
    }

    pub fn get_performance_report(&self) -> String {
        self.performance_monitor.get_performance_report()
    }

    pub fn shutdown(&self) {
        self.is_shutting_down.store(true, Ordering::Relaxed);
    }
}

pub fn create_mining_driver(
    mining_config: Option<Vec<MiningKeyConfig>>,
    mine: bool,
    init_complete_tx: Option<tokio::sync::oneshot::Sender<()>>,
    cli_config: Option<&NockchainCli>,
) -> IODriverFn {
    Box::new(move |mut handle| {
        Box::pin(async move {
            let Some(configs) = mining_config else {
                enable_mining(&handle, false).await?;

                if let Some(tx) = init_complete_tx {
                    tx.send(()).map_err(|_| {
                        warn!("Could not send driver initialization for mining driver.");
                        NockAppError::OtherError
                    })?;
                }

                return Ok(());
            };
            if configs.len() == 1
                && configs[0].share == 1
                && configs[0].m == 1
                && configs[0].keys.len() == 1
            {
                set_mining_key(&handle, configs[0].keys[0].clone()).await?;
            } else {
                set_mining_key_advanced(&handle, configs).await?;
            }
            enable_mining(&handle, mine).await?;

            if let Some(tx) = init_complete_tx {
                tx.send(()).map_err(|_| {
                    warn!("Could not send driver initialization for mining driver.");
                    NockAppError::OtherError
                })?;
            }

            if !mine {
                return Ok(());
            }
            let mut next_attempt: Option<NounSlab> = None;
            let mut current_attempt: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();

            loop {
                tokio::select! {
                    effect_res = handle.next_effect() => {
                        let Ok(effect) = effect_res else {
                          warn!("Error receiving effect in mining driver: {effect_res:?}");
                        continue;
                        };
                        let Ok(effect_cell) = (unsafe { effect.root().as_cell() }) else {
                            drop(effect);
                            continue;
                        };

                        if effect_cell.head().eq_bytes("mine") {
                            let candidate_slab = {
                                let mut slab = NounSlab::new();
                                slab.copy_into(effect_cell.tail());
                                slab
                            };
                            if !current_attempt.is_empty() {
                                next_attempt = Some(candidate_slab);
                            } else {
                                let (cur_handle, attempt_handle) = handle.dup();
                                handle = cur_handle;
                                current_attempt.spawn(mining_attempt(candidate_slab, attempt_handle, cli_config));
                            }
                        }
                    },
                    mining_attempt_res = current_attempt.join_next(), if !current_attempt.is_empty()  => {
                        if let Some(Err(e)) = mining_attempt_res {
                            warn!("Error during mining attempt: {e:?}");
                        }
                        let Some(candidate_slab) = next_attempt else {
                            continue;
                        };
                        next_attempt = None;
                        let (cur_handle, attempt_handle) = handle.dup();
                        handle = cur_handle;
                        current_attempt.spawn(mining_attempt(candidate_slab, attempt_handle, cli_config));
                    }
                }
            }
        })
    })
}

pub async fn mining_attempt(candidate: NounSlab, handle: NockAppHandle, config: Option<&NockchainCli>) -> () {
    let mining_manager = Arc::new(MiningManager::new(config));
    let memory_pool = Arc::new(std::sync::Mutex::new(MemoryPool::new(
        mining_manager.memory_pool_size * 1024 * 1024,
        mining_manager.num_threads,
        mining_manager.cleanup_interval,
        mining_manager.shrink_threshold
    )));
    
    let snapshot_dir = match tokio::task::spawn_blocking(|| tempdir()).await {
        Ok(Ok(dir)) => dir,
        Ok(Err(e)) => {
            error!("创建临时目录失败: {}", e);
            mining_manager.performance_monitor.record_error_type("tempdir_creation");
            return;
        }
        Err(e) => {
            error!("生成阻塞任务失败: {}", e);
            mining_manager.performance_monitor.record_error_type("task_spawn");
            return;
        }
    };
    
    let hot_state = zkvm_jetpack::hot::produce_prover_hot_state();
    let snapshot_path_buf = snapshot_dir.path().to_path_buf();
    let jam_paths = JamPaths::new(snapshot_dir.path());

    let kernel = match Kernel::load_with_hot_state_huge(
        snapshot_path_buf.clone(),
        jam_paths.clone(),
        KERNEL,
        &hot_state,
        false
    ).await {
        Ok(k) => k,
        Err(e) => {
            error!("加载挖矿内核失败: {}", e);
            mining_manager.performance_monitor.record_error_type("kernel_load");
            return;
        }
    };

    let active_threads = Arc::new(AtomicUsize::new(0));
    let total_threads = mining_manager.num_threads;

    mining_manager.thread_pool.install(|| {
        (0..mining_manager.num_threads).into_par_iter().for_each(|thread_id| {
            if mining_manager.is_shutting_down.load(Ordering::Relaxed) {
                return;
            }

            let task_id = format!("mining_thread_{}", thread_id);
            mining_manager.track_task(task_id.clone());

            // 设置线程优先级
            #[cfg(target_os = "linux")]
            unsafe {
                use libc::{sched_param, sched_setscheduler, SCHED_RR};
                let param = sched_param { sched_priority: 99 };
                sched_setscheduler(0, SCHED_RR, &param);
            }

            active_threads.fetch_add(1, Ordering::Relaxed);
            let start_time = Instant::now();
            let mut retry_count = 0;
            
            while retry_count < mining_manager.max_retries {
                if mining_manager.is_shutting_down.load(Ordering::Relaxed) {
                    break;
                }

                if mining_manager.check_task_timeout(&task_id) {
                    warn!("任务超时: {}", task_id);
                    mining_manager.performance_monitor.record_error_type("task_timeout");
                    break;
                }

                let memory_block = match memory_pool.lock() {
                    Ok(mut pool) => match pool.get_block() {
                        Some(block) => {
                            if !mining_manager.check_resource_limit("memory_blocks", 1) {
                                error!("内存块资源限制已达到");
                                mining_manager.performance_monitor.record_error_type("resource_limit");
                                std::thread::sleep(mining_manager.retry_delay);
                                retry_count += 1;
                                continue;
                            }
                            mining_manager.performance_monitor.record_memory_alloc();
                            mining_manager.performance_monitor.record_resource_usage("memory_blocks", 1);
                            block.clone()
                        },
                        None => {
                            error!("从内存池获取内存块失败");
                            mining_manager.performance_monitor.record_error_type("memory_pool_get");
                            std::thread::sleep(mining_manager.retry_delay);
                            retry_count += 1;
                            continue;
                        }
                    },
                    Err(e) => {
                        error!("锁定内存池失败: {}", e);
                        mining_manager.performance_monitor.record_error_type("memory_pool_lock");
                        std::thread::sleep(mining_manager.retry_delay);
                        retry_count += 1;
                        continue;
                    }
                };
                
                let hash_start = Instant::now();
                let effects_slab = match kernel.poke(MiningWire::Candidate.to_wire(), candidate.clone()) {
                    Ok(slab) => {
                        if !mining_manager.check_resource_limit("hash_computations", 1) {
                            error!("哈希计算资源限制已达到");
                            mining_manager.performance_monitor.record_error_type("resource_limit");
                            std::thread::sleep(mining_manager.retry_delay);
                            retry_count += 1;
                            continue;
                        }
                        mining_manager.performance_monitor.record_hash_compute(hash_start.elapsed());
                        mining_manager.performance_monitor.record_resource_usage("hash_computations", 1);
                        slab
                    },
                    Err(e) => {
                        error!("调用挖矿内核失败: {}", e);
                        mining_manager.performance_monitor.record_error_type("kernel_poke");
                        std::thread::sleep(mining_manager.retry_delay);
                        retry_count += 1;
                        continue;
                    }
                };

                let mut success = false;
                for effect in effects_slab.to_vec() {
                    let Ok(effect_cell) = (unsafe { effect.root().as_cell() }) else {
                        drop(effect);
                        continue;
                    };
                    if effect_cell.head().eq_bytes("command") {
                        if let Err(e) = mining_manager.runtime.block_on(async {
                            handle
                                .poke(MiningWire::Mined.to_wire(), effect)
                                .await
                        }) {
                            error!("调用 nockchain 失败: {}", e);
                            mining_manager.performance_monitor.record_error_type("nockchain_poke");
                            continue;
                        }
                        if !mining_manager.check_resource_limit("successful_proofs", 1) {
                            error!("成功证明资源限制已达到");
                            mining_manager.performance_monitor.record_error_type("resource_limit");
                            continue;
                        }
                        mining_manager.proof_counter.fetch_add(1, Ordering::Relaxed);
                        mining_manager.performance_monitor.record_success();
                        mining_manager.performance_monitor.record_resource_usage("successful_proofs", 1);
                        success = true;
                    }
                }

                if success {
                    break;
                }

                if let Ok(mut pool) = memory_pool.lock() {
                    pool.return_block(memory_block);
                    mining_manager.performance_monitor.record_memory_free();
                    mining_manager.performance_monitor.record_resource_usage("memory_blocks", -1);
                }

                std::thread::sleep(mining_manager.retry_delay);
                retry_count += 1;
            }

            active_threads.fetch_sub(1, Ordering::Relaxed);
            mining_manager.remove_task(&task_id);
            mining_manager.performance_monitor.update_thread_metrics(
                active_threads.load(Ordering::Relaxed),
                total_threads
            );
        });
    });

    if mining_manager.performance_monitor.should_report() {
        info!("{}", mining_manager.get_performance_report());
    }

    memory_pool.shutdown();
    mining_manager.shutdown();
}

#[instrument(skip(handle, pubkey))]
async fn set_mining_key(
    handle: &NockAppHandle,
    pubkey: String,
) -> Result<PokeResult, NockAppError> {
    let mut set_mining_key_slab = NounSlab::new();
    let set_mining_key = Atom::from_value(&mut set_mining_key_slab, "set-mining-key")
        .expect("Failed to create set-mining-key atom");
    let pubkey_cord =
        Atom::from_value(&mut set_mining_key_slab, pubkey).expect("Failed to create pubkey atom");
    let set_mining_key_poke = T(
        &mut set_mining_key_slab,
        &[D(tas!(b"command")), set_mining_key.as_noun(), pubkey_cord.as_noun()],
    );
    set_mining_key_slab.set_root(set_mining_key_poke);

    handle
        .poke(MiningWire::SetPubKey.to_wire(), set_mining_key_slab)
        .await
}

async fn set_mining_key_advanced(
    handle: &NockAppHandle,
    configs: Vec<MiningKeyConfig>,
) -> Result<PokeResult, NockAppError> {
    let mut set_mining_key_slab = NounSlab::new();
    let set_mining_key_adv = Atom::from_value(&mut set_mining_key_slab, "set-mining-key-advanced")
        .expect("Failed to create set-mining-key-advanced atom");

    // Create the list of configs
    let mut configs_list = D(0);
    for config in configs {
        // Create the list of keys
        let mut keys_noun = D(0);
        for key in config.keys {
            let key_atom =
                Atom::from_value(&mut set_mining_key_slab, key).expect("Failed to create key atom");
            keys_noun = T(&mut set_mining_key_slab, &[key_atom.as_noun(), keys_noun]);
        }

        // Create the config tuple [share m keys]
        let config_tuple = T(
            &mut set_mining_key_slab,
            &[D(config.share), D(config.m), keys_noun],
        );

        configs_list = T(&mut set_mining_key_slab, &[config_tuple, configs_list]);
    }

    let set_mining_key_poke = T(
        &mut set_mining_key_slab,
        &[D(tas!(b"command")), set_mining_key_adv.as_noun(), configs_list],
    );
    set_mining_key_slab.set_root(set_mining_key_poke);

    handle
        .poke(MiningWire::SetPubKey.to_wire(), set_mining_key_slab)
        .await
}

//TODO add %set-mining-key-multisig poke
#[instrument(skip(handle))]
async fn enable_mining(handle: &NockAppHandle, enable: bool) -> Result<PokeResult, NockAppError> {
    let mut enable_mining_slab = NounSlab::new();
    let enable_mining = Atom::from_value(&mut enable_mining_slab, "enable-mining")
        .expect("Failed to create enable-mining atom");
    let enable_mining_poke = T(
        &mut enable_mining_slab,
        &[D(tas!(b"command")), enable_mining.as_noun(), D(if enable { 0 } else { 1 })],
    );
    enable_mining_slab.set_root(enable_mining_poke);
    handle
        .poke(MiningWire::Enable.to_wire(), enable_mining_slab)
        .await
}
