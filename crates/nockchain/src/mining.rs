// crates/nockchain/src/mining.rs

use std::{
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    path::PathBuf,
    thread,
    time::Duration,
};
use tempfile::TempDir;
use tokio::{
    runtime::Runtime,
    sync::{oneshot, mpsc, Mutex},
    time::timeout,
};
use tracing::{instrument, info, warn};
use once_cell::sync::{Lazy, OnceCell};
use num_cpus;

use kernels::miner::KERNEL;
use nockapp::{
    kernel::{checkpoint::JamPaths, form::Kernel},
    nockapp::{
        driver::{IODriverFn, NockAppHandle, PokeResult},
        wire::{Wire, WireRepr},
        NockAppError,
    },
    noun::{AtomExt, slab::NounSlab, NounExt},
};
use nockvm::noun::{Atom, D, T};
use nockvm_macros::tas;
use zkvm_jetpack::hot::produce_prover_hot_state;

// Constants
const MINING_WIRE_VERSION: u64   = 1;
const MINING_WIRE_SOURCE:  &str  = "miner";
const MAX_MINING_THREADS: usize = 16;
const MINING_RETRY_DELAY_MS: u64 = 100;
const CANDIDATE_FETCH_TIMEOUT_MS: u64 = 5_000;

// Global “are we mining?” flag
static MINING_ACTIVE: Lazy<Arc<AtomicBool>> = Lazy::new(|| Arc::new(AtomicBool::new(false)));

// One-time caches: snapshot dir & loaded kernel
static SNAPSHOT_DIR: OnceCell<Arc<TempDir>> = OnceCell::new();
static STARK_KERNEL: OnceCell<Arc<Kernel>>     = OnceCell::new();

/// Your key config
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiningKeyConfig {
    pub share: u64,
    pub m:     u64,
    pub keys:  Vec<String>,
}
impl MiningKeyConfig {
    pub fn is_simple(&self) -> bool {
        self.share == 1 && self.m == 1 && self.keys.len() == 1
    }
    pub fn primary_key(&self) -> Result<&str, NockAppError> {
        if !self.is_simple() {
            return Err(NockAppError::InvalidConfig(
                "Not a simple configuration".into(),
            ));
        }
        Ok(&self.keys[0])
    }
}
impl FromStr for MiningKeyConfig {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.splitn(2, ':');
        let sm_part   = parts.next().ok_or("Missing share,m")?;
        let keys_part = parts.next().ok_or("Missing keys")?;

        let mut sm = sm_part.splitn(2, ',');
        let share = sm
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or("Invalid share")?;
        let m = sm
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or("Invalid m")?;

        let keys = keys_part
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect::<Vec<_>>();
        if keys.is_empty() {
            return Err("At least one key required".into());
        }

        Ok(MiningKeyConfig { share, m, keys })
    }
}

/// The wires we use
#[derive(Debug, Clone, Copy)]
pub enum MiningWire {
    Mined,
    Candidate,
    SetPubKey,
    Enable,
    Admin,
}
impl MiningWire {
    pub fn verb(&self) -> &'static str {
        match self {
            MiningWire::Mined     => "mined",
            MiningWire::Candidate => "candidate",
            MiningWire::SetPubKey => "setpubkey",
            MiningWire::Enable    => "enable",
            MiningWire::Admin     => "admin",
        }
    }
}
impl Wire for MiningWire {
    const VERSION: u64        = MINING_WIRE_VERSION;
    const SOURCE:  &'static str = MINING_WIRE_SOURCE;
    fn to_wire(&self) -> WireRepr {
        WireRepr::new(Self::SOURCE, Self::VERSION, vec![self.verb().into()])
    }
}

/// Initialize & cache the snapshot directory + STARK kernel
async fn init_engine() -> Result<Arc<Kernel>, NockAppError> {
    let snap = SNAPSHOT_DIR
        .get_or_init(|| {
            Arc::new(
                TempDir::new()
                    .map_err(|e| NockAppError::InitializationFailed(format!("{e}")))?,
            )
        })
        .clone();

    if let Some(k) = STARK_KERNEL.get() {
        return Ok(k.clone());
    }

    let path = snap.path().into();
    let jams = JamPaths::new(snap.path());
    let hot  = produce_prover_hot_state();
    let k = Kernel::load_with_hot_state_huge(path, jams, KERNEL, &hot, false)
        .await
        .map_err(|e| NockAppError::InitializationFailed(format!("{e}")))?;
    let arc = Arc::new(k);
    STARK_KERNEL.set(arc.clone()).ok();
    Ok(arc)
}

/// Build the mining driver
pub fn create_mining_driver(
    mining_config: Option<Vec<MiningKeyConfig>>,
    mine: bool,
    init_complete_tx: Option<oneshot::Sender<()>>,
) -> IODriverFn {
    Box::new(move |mut handle| {
        Box::pin(async move {
            // 1) init engine
            let kernel = init_engine().await?;

            // 2) configure keys / mining
            match &mining_config {
                None => {
                    enable_mining(&handle, false).await?;
                    MINING_ACTIVE.store(false, Ordering::SeqCst);
                    info!("Mining disabled");
                }
                Some(cfgs) => {
                    configure_mining(&handle, cfgs).await?;
                    if mine {
                        MINING_ACTIVE.store(true, Ordering::SeqCst);
                    }
                    info!("Mining configured with {} key sets", cfgs.len());
                }
            }

            // 3) subscribe to candidate & admin wires
            let mut candidate_rx = handle
                .subscribe(MiningWire::Candidate.to_wire())
                .await?;
            let mut admin_rx = handle
                .subscribe(MiningWire::Admin.to_wire())
                .await?;

            // 4) spawn miner threads if requested
            if mine {
                let pool = Arc::new(Mutex::new(candidate_rx));
                start_mining_threads(handle.dup(), kernel.clone(), pool.clone());
            }

            // 5) signal init complete
            if let Some(tx) = init_complete_tx {
                tx.send(()).map_err(|_| NockAppError::ChannelSendFailed)?;
            }

            // 6) driver event loop: handle admin commands
            loop {
                tokio::select! {
                    // admin pokes come in on the admin wire
                    Some(cmd_slab) = admin_rx.recv() => {
                        // assume command is an atom
                        if let Ok(cmd_atom) = unsafe { cmd_slab.root().as_atom() } {
                            let cmd_str = String::from_utf8_lossy(cmd_atom.as_bytes());
                            handle_admin_command(&handle, &cmd_str).await?;
                        }
                    }
                    else => {
                        // no other work: just yield
                        tokio::task::yield_now().await;
                    }
                }
            }
        })
    })
}

/// Launches a fixed-size pool of mining threads
fn start_mining_threads(
    handle: NockAppHandle,
    kernel: Arc<Kernel>,
    candidate_rx: Arc<Mutex<mpsc::Receiver<NounSlab>>>,
) {
    let threads = num_cpus::get().min(MAX_MINING_THREADS);
    info!("Spawning {} miner threads", threads);

    for i in 0..threads {
        let mut handle = handle.dup();
        let kernel = kernel.clone();
        let active = MINING_ACTIVE.clone();
        let cand_rx = candidate_rx.clone();

        thread::Builder::new()
            .name(format!("miner-{}", i))
            .spawn(move || {
                let rt = Runtime::new().expect("make tokio rt");
                rt.block_on(async move {
                    info!("Miner thread {} up", i);
                    loop {
                        if !active.load(Ordering::SeqCst) {
                            info!("Miner thread {} shutting down", i);
                            break;
                        }

                        // fetch next candidate with timeout
                        let maybe_cand = {
                            let mut rx = cand_rx.lock().await;
                            timeout(
                                Duration::from_millis(CANDIDATE_FETCH_TIMEOUT_MS),
                                rx.recv(),
                            ).await
                        };
                        match maybe_cand {
                            Ok(Some(cand)) => {
                                // got a candidate: try to mine it
                                if let Err(e) = attempt_mining_cycle(&handle, &kernel, cand).await {
                                    warn!("Thread {} mining error: {}", i, e);
                                    tokio::time::sleep(Duration::from_millis(MINING_RETRY_DELAY_MS)).await;
                                }
                            }
                            Ok(None) => {
                                // channel closed
                                warn!("Thread {} candidate channel closed", i);
                                break;
                            }
                            Err(_) => {
                                // timeout waiting for candidate
                                continue;
                            }
                        }
                    }
                });
            })
            .expect("failed to spawn miner thread");
    }
}

/// One full mining cycle from candidate → proof → submission
#[instrument(skip(handle, kernel, candidate))]
async fn attempt_mining_cycle(
    handle: &NockAppHandle,
    kernel: &Arc<Kernel>,
    candidate: NounSlab,
) -> Result<(), NockAppError> {
    // run kernel
    let effects = kernel.poke(MiningWire::Candidate.to_wire(), candidate).await?;
    for effect in effects.to_vec() {
        if let Ok(cell) = unsafe { effect.root().as_cell() } {
            if cell.head().eq_bytes("command") {
                // submit it
                submit_mined_block(handle, effect).await?;
                break;
            }
        }
    }
    Ok(())
}

/// Submit a mined block
async fn submit_mined_block(
    handle: &NockAppHandle,
    block: NounSlab,
) -> Result<(), NockAppError> {
    handle.poke(MiningWire::Mined.to_wire(), block).await?;
    info!("Mined block submitted");
    Ok(())
}

/// Configure basic or advanced key setup
#[instrument(skip(handle, configs))]
async fn configure_mining(
    handle: &NockAppHandle,
    configs: &[MiningKeyConfig],
) -> Result<(), NockAppError> {
    if configs.len() == 1 && configs[0].is_simple() {
        set_mining_key(handle, configs[0].primary_key()?).await
    } else {
        set_mining_key_advanced(handle, configs).await
    }?;
    enable_mining(handle, true).await
}

/// set-mining-key poke
#[instrument(skip(handle, pubkey))]
async fn set_mining_key(
    handle: &NockAppHandle,
    pubkey: &str,
) -> Result<PokeResult, NockAppError> {
    let mut slab = NounSlab::new();
    let cmd = Atom::from_value(&mut slab, "set-mining-key")?;
    let pk  = Atom::from_value(&mut slab, pubkey)?;
    let poke = T(&mut slab, &[D(tas!(b"command")), cmd.as_noun(), pk.as_noun()]);
    slab.set_root(poke);
    handle.poke(MiningWire::SetPubKey.to_wire(), slab).await
}

/// set-mining-key-advanced poke
#[instrument(skip(handle, configs))]
async fn set_mining_key_advanced(
    handle: &NockAppHandle,
    configs: &[MiningKeyConfig],
) -> Result<PokeResult, NockAppError> {
    let mut slab = NounSlab::new();
    let cmd_adv  = Atom::from_value(&mut slab, "set-mining-key-advanced")?;
    let mut list = D(0);
    for cfg in configs.iter().rev() {
        let mut keys = D(0);
        for k in cfg.keys.iter().rev() {
            let a = Atom::from_value(&mut slab, k)?;
            keys = T(&mut slab, &[a.as_noun(), keys]);
        }
        let tup = T(&mut slab, &[D(cfg.share), D(cfg.m), keys]);
        list    = T(&mut slab, &[tup, list]);
    }
    let poke = T(&mut slab, &[D(tas!(b"command")), cmd_adv.as_noun(), list]);
    slab.set_root(poke);
    handle.poke(MiningWire::SetPubKey.to_wire(), slab).await
}

/// enable-mining poke
#[instrument(skip(handle))]
async fn enable_mining(
    handle: &NockAppHandle,
    enable: bool,
) -> Result<PokeResult, NockAppError> {
    let mut slab = NounSlab::new();
    let cmd      = Atom::from_value(&mut slab, "enable-mining")?;
    let flag     = if enable { 0 } else { 1 };
    let poke     = T(&mut slab, &[D(tas!(b"command")), cmd.as_noun(), D(flag)]);
    slab.set_root(poke);
    handle.poke(MiningWire::Enable.to_wire(), slab).await
}

/// Handle admin commands like “disable-mining”
#[instrument(skip(handle, command))]
async fn handle_admin_command(
    handle:  &NockAppHandle,
    command: &str,
) -> Result<(), NockAppError> {
    match command {
        "disable-mining" => {
            MINING_ACTIVE.store(false, Ordering::SeqCst);
            enable_mining(handle, false).await?;
            info!("Mining disabled by admin");
            Ok(())
        }
        other => Err(NockAppError::InvalidCommand(other.to_string())),
    }
}
