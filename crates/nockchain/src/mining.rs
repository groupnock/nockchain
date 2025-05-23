// crates/nockchain/src/mining.rs

use std::{
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
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

// —————————————————————————————————————————————————————————————————————
// Constants & globals
// —————————————————————————————————————————————————————————————————————

const MINING_WIRE_VERSION: u64        = 1;
const MINING_WIRE_SOURCE:  &str       = "miner";
const MAX_MINING_THREADS: usize       = 16;
const MINING_RETRY_DELAY_MS: u64      = 100;
const CANDIDATE_FETCH_TIMEOUT_MS: u64 = 5_000;

// “Are we currently mining?” flag
static MINING_ACTIVE: Lazy<Arc<AtomicBool>> = Lazy::new(|| Arc::new(AtomicBool::new(false)));

// One-time caches for snapshot dir & loaded kernel
static SNAPSHOT_DIR: OnceCell<Arc<TempDir>> = OnceCell::new();
static STARK_KERNEL: OnceCell<Arc<Kernel>>   = OnceCell::new();

// —————————————————————————————————————————————————————————————————————
// Mining configuration type
// —————————————————————————————————————————————————————————————————————

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
            Err(NockAppError::OtherError)
        } else {
            Ok(&self.keys[0])
        }
    }
}

impl FromStr for MiningKeyConfig {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.splitn(2, ':');
        let sm_part   = parts.next().ok_or("Missing share,m")?;
        let keys_part = parts.next().ok_or("Missing keys")?;

        let mut sm = sm_part.splitn(2, ',');
        let share = sm.next().and_then(|s| s.parse().ok()).ok_or("Invalid share")?;
        let m     = sm.next().and_then(|s| s.parse().ok()).ok_or("Invalid m")?;

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

// —————————————————————————————————————————————————————————————————————
// Wire enum
// —————————————————————————————————————————————————————————————————————

#[derive(Debug, Clone, Copy)]
pub enum MiningWire {
    Mined,
    Candidate,
    SetPubKey,
    Enable,
}

impl MiningWire {
    pub fn verb(&self) -> &'static str {
        match self {
            MiningWire::Mined     => "mined",
            MiningWire::Candidate => "candidate",
            MiningWire::SetPubKey => "setpubkey",
            MiningWire::Enable    => "enable",
        }
    }
}

impl Wire for MiningWire {
    const VERSION: u64          = MINING_WIRE_VERSION;
    const SOURCE:  &'static str = MINING_WIRE_SOURCE;
    fn to_wire(&self) -> WireRepr {
        WireRepr::new(Self::SOURCE, Self::VERSION, vec![self.verb().into()])
    }
}

// —————————————————————————————————————————————————————————————————————
// Engine init: one tempdir + one kernel load
// —————————————————————————————————————————————————————————————————————

async fn init_engine() -> Result<Arc<Kernel>, NockAppError> {
    let snap = SNAPSHOT_DIR
        .get_or_init(|| Arc::new(TempDir::new().expect("failed to create snapshot dir")))
        .clone();

    if let Some(k) = STARK_KERNEL.get() {
        return Ok(k.clone());
    }

    let jams   = JamPaths::new(snap.path());
    let hot    = produce_prover_hot_state();
    let kernel = Kernel::load_with_hot_state_huge(snap.path().into(), jams, KERNEL, &hot, false)
        .await
        .map_err(|_| NockAppError::OtherError)?;

    let arc = Arc::new(kernel);
    STARK_KERNEL.set(arc.clone()).ok();
    Ok(arc)
}

// —————————————————————————————————————————————————————————————————————
// Create the mining I/O driver
// —————————————————————————————————————————————————————————————————————

pub fn create_mining_driver(
    mining_config: Option<Vec<MiningKeyConfig>>,
    mine: bool,
    init_complete_tx: Option<oneshot::Sender<()>>,
) -> IODriverFn {
    Box::new(move |handle| {
        Box::pin(async move {
            // 1) Init the engine
            let kernel = init_engine().await?;

            // 2) Configure keys & mining on-chain
            if let Some(cfgs) = &mining_config {
                if cfgs.len() == 1 && cfgs[0].is_simple() {
                    set_mining_key(&handle, cfgs[0].primary_key()?).await?;
                } else {
                    set_mining_key_advanced(&handle, cfgs).await?;
                }
                enable_mining(&handle, mine).await?;
                MINING_ACTIVE.store(mine, Ordering::SeqCst);
                info!("Mining {}", if mine { "enabled" } else { "disabled" });
            } else {
                enable_mining(&handle, false).await?;
                MINING_ACTIVE.store(false, Ordering::SeqCst);
                info!("Mining disabled");
            }

            // 3) Split handle into listener + poke_handle
            let (listener, poke_handle) = handle.dup();

            // 4) Channel for block candidates
            let (cand_tx, cand_rx) = mpsc::channel::<NounSlab>(MAX_MINING_THREADS);
            let cand_rx = Arc::new(Mutex::new(cand_rx));

            // 5) Spawn mining threads if mining is enabled
            if mine {
                start_mining_threads(poke_handle, kernel.clone(), cand_rx.clone());
            }

            // 6) Signal init complete
            if let Some(tx) = init_complete_tx {
                tx.send(()).map_err(|_| NockAppError::OtherError)?;
            }

            // 7) Driver event loop: handle candidate pokes
            loop {
                let effect = listener.next_effect().await?;
                if let Ok(cell) = unsafe { effect.root().as_cell() } {
                    if cell.head().eq_bytes("candidate") {
                        let mut slab = NounSlab::new();
                        slab.copy_into(cell.tail());
                        let _ = cand_tx.send(slab).await;
                    }
                }
            }
        })
    })
}

// —————————————————————————————————————————————————————————————————————
// Spawn a fixed pool of mining threads
// —————————————————————————————————————————————————————————————————————

fn start_mining_threads(
    mut handle: NockAppHandle,
    kernel:     Arc<Kernel>,
    cand_rx:    Arc<Mutex<mpsc::Receiver<NounSlab>>>,
) {
    let threads = num_cpus::get().min(MAX_MINING_THREADS);
    info!("Spawning {} miner threads", threads);

    for i in 0..threads {
        let (next_handle, thread_handle) = handle.dup();
        handle = next_handle;
        let kernel = kernel.clone();
        let active = MINING_ACTIVE.clone();
        let cand_rx = cand_rx.clone();

        thread::Builder::new()
            .name(format!("miner-{}", i))
            .spawn(move || {
                let rt = Runtime::new().expect("tokio RT");
                rt.block_on(async move {
                    info!("Miner thread {} up", i);
                    loop {
                        if !active.load(Ordering::SeqCst) {
                            info!("Miner thread {} stopping", i);
                            break;
                        }
                        // fetch candidate with timeout
                        let maybe = {
                            let mut rx = cand_rx.lock().await;
                            timeout(
                                Duration::from_millis(CANDIDATE_FETCH_TIMEOUT_MS),
                                rx.recv(),
                            )
                            .await
                            .ok()
                            .flatten()
                        };
                        if let Some(cand) = maybe {
                            if let Err(e) = attempt_mining_cycle(&thread_handle, &kernel, cand).await {
                                warn!("Thread {} error: {}", i, e);
                                tokio::time::sleep(Duration::from_millis(MINING_RETRY_DELAY_MS)).await;
                            }
                        }
                    }
                });
            })
            .expect("spawn thread");
    }
}

// —————————————————————————————————————————————————————————————————————
// One full mining cycle: proof + poke
// —————————————————————————————————————————————————————————————————————

#[instrument(skip(handle, kernel, candidate))]
async fn attempt_mining_cycle(
    handle:    &NockAppHandle,
    kernel:    &Arc<Kernel>,
    candidate: NounSlab,
) -> Result<(), NockAppError> {
    let effects = kernel.poke(MiningWire::Candidate.to_wire(), candidate).await?;
    for effect in effects.to_vec() {
        if let Ok(c) = unsafe { effect.root().as_cell() } {
            if c.head().eq_bytes("command") {
                handle.poke(MiningWire::Mined.to_wire(), effect).await?;
                info!("Mined block submitted");
                break;
            }
        }
    }
    Ok(())
}

// —————————————————————————————————————————————————————————————————————
// On-chain pokes: set key(s) & enable/disable
// —————————————————————————————————————————————————————————————————————

#[instrument(skip(handle))]
async fn set_mining_key(
    handle: &NockAppHandle,
    pubkey: &str,
) -> Result<PokeResult, NockAppError> {
    let mut slab = NounSlab::new();
    let cmd       = Atom::from_value(&mut slab, "set-mining-key")?;
    let pk        = Atom::from_value(&mut slab, pubkey)?;
    let body      = T(&mut slab, &[D(tas!(b"command")), cmd.as_noun(), pk.as_noun()]);
    slab.set_root(body);
    handle.poke(MiningWire::SetPubKey.to_wire(), slab).await
}

#[instrument(skip(handle, configs))]
async fn set_mining_key_advanced(
    handle:  &NockAppHandle,
    configs: &[MiningKeyConfig],
) -> Result<PokeResult, NockAppError> {
    let mut slab = NounSlab::new();
    let cmd_adv  = Atom::from_value(&mut slab, "set-mining-key-advanced")?;
    let mut list = D(0);
    for cfg in configs.iter().rev() {
        let mut keys = D(0);
        for k in cfg.keys.iter().rev() {
            let a    = Atom::from_value(&mut slab, k.clone())?;
            keys      = T(&mut slab, &[a.as_noun(), keys]);
        }
        let tup  = T(&mut slab, &[D(cfg.share), D(cfg.m), keys]);
        list      = T(&mut slab, &[tup, list]);
    }
    let body       = T(&mut slab, &[D(tas!(b"command")), cmd_adv.as_noun(), list]);
    slab.set_root(body);
    handle.poke(MiningWire::SetPubKey.to_wire(), slab).await
}

#[instrument(skip(handle))]
async fn enable_mining(
    handle: &NockAppHandle,
    enable: bool,
) -> Result<PokeResult, NockAppError> {
    let mut slab = NounSlab::new();
    let cmd       = Atom::from_value(&mut slab, "enable-mining")?;
    let flag      = if enable { 0 } else { 1 };
    let body      = T(&mut slab, &[D(tas!(b"command")), cmd.as_noun(), D(flag)]);
    slab.set_root(body);
    handle.poke(MiningWire::Enable.to_wire(), slab).await
}
