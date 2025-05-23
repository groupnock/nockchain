// crates/nockchain/src/mining.rs

use std::{str::FromStr, sync::Arc, path::PathBuf, thread};
use tempfile::TempDir;
use tokio::runtime::{Builder, Runtime};
use tokio::sync::oneshot;
use tracing::{instrument, warn};
use once_cell::sync::OnceCell;
use num_cpus;

use kernels::miner::KERNEL;
use nockapp::kernel::checkpoint::JamPaths;
use nockapp::kernel::form::Kernel;
use nockapp::nockapp::driver::{IODriverFn, NockAppHandle, PokeResult};
use nockapp::nockapp::wire::Wire;
use nockapp::nockapp::NockAppError;
use nockapp::noun::{AtomExt, slab::NounSlab};
use nockapp::NounExt;                // for eq_bytes
use nockvm::noun::{Atom, D, T};
use nockvm_macros::tas;
use zkvm_jetpack::hot::produce_prover_hot_state;

// —————————————————————————————
// Global one-time caches
// —————————————————————————————
static SNAPSHOT_DIR: OnceCell<Arc<TempDir>> = OnceCell::new();
static STARK_KERNEL: OnceCell<Arc<Kernel>>  = OnceCell::new();

/// Async init of snapshot dir & STARK kernel
async fn init_engine() -> Arc<Kernel> {
    let snap = SNAPSHOT_DIR
        .get_or_init(|| Arc::new(TempDir::new().expect("failed to create snapshot dir")))
        .clone();

    if let Some(k) = STARK_KERNEL.get() {
        return k.clone();
    }

    let path: PathBuf = snap.path().into();
    let jams = JamPaths::new(snap.path());
    let hot   = produce_prover_hot_state();
    let kernel = Kernel::load_with_hot_state_huge(path, jams, KERNEL, &hot, false)
        .await
        .expect("kernel load failed");
    let arc_k = Arc::new(kernel);
    STARK_KERNEL.set(arc_k.clone()).expect("kernel set only once");
    arc_k
}

/// Builds the mining I/O driver function
pub fn create_mining_driver(
    mining_config: Option<Vec<MiningKeyConfig>>,
    mine: bool,
    init_complete_tx: Option<oneshot::Sender<()>>,
) -> IODriverFn {
    Box::new(move |mut handle| {
        Box::pin(async move {
            // 1) initialize kernel & snapshot
            let kernel = init_engine().await;

            // 2) configure mining key(s) or disable
            if let Some(cfgs) = &mining_config {
                if cfgs.len() == 1
                    && cfgs[0].share == 1
                    && cfgs[0].m == 1
                    && cfgs[0].keys.len() == 1
                {
                    set_mining_key(&handle, cfgs[0].keys[0].clone()).await?;
                } else {
                    set_mining_key_advanced(&handle, cfgs.clone()).await?;
                }
            } else {
                enable_mining(&handle, false).await?;
            }

            // 3) signal init complete
            if let Some(tx) = init_complete_tx {
                tx.send(()).map_err(|_| {
                    warn!("init tx failed");
                    NockAppError::OtherError
                })?;
            }

            // 4) if not mining, exit
            if !mine {
                return Ok(());
            }
            enable_mining(&handle, true).await?;

            // 5) main loop: spawn one thread per core on each mine-effect
            loop {
                let effect = match handle.next_effect().await {
                    Ok(e) => e,
                    Err(e) => {
                        warn!("error receiving effect: {:?}", e);
                        continue;
                    }
                };
                if let Ok(cell) = unsafe { effect.root().as_cell() } {
                    if cell.head().eq_bytes("mine") {
                        let tail   = cell.tail();
                        let cores  = num_cpus::get();
                        for _ in 0..cores {
                            let mut slab = NounSlab::new();
                            slab.copy_into(tail);

                            // split handle for each thread
                            let (new_listener, poke_handle) = handle.dup();
                            handle = new_listener;
                            let kernel_clone = kernel.clone();

                            thread::spawn(move || {
                                let rt = Builder::new_current_thread()
                                    .enable_all()
                                    .build()
                                    .unwrap();
                                rt.block_on(async move {
                                    mining_attempt(slab, poke_handle, kernel_clone).await;
                                });
                            });
                        }
                    }
                }
            }
        })
    })
}

/// Execute one proof attempt
pub async fn mining_attempt(
    candidate: NounSlab,
    handle:    NockAppHandle,
    kernel:    Arc<Kernel>,
) {
    let effects = kernel
        .poke(MiningWire::Candidate.to_wire(), candidate)
        .await
        .expect("poke candidate failed");

    for effect in effects.to_vec() {
        if let Ok(c) = unsafe { effect.root().as_cell() } {
            if c.head().eq_bytes("command") {
                handle
                    .poke(MiningWire::Mined.to_wire(), effect)
                    .await
                    .expect("poke mined failed");
            }
        }
    }
}

#[instrument(skip(handle, pubkey))]
async fn set_mining_key(
    handle: &NockAppHandle,
    pubkey: String,
) -> Result<PokeResult, NockAppError> {
    let mut slab  = NounSlab::new();
    let cmd       = Atom::from_value(&mut slab, "set-mining-key").unwrap();
    let pk        = Atom::from_value(&mut slab, pubkey).unwrap();
    let poke      = T(&mut slab, &[D(tas!(b"command")), cmd.as_noun(), pk.as_noun()]);
    slab.set_root(poke);
    handle.poke(MiningWire::SetPubKey.to_wire(), slab).await
}

async fn set_mining_key_advanced(
    handle:  &NockAppHandle,
    configs: Vec<MiningKeyConfig>,
) -> Result<PokeResult, NockAppError> {
    let mut slab  = NounSlab::new();
    let cmd_adv   = Atom::from_value(&mut slab, "set-mining-key-advanced").unwrap();
    let mut list  = D(0);

    for cfg in configs {
        let mut keys = D(0);
        for k in cfg.keys {
            let a = Atom::from_value(&mut slab, k).unwrap();
            keys = T(&mut slab, &[a.as_noun(), keys]);
        }
        let tup   = T(&mut slab, &[D(cfg.share), D(cfg.m), keys]);
        list       = T(&mut slab, &[tup, list]);
    }

    let poke = T(&mut slab, &[D(tas!(b"command")), cmd_adv.as_noun(), list]);
    slab.set_root(poke);
    handle.poke(MiningWire::SetPubKey.to_wire(), slab).await
}

#[instrument(skip(handle))]
async fn enable_mining(
    handle: &NockAppHandle,
    enable: bool,
) -> Result<PokeResult, NockAppError> {
    let mut slab  = NounSlab::new();
    let cmd       = Atom::from_value(&mut slab, "enable-mining").unwrap();
    let flag      = if enable { 0 } else { 1 };
    let poke      = T(&mut slab, &[D(tas!(b"command")), cmd.as_noun(), D(flag)]);
    slab.set_root(poke);
    handle.poke(MiningWire::Enable.to_wire(), slab).await
}

#[derive(Debug, Clone)]
pub struct MiningKeyConfig {
    pub share: u64,
    pub m:     u64,
    pub keys:  Vec<String>,
}

impl FromStr for MiningKeyConfig {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err("invalid format; expected share,m:key1,key2".into());
        }
        let sm = parts[0].split(',').collect::<Vec<_>>();
        if sm.len() != 2 {
            return Err("invalid share,m".into());
        }
        let share = sm[0]
            .parse::<u64>()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        let m     = sm[1]
            .parse::<u64>()
            .map_err(|e: std::num::ParseIntError| e.to_string())?;
        let keys  = parts[1].split(',').map(String::from).collect();
        Ok(MiningKeyConfig { share, m, keys })
    }
}

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
    const VERSION: u64         = 1;
    const SOURCE:  &'static str = "miner";
    fn to_wire(&self) -> nockapp::wire::WireRepr {
        let tags = vec![self.verb().into()];
        nockapp::wire::WireRepr::new(Self::SOURCE, Self::VERSION, tags)
    }
}
