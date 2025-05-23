use std::sync::Arc;
use std::str::FromStr;
use rayon::prelude::*;
use tokio::sync::{mpsc, Semaphore};
use thiserror::Error;
use tracing::{debug, error, instrument, warn};

use nockapp::nockapp::driver::{IODriverFn, NockAppHandle, PokeResult};
use nockapp::nockapp::wire::{Wire, WireRepr};
use nockapp::nockapp::NockAppError;
use nockapp::noun::slab::NounSlab;
use nockapp::noun::{AtomExt, NounExt};
use nockvm::noun::{Atom, D, T, Noun};
use nockvm_macros::tas;

#[derive(Debug, Error)]
pub enum MiningError {
    #[error("Invalid mining configuration format: {0}")]
    ConfigFormat(String),
    #[error("Invalid share/m value: {0}")]
    InvalidShareM(String),
    #[error("Empty keys list")]
    EmptyKeys,
    #[error("Driver initialization failed")]
    DriverInitFailed,
    #[error("Noun construction error")]
    NounConstruction,
    #[error("Mining computation failed")]
    MiningFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MiningKeyConfig {
    pub share: u64,
    pub m: u64,
    pub keys: Vec<String>,
}

impl FromStr for MiningKeyConfig {
    type Err = MiningError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(MiningError::ConfigFormat(
                "Expected 'share,m:key1,key2,key3'".into(),
            ));
        }

        let share_m: Vec<&str> = parts[0].splitn(2, ',').collect();
        if share_m.len() != 2 {
            return Err(MiningError::ConfigFormat("Expected share,m format".into()));
        }

        let share = share_m[0]
            .parse::<u64>()
            .map_err(|e| MiningError::InvalidShareM(e.to_string()))?;
        let m = share_m[1]
            .parse::<u64>()
            .map_err(|e| MiningError::InvalidShareM(e.to_string()))?;

        let keys_str = parts[1].trim();
        if keys_str.is_empty() {
            return Err(MiningError::EmptyKeys);
        }

        let keys: Vec<String> = keys_str
            .split(',')
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
            .collect();

        if keys.is_empty() {
            return Err(MiningError::EmptyKeys);
        }

        if m == 0 || m > keys.len() as u64 {
            return Err(MiningError::InvalidShareM(format!(
                "m value {} exceeds number of keys {}",
                m,
                keys.len()
            )));
        }

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

    fn to_wire(&self) -> WireRepr {
        let tags = vec![self.verb().into()];
        WireRepr::new(Self::SOURCE, Self::VERSION, tags)
    }
}

#[derive(Debug, Clone)]
struct MiningCandidate {
    block_number: u64,
    difficulty: u64,
    parent_hash: Vec<u8>,
}

impl MiningCandidate {
    fn from_noun(noun: Noun) -> Result<Self, MiningError> {
        let cell = noun.as_cell().map_err(|_| MiningError::MiningFailure)?;
        let block_num_atom = cell.head().as_atom().map_err(|_| MiningError::MiningFailure)?;
        let tail_cell = cell.tail().as_cell().map_err(|_| MiningError::MiningFailure)?;
        let difficulty_atom = tail_cell.head().as_atom().map_err(|_| MiningError::MiningFailure)?;
        let parent_hash_atom = tail_cell.tail().as_atom().map_err(|_| MiningError::MiningFailure)?;

        // Map any noun::Error into MiningError::MiningFailure
        let block_number = block_num_atom
            .as_u64()
            .map_err(|_| MiningError::MiningFailure)?;
        let difficulty = difficulty_atom
            .as_u64()
            .map_err(|_| MiningError::MiningFailure)?;
        let parent_hash = parent_hash_atom.as_ne_bytes().to_vec();

        Ok(MiningCandidate {
            block_number,
            difficulty,
            parent_hash,
        })
    }
}

#[instrument(skip(handle))]
async fn set_mining_key(
    handle: &NockAppHandle,
    pubkey: &str,
) -> Result<PokeResult, MiningError> {
    let mut slab = NounSlab::new();
    let command = Atom::from_value(&mut slab, "set-mining-key")
        .map_err(|_| MiningError::NounConstruction)?;
    let pubkey_atom = Atom::from_value(&mut slab, pubkey)
        .map_err(|_| MiningError::NounConstruction)?;

    let poke = T(
        &mut slab,
        &[D(tas!(b"command")), command.as_noun(), pubkey_atom.as_noun()],
    );
    slab.set_root(poke);

    handle
        .poke(MiningWire::SetPubKey.to_wire(), slab)
        .await
        .map_err(|_| MiningError::DriverInitFailed)
}

#[instrument(skip(handle))]
async fn set_mining_key_advanced(
    handle: &NockAppHandle,
    configs: &[MiningKeyConfig],
) -> Result<PokeResult, MiningError> {
    let mut slab = NounSlab::new();
    let command = Atom::from_value(&mut slab, "set-mining-key-advanced")
        .map_err(|_| MiningError::NounConstruction)?;

    let configs_list = configs.iter().rev().try_fold(D(0), |acc, config| {
        let keys = config.keys.iter().rev().try_fold(D(0), |keys_acc, key| {
            Atom::from_value(&mut slab, key.as_str())
                .map(|k| T(&mut slab, &[k.as_noun(), keys_acc]))
                .map_err(|_| MiningError::NounConstruction)
        })?;

        Ok::<_, MiningError>(T(
            &mut slab,
            &[D(config.share), D(config.m), keys, acc],
        ))
    })?;

    let poke = T(&mut slab, &[D(tas!(b"command")), command.as_noun(), configs_list]);
    slab.set_root(poke);

    handle
        .poke(MiningWire::SetPubKey.to_wire(), slab)
        .await
        .map_err(|_| MiningError::DriverInitFailed)
}

#[instrument(skip(handle))]
async fn enable_mining(
    handle: &NockAppHandle,
    enable: bool,
) -> Result<PokeResult, MiningError> {
    let mut slab = NounSlab::new();
    let command = Atom::from_value(&mut slab, "enable-mining")
        .map_err(|_| MiningError::NounConstruction)?;
    let state = Atom::from_value(&mut slab, if enable { "enable" } else { "disable" })
        .map_err(|_| MiningError::NounConstruction)?;

    let poke = T(
        &mut slab,
        &[D(tas!(b"command")), command.as_noun(), state.as_noun()],
    );
    slab.set_root(poke);

    handle
        .poke(MiningWire::Enable.to_wire(), slab)
        .await
        .map_err(|_| MiningError::DriverInitFailed)
}

async fn mine_candidate(
    handle: Arc<NockAppHandle>,
    candidate: MiningCandidate,
) -> Result<(), MiningError> {
    let cores = num_cpus::get();
    let (tx, mut rx) = mpsc::channel(1);
    let candidate = Arc::new(candidate);
    let semaphore = Arc::new(Semaphore::new(cores));

    (0..cores).into_par_iter().for_each(|thread_id| {
        let candidate = Arc::clone(&candidate);
        let tx = tx.clone();
        let semaphore = Arc::clone(&semaphore);

        tokio::spawn(async move {
            let _permit = semaphore.acquire().await;
            let nonce_start = u64::MAX / cores as u64 * thread_id as u64;
            let nonce_end = nonce_start + (u64::MAX / cores as u64);

            for nonce in nonce_start..=nonce_end {
                let hash = blake3::hash(
                    &[
                        &nonce.to_be_bytes(),
                        &candidate.block_number.to_be_bytes(),
                        &candidate.parent_hash[..],
                    ]
                    .concat(),
                );

                if u64::from_be_bytes(hash.as_bytes()[0..8].try_into().unwrap())
                    < candidate.difficulty
                {
                    let _ = tx.send(nonce).await;
                    break;
                }
            }
        });
    });

    if let Some(nonce) = rx.recv().await {
        debug!("Found valid nonce: {}", nonce);
        submit_solution(Arc::clone(&handle), nonce).await?;
    }

    Ok(())
}

async fn submit_solution(
    handle: Arc<NockAppHandle>,
    nonce: u64,
) -> Result<(), MiningError> {
    let mut slab = NounSlab::new();
    let command = Atom::from_value(&mut slab, "submit-solution")
        .map_err(|_| MiningError::NounConstruction)?;
    let nonce_atom = Atom::from_value(&mut slab, nonce)
        .map_err(|_| MiningError::NounConstruction)?;

    let poke = T(
        &mut slab,
        &[D(tas!(b"command")), command.as_noun(), nonce_atom.as_noun()],
    );
    slab.set_root(poke);

    handle
        .poke(MiningWire::Mined.to_wire(), slab)
        .await
        .map_err(|_| MiningError::DriverInitFailed)?;

    Ok(())
}

fn handle_mining_effect(
    handle: Arc<NockAppHandle>,
    effect: Noun,
    tasks: &mut tokio::task::JoinSet<Result<(), MiningError>>,
) -> Result<(), NockAppError> {
    let effect_cell = effect.as_cell().map_err(|_| {
        error!("Received non-cell effect");
        NockAppError::OtherError
    })?;

    if effect_cell.head().eq_bytes("mine") {
        let candidate = match MiningCandidate::from_noun(effect_cell.tail()) {
            Ok(c) => c,
            Err(e) => {
                error!("Invalid mining candidate: {:?}", e);
                return Ok(());
            }
        };

        tasks.spawn(async move {
            mine_candidate(Arc::clone(&handle), candidate).await
        });
    }

    Ok(())
}

pub fn create_mining_driver(
    mining_config: Option<Vec<MiningKeyConfig>>,
    mine: bool,
    init_complete_tx: Option<tokio::sync::oneshot::Sender<()>>,
) -> IODriverFn {
    Box::new(move |handle| {
        Box::pin(async move {
            let handle = Arc::new(handle);
            if let Some(configs) = mining_config.as_deref() {
                let result = if configs.len() == 1
                    && configs[0].share == 1
                    && configs[0].m == 1
                    && configs[0].keys.len() == 1
                {
                    set_mining_key(&handle, &configs[0].keys[0]).await
                } else {
                    set_mining_key_advanced(&handle, configs).await
                };

                result.map_err(|e| {
                    error!("Mining configuration failed: {:?}", e);
                    NockAppError::OtherError
                })?;

                enable_mining(&handle, mine).await.map_err(|e| {
                    error!("Mining enable failed: {:?}", e);
                    NockAppError::OtherError
                })?;

                if let Some(tx) = init_complete_tx {
                    tx.send(()).map_err(|_| {
                        error!("Failed to send initialization complete");
                        NockAppError::OtherError
                    })?;
                }
            } else {
                // no config => disable mining
                enable_mining(&handle, false).await.map_err(|e| {
                    error!("Disabling mining failed: {:?}", e);
                    NockAppError::OtherError
                })?;

                if let Some(tx) = init_complete_tx {
                    tx.send(()).map_err(|_| {
                        error!("Failed to send mining driver init completion");
                        NockAppError::OtherError
                    })?;
                }
                return Ok(());
            }

            let mut tasks = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    effect_res = handle.next_effect() => {
                        match effect_res {
                            Ok(effect_slab) => {
                                // SAFETY: NounSlab::root is unsafe
                                let effect_noun = unsafe { *effect_slab.root() };
                                handle_mining_effect(Arc::clone(&handle), effect_noun, &mut tasks)?
                            },
                            Err(e) => {
                                warn!("Error receiving effect: {:?}", e);
                                continue;
                            }
                        }
                    }
                    Some(res) = tasks.join_next() => {
                        match res {
                            Ok(Ok(())) => debug!("Mining task completed"),
                            Ok(Err(e)) => error!("Mining task failed: {:?}", e),
                            Err(e) => error!("Task join error: {:?}", e),
                        }
                    }
                }
            }
        })
    })
}
