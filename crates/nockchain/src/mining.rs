use std::str::FromStr;
use std::sync::Arc;

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
use tempfile::{tempdir, TempDir};
use tokio::sync::Mutex;
use tracing::{instrument, warn, info, debug, error};
use uuid::Uuid;

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

// Kernel实例包装器，包含其临时目录
struct KernelInstance {
    kernel: Kernel,
    _temp_dir: TempDir, // 保持临时目录存活
}

impl KernelInstance {
    async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let temp_dir = tempdir().map_err(|e| format!("Failed to create temp dir: {}", e))?;
        let hot_state = zkvm_jetpack::hot::produce_prover_hot_state(); // 直接使用，就像原始代码
        let snapshot_path_buf = temp_dir.path().to_path_buf();
        let jam_paths = JamPaths::new(temp_dir.path());
        
        let kernel = Kernel::load_with_hot_state_huge(
            snapshot_path_buf,
            jam_paths,
            KERNEL,
            &hot_state, // 直接传递引用，就像原始代码
            false,
        ).await.map_err(|e| format!("Failed to load kernel: {}", e))?;

        Ok(Self {
            kernel,
            _temp_dir: temp_dir,
        })
    }

    async fn poke(&self, wire: nockapp::wire::WireRepr, slab: NounSlab) -> Result<NounSlab, nockapp::CrownError> {
        self.kernel.poke(wire, slab).await
    }
}

// Mining Kernel Pool
pub struct MiningKernelPool {
    kernels: Arc<Mutex<Vec<KernelInstance>>>,
    max_kernels: usize,
    created_kernels: Arc<Mutex<usize>>,
}

impl MiningKernelPool {
    pub async fn new(pool_size: usize) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        info!(pool_size, "Creating mining kernel pool...");
        
        let kernels = Vec::with_capacity(pool_size);
        
        Ok(Self {
            kernels: Arc::new(Mutex::new(kernels)),
            max_kernels: pool_size,
            created_kernels: Arc::new(Mutex::new(0)),
        })
    }

    // 懒加载：只在需要时创建kernel
    async fn ensure_kernel_available(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut kernels = self.kernels.lock().await;
        let mut created_count = self.created_kernels.lock().await;
        
        if kernels.is_empty() && *created_count < self.max_kernels {
            info!("Creating new kernel instance ({}/{})", *created_count + 1, self.max_kernels);
            let kernel_instance = KernelInstance::new().await?; // 简化：每次创建时生成hot_state
            kernels.push(kernel_instance);
            *created_count += 1;
            info!("Kernel instance created successfully");
        }
        
        Ok(())
    }

    // 借用一个kernel实例
    pub async fn borrow_kernel(&self) -> Result<Option<KernelInstance>, Box<dyn std::error::Error + Send + Sync>> {
        self.ensure_kernel_available().await?;
        
        let mut kernels = self.kernels.lock().await;
        Ok(kernels.pop())
    }

    // 归还kernel实例
    pub async fn return_kernel(&self, kernel: KernelInstance) {
        let mut kernels = self.kernels.lock().await;
        if kernels.len() < self.max_kernels {
            kernels.push(kernel);
        }
        // 如果超出容量限制，kernel会被丢弃（临时目录也会被清理）
    }

    pub async fn get_stats(&self) -> (usize, usize, usize) {
        let kernels = self.kernels.lock().await;
        let created = self.created_kernels.lock().await;
        (kernels.len(), *created, self.max_kernels)
    }
}

// 使用kernel池的挖矿函数
pub async fn mining_attempt_with_pool(
    candidate: NounSlab,
    handle: NockAppHandle,
    kernel_pool: Arc<MiningKernelPool>,
) -> Option<NounSlab> {
    let task_id = Uuid::new_v4();
    let mining_start = std::time::Instant::now();
    
    debug!(task_id = %task_id, "Mining attempt started");

    // 从池中借用kernel
    let kernel = match kernel_pool.borrow_kernel().await {
        Ok(Some(k)) => k,
        Ok(None) => {
            warn!(task_id = %task_id, "No available kernel in pool");
            return None;
        }
        Err(e) => {
            error!(task_id = %task_id, error = %e, "Failed to borrow kernel");
            return None;
        }
    };

    debug!(task_id = %task_id, "Borrowed kernel from pool");

    // 执行挖矿 - 和原始代码逻辑相同
    let result = match kernel.poke(MiningWire::Candidate.to_wire(), candidate).await {
        Ok(effects_slab) => {
            let mut found_result = None;
            for effect in effects_slab.to_vec() {
                let Ok(effect_cell) = (unsafe { effect.root().as_cell() }) else {
                    drop(effect);
                    continue;
                };
                if effect_cell.head().eq_bytes("command") {
                    match handle.poke(MiningWire::Mined.to_wire(), effect.clone()).await {
                        Ok(_) => {
                            let mining_duration = mining_start.elapsed();
                            info!(
                                task_id = %task_id,
                                duration_ms = mining_duration.as_millis(),
                                "Mining completed successfully"
                            );
                            found_result = Some(effect);
                        }
                        Err(e) => {
                            error!(task_id = %task_id, error = %e, "Failed to poke mined result");
                        }
                    }
                }
            }
            found_result
        }
        Err(e) => {
            error!(task_id = %task_id, error = %e, "Mining kernel poke failed");
            None
        }
    };

    // 归还kernel到池中
    kernel_pool.return_kernel(kernel).await;
    debug!(task_id = %task_id, "Returned kernel to pool");

    result
}

// 统计信息结构
#[derive(Clone)]
struct MiningStats {
    total_attempts: std::sync::Arc<std::sync::atomic::AtomicU64>,
    successful_attempts: std::sync::Arc<std::sync::atomic::AtomicU64>,
    start_time: std::time::Instant,
}

impl MiningStats {
    fn new() -> Self {
        Self {
            total_attempts: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            successful_attempts: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            start_time: std::time::Instant::now(),
        }
    }

    fn record_attempt(&self, successful: bool) {
        self.total_attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if successful {
            self.successful_attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    async fn log_periodic_stats(&self, kernel_pool: &MiningKernelPool) {
        let total = self.total_attempts.load(std::sync::atomic::Ordering::Relaxed);
        let successful = self.successful_attempts.load(std::sync::atomic::Ordering::Relaxed);
        let elapsed = self.start_time.elapsed();
        let (available_kernels, created_kernels, max_kernels) = kernel_pool.get_stats().await;

        if total > 0 {
            info!(
                total_attempts = total,
                successful = successful,
                success_rate = format!("{:.2}%", (successful as f64 / total as f64) * 100.0),
                attempts_per_minute = format!("{:.2}", total as f64 / elapsed.as_secs_f64() * 60.0),
                runtime = format!("{:.2}s", elapsed.as_secs_f64()),
                available_kernels,
                created_kernels,
                max_kernels,
                "Mining statistics"
            );
        }
    }
}

// create_mining_driver 和其他函数保持不变...
pub fn create_mining_driver(
    mining_config: Option<Vec<MiningKeyConfig>>,
    mine: bool,
    mining_threads: usize,
    kernel_pool_size: usize,
    init_complete_tx: Option<tokio::sync::oneshot::Sender<()>>,
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

            // 设置挖矿密钥
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

            // 创建kernel池
            let kernel_pool = match MiningKernelPool::new(kernel_pool_size).await {
                Ok(pool) => Arc::new(pool),
                Err(e) => {
                    error!("Failed to create kernel pool: {}", e);
                    return Err(NockAppError::OtherError);
                }
            };

            info!(
                mining_threads,
                kernel_pool_size,
                "Mining driver initialized with kernel pool"
            );

            let mut pending_candidates: Vec<NounSlab> = Vec::new();
            let mut current_attempts: tokio::task::JoinSet<Option<NounSlab>> = tokio::task::JoinSet::new();
            let mut active_mining_count = 0;
            let max_concurrent_mining = mining_threads.max(1);
            let stats = MiningStats::new();

            // 定期统计日志
            let stats_clone = stats.clone();
            let pool_clone = kernel_pool.clone();
            let mut stats_interval = tokio::time::interval(std::time::Duration::from_secs(30));

            loop {
                tokio::select! {
                    _ = stats_interval.tick() => {
                        stats_clone.log_periodic_stats(&pool_clone).await;
                    },
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

                            if active_mining_count < max_concurrent_mining {
                                let (cur_handle, attempt_handle) = handle.dup();
                                handle = cur_handle;
                                let pool_clone = kernel_pool.clone();

                                current_attempts.spawn(
                                    mining_attempt_with_pool(candidate_slab, attempt_handle, pool_clone)
                                );
                                active_mining_count += 1;

                                debug!(
                                    active_threads = active_mining_count,
                                    max_threads = max_concurrent_mining,
                                    queue_size = pending_candidates.len(),
                                    "Started mining task"
                                );
                            } else {
                                pending_candidates.push(candidate_slab);
                                debug!(
                                    queue_size = pending_candidates.len(),
                                    "Mining task queued (all threads busy)"
                                );
                            }
                        }
                    },
                    mining_result = current_attempts.join_next(), if !current_attempts.is_empty() => {
                        active_mining_count -= 1;

                        match mining_result {
                            Some(Ok(Some(_result))) => {
                                stats.record_attempt(true);
                                info!(
                                    active_threads = active_mining_count,
                                    "Mining attempt successful"
                                );
                            },
                            Some(Ok(None)) => {
                                stats.record_attempt(false);
                                debug!(
                                    active_threads = active_mining_count,
                                    "Mining attempt completed without result"
                                );
                            },
                            Some(Err(e)) => {
                                stats.record_attempt(false);
                                warn!(
                                    error = %e,
                                    active_threads = active_mining_count,
                                    "Mining attempt failed"
                                );
                            },
                            None => {
                                error!("Unexpected empty mining result");
                            }
                        }

                        // 启动队列中的下一个任务
                        if let Some(next_candidate) = pending_candidates.pop() {
                            let (cur_handle, attempt_handle) = handle.dup();
                            handle = cur_handle;
                            let pool_clone = kernel_pool.clone();

                            current_attempts.spawn(
                                mining_attempt_with_pool(next_candidate, attempt_handle, pool_clone)
                            );
                            active_mining_count += 1;

                            debug!(
                                active_threads = active_mining_count,
                                queue_size = pending_candidates.len(),
                                "Started queued mining task"
                            );
                        }
                    }
                }
            }
        })
    })
}

// 保留原来的函数用于向后兼容 - 完全按照原始代码
pub async fn mining_attempt(candidate: NounSlab, handle: NockAppHandle) -> () {
    let snapshot_dir =
        tokio::task::spawn_blocking(|| tempdir().expect("Failed to create temporary directory"))
            .await
            .expect("Failed to create temporary directory");
    let hot_state = zkvm_jetpack::hot::produce_prover_hot_state();
    let snapshot_path_buf = snapshot_dir.path().to_path_buf();
    let jam_paths = JamPaths::new(snapshot_dir.path());
    
    let kernel =
        Kernel::load_with_hot_state_huge(snapshot_path_buf, jam_paths, KERNEL, &hot_state, false)
            .await
            .expect("Could not load mining kernel");
    let effects_slab = kernel
        .poke(MiningWire::Candidate.to_wire(), candidate)
        .await
        .expect("Could not poke mining kernel with candidate");
    for effect in effects_slab.to_vec() {
        let Ok(effect_cell) = (unsafe { effect.root().as_cell() }) else {
            drop(effect);
            continue;
        };
        if effect_cell.head().eq_bytes("command") {
            handle
                .poke(MiningWire::Mined.to_wire(), effect)
                .await
                .expect("Could not poke nockchain with mined PoW");
        }
    }
}

// 其余函数保持不变...
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

    let mut configs_list = D(0);
    for config in configs {
        let mut keys_noun = D(0);
        for key in config.keys {
            let key_atom =
                Atom::from_value(&mut set_mining_key_slab, key).expect("Failed to create key atom");
            keys_noun = T(&mut set_mining_key_slab, &[key_atom.as_noun(), keys_noun]);
        }

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
