use crate::simulation::{SimulationEngine, SimulationStats};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Controls simulation pacing — realtime, fast-as-possible, or step-by-step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimSpeed {
    Realtime,
    FastAsPossible,
    StepByStep,
}

/// Scheduler that runs the simulation loop in a background thread.
///
/// Shares the engine (via `Arc<parking_lot::Mutex>`) with the GUI or other
/// frontends so they can read stats without blocking the simulation.
///
/// The speed and target step duration are read from `Arc<AtomicU8>` / shared
/// state so they can be changed while the loop is running.
pub struct Scheduler {
    pub engine: Arc<Mutex<SimulationEngine>>,
    pub stats: Arc<parking_lot::RwLock<SimulationStats>>,
    running: Arc<AtomicBool>,
    /// Encoded `SimSpeed` as `u8` so the worker thread can observe changes.
    speed: Arc<std::sync::atomic::AtomicU8>,
    pub target_step_dt: Duration,
}

impl Scheduler {
    pub fn new(
        engine: Arc<Mutex<SimulationEngine>>,
        stats: Arc<parking_lot::RwLock<SimulationStats>>,
    ) -> Self {
        Self {
            engine,
            stats,
            running: Arc::new(AtomicBool::new(false)),
            speed: Arc::new(std::sync::atomic::AtomicU8::new(
                SimSpeed::FastAsPossible as u8,
            )),
            target_step_dt: Duration::from_secs_f64(0.001),
        }
    }

    /// Start the simulation loop in a background thread.
    ///
    /// If a loop is already running, this is a no-op (prevents duplicate workers).
    pub fn start(&self) {
        // CAS-style guard: only spawn if not already running.
        if self
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let running = self.running.clone();
        let engine = self.engine.clone();
        let stats = self.stats.clone();
        let speed_atomic = self.speed.clone();
        let target = self.target_step_dt;

        std::thread::Builder::new()
            .name("neural-sim-worker".into())
            .spawn(move || {
                while running.load(Ordering::SeqCst) {
                    let step_start = Instant::now();
                    let speed = match speed_atomic.load(Ordering::Relaxed) {
                        0 => SimSpeed::Realtime,
                        1 => SimSpeed::FastAsPossible,
                        _ => SimSpeed::StepByStep,
                    };
                    if speed == SimSpeed::StepByStep {
                        // Yield until the user advances manually; for now just pause.
                        std::thread::sleep(Duration::from_millis(10));
                        continue;
                    }
                    {
                        let mut eng = engine.lock();
                        eng.step();
                        *stats.write() = eng.stats();
                    }
                    if speed == SimSpeed::Realtime {
                        let elapsed = step_start.elapsed();
                        if elapsed < target {
                            std::thread::sleep(target - elapsed);
                        }
                    }
                }
            })
            .expect("failed to spawn simulation worker thread");
    }

    /// Stop the simulation loop. The worker thread will exit on its next iteration.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Change the simulation speed while the loop is running.
    pub fn set_speed(&self, speed: SimSpeed) {
        self.speed.store(speed as u8, Ordering::Relaxed);
    }
}
