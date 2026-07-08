use crate::simulation::{SimulationEngine, SimulationStats};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Controls simulation pacing — realtime, fast-as-possible, or step-by-step
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimSpeed {
    Realtime,
    FastAsPossible,
    StepByStep,
}

/// Scheduler that runs the simulation loop in a background thread.
///
/// Shares the engine (via Arc<parking_lot::Mutex>) with the GUI or other
/// frontends so they can read stats without locking.
pub struct Scheduler {
    pub engine: Arc<Mutex<SimulationEngine>>,
    pub stats: Arc<parking_lot::RwLock<SimulationStats>>,
    running: Arc<AtomicBool>,
    pub speed: SimSpeed,
    pub target_step_dt: Duration,
}

impl Scheduler {
    pub fn new(engine: Arc<Mutex<SimulationEngine>>, stats: Arc<parking_lot::RwLock<SimulationStats>>) -> Self {
        Self {
            engine,
            stats,
            running: Arc::new(AtomicBool::new(false)),
            speed: SimSpeed::FastAsPossible,
            target_step_dt: Duration::from_secs_f64(0.001),
        }
    }

    /// Start the simulation loop in a background thread.
    /// Runs until `stop()` is called.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();
        let engine = self.engine.clone();
        let stats = self.stats.clone();
        let speed = self.speed;
        let target = self.target_step_dt;

        std::thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                let step_start = Instant::now();
                let mut eng = engine.lock();
                eng.step();
                // Sync stats for GUI reads
                *stats.write() = eng.stats();
                drop(eng);
                if speed == SimSpeed::Realtime {
                    let elapsed = step_start.elapsed();
                    if elapsed < target {
                        std::thread::sleep(target - elapsed);
                    }
                }
            }
        });
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn set_speed(&mut self, speed: SimSpeed) {
        self.speed = speed;
    }
}
