use crate::simulation::SimulationEngine;
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

/// Scheduler that runs the simulation loop in a background thread
pub struct Scheduler {
    engine: Arc<std::sync::Mutex<SimulationEngine>>,
    running: Arc<AtomicBool>,
    speed: SimSpeed,
    target_step_dt: Duration,
}

impl Scheduler {
    pub fn new(engine: SimulationEngine) -> Self {
        Self {
            engine: Arc::new(std::sync::Mutex::new(engine)),
            running: Arc::new(AtomicBool::new(false)),
            speed: SimSpeed::FastAsPossible,
            target_step_dt: Duration::from_secs_f64(0.001),
        }
    }

    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();
        let engine = self.engine.clone();
        let speed = self.speed;
        let target = self.target_step_dt;

        std::thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                let step_start = Instant::now();
                if let Ok(mut eng) = engine.lock() {
                    eng.step();
                }
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
}
