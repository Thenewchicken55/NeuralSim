//! The simulation engine, scheduler, and optional GPU backend.
//!
//! # Engine
//! [`SimulationEngine`] is the core driver. It performs three phases per step:
//! 1. **Neuron update** — parallel SoA mutation via rayon's `zip_eq`
//! 2. **Synapse propagation** — conductance dynamics, STDP, R-STDP, homeostasis
//! 3. **Stats update** — firing rates, synchrony, weight statistics
//!
//! # Scheduler
//! [`scheduler::Scheduler`] runs the engine in a background thread so the GUI
//! or other frontends can read stats without blocking the simulation.
//!
//! # GPU backend
//! Behind the `gpu` feature flag, [`gpu_backend::GpuBackend`] dispatches LIF
//! updates and sparse matrix-vector multiplication (SpMV) via wgpu compute shaders.

pub mod engine;
pub mod scheduler;

#[cfg(feature = "gpu")]
pub mod gpu_backend;

pub use engine::{SimulationEngine, SimulationStats, StepResult};
