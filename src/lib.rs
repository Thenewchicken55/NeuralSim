//! # NeuralSim
//!
//! A biologically-realistic spiking neural network simulator for massive-scale
//! cortical and brain-region modelling.
//!
//! ## Quick start
//!
//! ```no_run
//! use neural_sim::network::builder::BrainBuilder;
//! use neural_sim::neuron::NeuronModelParams;
//! use neural_sim::simulation::SimulationEngine;
//!
//! let net = BrainBuilder::new()
//!     .add_region("Cortex", 1000, 0.80, NeuronModelParams::default())
//!     .build();
//! let mut engine = SimulationEngine::new(net);
//! engine.simulate_ms(100.0);
//! println!("Total spikes: {}", engine.stats().total_spikes);
//! ```
//!
//! See the `examples/` directory in the repository for more runnable demos.
//!
//! ## Features
//!
//! - **Neuron models**: LIF, Izhikevich, Hodgkin-Huxley (see [`neuron`])
//! - **Synapse types**: AMPA, GABA, GABA-B, NMDA with Mg²⁺ block (see [`synapse`])
//! - **Plasticity**: STDP, triplet STDP, R-STDP, BCM, homeostatic, intrinsic
//! - **Brain regions**: cortical column, thalamocortical loop, hippocampus (see [`network`])
//! - **Engine**: parallel SoA updates via rayon, optional GPU backend (see [`simulation`])
//! - **I/O**: JSON save/load, checkpoints, CSV stats, text↔spike pipeline (see [`io`])

pub mod config;
pub mod error;
pub mod evolution;
pub mod io;
pub mod network;
pub mod neuron;
pub mod simulation;
pub mod synapse;

#[cfg(feature = "gui")]
pub mod gui;

pub mod prelude;
