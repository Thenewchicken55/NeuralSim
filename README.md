# NeuralSim

**A biologically realistic spiking neural network simulator built for massive scale.**

NeuralSim is a high-performance, massively scalable simulator for spiking neural networks (SNNs) that models individual neurons and their synaptic connections with biological fidelity. Each neuron is a discrete agent connected to one or more other neurons, forming complex, dynamic networks that mirror the architecture of real brains.

## Features

- **Single-neuron resolution** — Every neuron is an independent computational unit with membrane potential, firing threshold, refractory period, and multiple built-in models (LIF, Izhikevich, Hodgkin-Huxley)
- **Synaptic plasticity** — Spike-timing-dependent plasticity (STDP), short-term depression/facilitation, and homeostatic scaling
- **Real brain architecture** — Modular brain regions (cortical columns, thalamocortical loops, hippocampus), configurable excitatory/inhibitory ratios (80/20), layer-specific connectivity patterns
- **Massive scalability** — Written in Rust with zero-cost abstractions, SIMD optimizations, and lock-free concurrency. Designed to simulate millions of neurons on consumer hardware and billions on clusters
- **Bare-metal performance** — No runtime overhead, cache-friendly memory layouts (AoS → SoA), CSR adjacency matrices, optional GPU compute shader backend
- **Optional GUI** — Real-time 3D visualization of network activity, spike raster plots, membrane potential traces, and interactive neuron inspection powered by `egui`
- **Deterministic simulation** — Reproducible results with seeded RNG for scientific use
- **Serialization** — Save/load full network state, configurable checkpoints, JSON export for analysis

## Architecture

```
┌─────────────────────────────────────────────┐
│                 GUI Layer                    │
│  (egui + 3D visualization, optional)        │
├─────────────────────────────────────────────┤
│           Simulation Engine                  │
│  ┌─────────┐ ┌─────────┐ ┌───────────────┐ │
│  │Neuron   │ │Synapse  │ │Plasticity     │ │
│  │Models   │ │Dynamics │ │Rules (STDP,   │ │
│  │(LIF, HH,│ │(AMPA,   │ │BCM, triplet)  │ │
│  │Izhikevich)│GABA, NMDA)│ │               │ │
│  └─────────┘ └─────────┘ └───────────────┘ │
├─────────────────────────────────────────────┤
│         Data Layer (SoA, CSR, pools)        │
├─────────────────────────────────────────────┤
│     Parallel Runtime (Rayon + async IO)     │
├─────────────────────────────────────────────┤
│           Hardware Backend                  │
│  (CPU SIMD / optional GPU via wgpu)        │
└─────────────────────────────────────────────┘
```

### Neuron Models

| Model | Biological Fidelity | Performance | Use Case |
|-------|-------------------|-------------|----------|
| LIF | Medium | Highest | Large-scale simulation |
| Izhikevich | High | High | Balance of speed and realism |
| Hodgkin-Huxley | Highest | Moderate | Detailed biophysical study |

### Synapse Types

| Type | Time Scale | Function |
|------|-----------|----------|
| AMPA | ~2 ms | Fast excitation |
| GABA | ~10 ms | Fast inhibition |
| NMDA | ~100 ms | Slow excitation, plasticity |

### Brain Region Templates

- Cortical column (6 layers, layer-specific connectivity)
- Thalamocortical loop
- Hippocampal formation (DG → CA3 → CA1)
- Basal ganglia circuit
- Custom user-defined regions via JSON config

## Getting Started

```bash
# Clone and build
git clone https://github.com/Thenewchicken55/NeuralSim.git
cd NeuralSim
cargo build --release

# Run with GUI
cargo run --release --features gui

# Run headless (CLI only)
cargo run --release
```

### Quick Example

```rust
use neural_sim::prelude::*;

fn main() {
    // Create a cortical column with 10,000 neurons
    let mut column = CorticalColumn::builder()
        .excitatory_ratio(0.8)
        .build(10_000);

    // Simulate 1000 ms of activity
    column.simulate_ms(1000.0);

    // Inspect spike activity
    println!("Total spikes: {}", column.spike_count());
}
```

## Performance Goals

| Scale | Neurons | Synapses | Memory | Performance Target |
|-------|---------|----------|--------|-------------------|
| Small | 10k | 1M | ~50 MB | >1000x realtime |
| Medium | 1M | 100M | ~2 GB | >10x realtime |
| Large | 100M | 10B | ~200 GB | ~1x realtime |
| Cluster | 1B+ | 100B+ | Distributed | ~1x realtime |

## Project Structure

```
src/
├── main.rs                  # Entry point (GUI or CLI)
├── neuron/                  # Neuron models
│   ├── mod.rs
│   ├── lif.rs               # Leaky Integrate-and-Fire
│   ├── izhikevich.rs        # Izhikevich model
│   └── hodgkin_huxley.rs    # Hodgkin-Huxley model
├── synapse/                 # Synaptic dynamics
│   ├── mod.rs
│   ├── types.rs             # AMPA, GABA, NMDA
│   └── plasticity.rs        # STDP, BCM, triplet rules
├── network/                 # Network topology
│   ├── mod.rs
│   ├── graph.rs             # Adjacency (CSR format)
│   ├── region.rs            # Brain region templates
│   └── connectivity.rs      # Connectivity patterns
├── simulation/              # Simulation engine
│   ├── mod.rs
│   ├── engine.rs            # Parallel tick execution
│   └── scheduler.rs         # Event scheduling
├── io/                      # Serialization
│   ├── mod.rs
│   ├── save.rs
│   └── load.rs
├── gui/                     # GUI (optional, behind feature flag)
│   ├── mod.rs
│   ├── app.rs               # egui application
│   └── viz.rs               # 3D visualization
└── prelude.rs               # Re-exports
```

## Research Foundations

NeuralSim is inspired by:
- **SpiNNaker** — Massively parallel neuromorphic architecture
- **Blue Brain Project** — Detailed cortical column reconstruction
- **BrainCog** — Multi-scale brain simulation framework
- **NeuroxAI** — GPU-accelerated SNN platform
- **Neuromod** — Rust SNN library with Hodgkin-Huxley dynamics
- **Neuropool** — Cache-friendly LIF neuron substrate

## License

MIT
