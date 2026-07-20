# NeuralSim

**A biologically realistic spiking neural network simulator built for massive scale.**

NeuralSim is a high-performance, massively scalable simulator for spiking neural networks (SNNs) that models individual neurons and their synaptic connections with biological fidelity. Each neuron is a discrete agent connected to one or more other neurons, forming complex, dynamic networks that mirror the architecture of real brains.

## Features

- **Single-neuron resolution** — Every neuron is an independent computational unit with membrane potential, firing threshold, refractory period, and multiple built-in models (LIF, Izhikevich, Hodgkin-Huxley)
- **Synaptic plasticity** — Spike-timing-dependent plasticity (STDP), triplet STDP, R-STDP (reward-modulated), BCM, short-term depression/facilitation, homeostatic scaling, intrinsic plasticity, and synaptic consolidation
- **Real brain architecture** — Modular brain regions (cortical columns, thalamocortical loops, hippocampus), configurable excitatory/inhibitory ratios (80/20), layer-specific connectivity patterns
- **Massive scalability** — Written in Rust with zero-cost abstractions, lock-free concurrency (rayon), and cache-friendly SoA memory layouts. Designed to simulate millions of neurons on consumer hardware
- **Bare-metal performance** — CSR adjacency matrices, split-borrow SoA parallel updates, optional GPU compute shader backend (wgpu)
- **Optional GUI** — Real-time visualization of network activity, spike raster plots, membrane potential traces, and interactive neuron inspection powered by `egui`
- **Deterministic simulation** — Reproducible results with seeded RNG for scientific use; seed recorded in checkpoint metadata
- **Serialization** — Save/load full network state (JSON), configurable checkpoints with pruning, CSV stats export, text↔spike I/O pipeline

## Architecture

```
┌─────────────────────────────────────────────┐
│                 GUI Layer                    │
│  (egui + brain visualization, optional)      │
├─────────────────────────────────────────────┤
│           Simulation Engine                  │
│  ┌─────────┐ ┌─────────┐ ┌───────────────┐ │
│  │Neuron   │ │Synapse  │ │Plasticity     │ │
│  │Models   │ │Dynamics │ │Rules (STDP,   │ │
│  │(LIF, HH,│ │(AMPA,   │ │BCM, triplet,  │ │
│  │Izhikevich)│GABA, NMDA)│ R-STDP, homeo) │ │
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
| GABA-B | ~150 ms | Slow metabotropic inhibition |
| NMDA | ~100 ms | Slow excitation, voltage-gated (Mg²⁺ block), plasticity |

### Brain Region Templates

- Cortical column (6 layers, layer-specific connectivity)
- Thalamocortical loop
- Hippocampal formation (DG → CA3 → CA1)
- Custom user-defined regions via JSON/YAML config

## Installation

### Pre-built binaries (recommended)

Download the latest binary for your platform from the
[Releases](https://github.com/Thenewchicken55/NeuralSim/releases) page.

### Build from source

```bash
# Clone and build
git clone https://github.com/Thenewchicken55/NeuralSim.git
cd NeuralSim
cargo build --release

# Run headless CLI demo
cargo run --release --no-default-features -- --cli

# Run with GPU acceleration
cargo run --release --no-default-features --features gpu -- --cli

# Run with GUI (default features)
cargo run --release
```

### Quick Example

```rust
use neural_sim::network::builder::BrainBuilder;
use neural_sim::neuron::NeuronModelParams;
use neural_sim::simulation::SimulationEngine;
use neural_sim::synapse::PlasticityConfig;
use neural_sim::synapse::plasticity::{StdpRule, ConsolidationRule, IntrinsicPlasticity};

fn main() {
    let izh = NeuronModelParams::Izhikevich { a: 0.02, b: 0.2, c: -65.0, d: 8.0 };

    // Build a multi-region brain
    let network = BrainBuilder::new()
        .with_name("DemoBrain")
        .with_plasticity(true)
        .add_region("Input", 100, 0.80, izh)
        .mark_input("Input")
        .add_cortical_column("V1", 500)
        .add_region("Output", 100, 0.80, izh)
        .mark_output("Output")
        .connect_regions("Input", "V1", 0.03, 1.0, None)
        .connect_regions("V1", "Output", 0.03, 1.0, None)
        .build();

    let plasticity = PlasticityConfig {
        stdp: Some(StdpRule::default()),
        consolidation: Some(ConsolidationRule::default()),
        intrinsic: Some(IntrinsicPlasticity::default()),
        enabled: true,
        homeostatic_target_rate: 5.0,
        homeostatic_tau: 5000.0,
        ..Default::default()
    };

    let mut engine = SimulationEngine::new(network)
        .with_noise(10.0)
        .with_plasticity(plasticity)
        .with_conductance(true);

    // Enable plasticity on existing synapses
    {
        let mut net = engine.network.write();
        for syn in net.synapses.iter_mut() {
            *syn = syn.clone().with_plasticity();
        }
    }

    // Simulate 1000 ms of activity
    engine.simulate_ms(1000.0);

    // Inspect spike activity
    let stats = engine.stats();
    println!("Total spikes:  {}", stats.total_spikes);
    println!("Output spikes: {}", stats.output_spikes);
    println!("Mean rate:     {:.1} Hz", stats.mean_firing_rate);
}
```

See `examples/hello_brain.rs`, `examples/neuron_models.rs`, and `examples/stdp_learning.rs`
for more runnable examples.

## CLI

```sh
neural_sim                 # launch GUI (if built with the `gui` feature)
neural_sim --cli           # run the headless demo pipeline
neural_sim --cli --seed 7  # override the RNG seed
neural_sim --cli -c brain.yaml   # run from a YAML/JSON config
neural_sim --help          # full option list
```

## Performance Goals

| Scale | Neurons | Synapses | Memory | Performance Target |
|-------|---------|----------|--------|-------------------|
| Small | 10k | 1M | ~50 MB | >1000x realtime |
| Medium | 1M | 100M | ~2 GB | >10x realtime |
| Large | 100M | 10B | ~200 GB | ~1x realtime |

## Project Structure

```
src/
├── main.rs                  # Entry point (GUI or CLI)
├── cli.rs                   # clap-based CLI + phased demo pipeline
├── gui_entry.rs             # Wires up the GUI when `gui` is on
├── error.rs                 # NeuralSimError / Result aliases
├── prelude.rs               # Convenience re-exports
├── config/                  # YAML/JSON config loading
├── neuron/                  # Neuron models (LIF, Izhikevich, HH)
├── synapse/                 # Synapse types, dynamics, plasticity rules
├── network/                 # CSR graph, brain builder, region templates
├── simulation/              # Engine, scheduler, GPU backend + shaders
├── io/                      # Save/load, checkpointing, text I/O
└── gui/                     # egui app + brain visualization
examples/                    # Runnable examples (hello_brain, neuron_models, stdp_learning)
benches/                     # criterion benchmarks
tests/                       # integration tests
```

See `AGENTS.md` for build/test commands and conventions for contributors.

## Research Foundations

NeuralSim is inspired by:
- **SpiNNaker** — Massively parallel neuromorphic architecture
- **Blue Brain Project** — Detailed cortical column reconstruction
- **BrainCog** — Multi-scale brain simulation framework
- **NeuroxAI** — GPU-accelerated SNN platform
- **Neuromod** — Rust SNN library with Hodgkin-Huxley dynamics
- **Neuropool** — Cache-friendly LIF neuron substrate

## License

Dual-licensed under [MIT](LICENSE-MIT) OR [Apache 2.0](LICENSE-APACHE).
