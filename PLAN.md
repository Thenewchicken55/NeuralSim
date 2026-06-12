# NeuralSim Implementation Plan

## Phase 0: Foundation (this commit)
- [x] Delete old Readme.md.txt
- [x] Create README.md
- [x] Create PLAN.md
- [x] Add .gitignore for Rust projects
- [x] Configure Cargo.toml with dependencies
- [x] Initialize git repo
- [x] Initial commit: "Initial commit: project scaffold, README, and plan"

## Phase 1: Core Data Structures
- [ ] Define NeuronId, SynapseId newtypes
- [ ] Implement Neuron struct (membrane potential, firing threshold, refractory counter)
- [ ] Implement Synapse struct (source, target, weight, delay, type)
- [ ] Implement SynapseType enum (AMPA, GABA, NMDA)
- [ ] Implement Network struct with adjacency storage (CSR format)
- [ ] Implement NeuronState SoA (separate arrays for V, refractory, spikes)
- [ ] Add unit tests for data structures
- [ ] Commit: "Phase 1: core data structures — Neuron, Synapse, Network"

## Phase 2: Neuron Models
- [ ] Implement LIF (Leaky Integrate-and-Fire) — `dV/dt = -V/τ + I`
- [ ] Implement Izhikevich model — 2D system with recovery variable
- [ ] Implement Hodgkin-Huxley model — 4 ODE system (V, m, n, h)
- [ ] Create NeuronModel trait with `step(dt, current) -> bool`
- [ ] Add NeuronModel enum dispatching to concrete models
- [ ] Implement input current integration (exponential, alpha, delta)
- [ ] Add unit tests with known spike patterns
- [ ] Commit: "Phase 2: neuron models — LIF, Izhikevich, Hodgkin-Huxley"

## Phase 3: Synaptic Dynamics & Plasticity
- [ ] Implement conductance-based synapse dynamics (AMPA, GABA, NMDA)
- [ ] Implement current-based exponential synapse model
- [ ] Implement STDP (Spike-Timing-Dependent Plasticity) with nearest-spike pair rule
- [ ] Implement short-term plasticity (Tsodyks-Markram model)
- [ ] Implement homeostatic scaling (synaptic scaling, intrinsic plasticity)
- [ ] Implement triplet STDP rule for biological accuracy
- [ ] Add unit tests verifying STDP weight changes
- [ ] Commit: "Phase 3: synaptic dynamics and STDP plasticity rules"

## Phase 4: Network Graph & Connectivity
- [ ] Implement Compressed Sparse Row (CSR) adjacency
- [ ] Implement random connectivity (Erdos-Renyi, fixed probability)
- [ ] Implement distance-based connectivity (Gaussian profile)
- [ ] Implement small-world connectivity (Watts-Strogatz)
- [ ] Implement layer-specific connectivity (cortical column motifs)
- [ ] Add connectivity validation (no self-loops, Dale's law enforcement)
- [ ] Add unit tests for connectivity patterns
- [ ] Commit: "Phase 4: network graph and connectivity patterns"

## Phase 5: Simulation Engine
- [ ] Implement sequential tick: update neurons → propagate spikes → update synapses
- [ ] Implement parallel tick using Rayon (split neuron population)
- [ ] Implement spike event buffer with lock-free push
- [ ] Implement scheduler with configurable time step (dt)
- [ ] Implement progress tracking and statistics
- [ ] Add deterministic mode (seeded RNG per thread)
- [ ] Add benchmarks for 10k/100k/1M neurons
- [ ] Commit: "Phase 5: parallel simulation engine with Rayon"

## Phase 6: Brain Region Templates
- [ ] Implement CorticalColumn template (6 layers, layer-specific E/I ratios)
- [ ] Implement ThalamocorticalLoop template
- [ ] Implement HippocampalFormation template (DG → CA3 → CA1)
- [ ] Implement RegionConfig for user-defined region parameters
- [ ] Implement region composition (multiple regions connected)
- [ ] Add validation for inter-region connectivity
- [ ] Commit: "Phase 6: brain region templates and composition"

## Phase 7: Serialization & IO
- [ ] Implement binary serialization (bincode) for network state
- [ ] Implement JSON import/export for configuration
- [ ] Implement checkpoint system (periodic save during simulation)
- [ ] Implement deterministic replay from saved state
- [ ] Add migration support for versioned save files
- [ ] Commit: "Phase 7: serialization and IO"

## Phase 8: GUI (Optional)
- [ ] Set up egui framework with winit window
- [ ] Implement network control panel (start/stop, speed, neuron count)
- [ ] Implement spike raster plot rendering
- [ ] Implement membrane potential trace viewer
- [ ] Implement 3D network visualization with wgpu
- [ ] Implement neuron inspector (click to inspect V, spikes, connections)
- [ ] Add feature flag `gui` behind `cfg(feature = "gui")`
- [ ] Commit: "Phase 8: optional GUI with egui and 3D visualization"

## Phase 9: Performance Optimization
- [ ] Profile with perf/optick and identify bottlenecks
- [ ] Implement SIMD-accelerated neuron updates (portable-simd)
- [ ] Optimize memory layout — SoA with cache-line alignment
- [ ] Implement adaptive time stepping
- [ ] Add wgpu-based GPU compute backend (optional feature)
- [ ] Implement multi-threaded synapse pruning for inactive connections
- [ ] Add benchmarks suite with criterion
- [ ] Commit: "Phase 9: performance optimization and GPU backend"

## Phase 10: Polish & Documentation
- [ ] Write API documentation with examples for all public types
- [ ] Add doc-tests for main entry points
- [ ] Create examples directory with demo scripts
- [ ] Add GitHub Actions CI (test, lint, benchmark)
- [ ] Add CONTRIBUTING.md with development guidelines
- [ ] Final review and README updates
- [ ] Commit: "Phase 10: documentation, CI, and final polish"
