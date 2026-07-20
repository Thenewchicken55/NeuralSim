# AGENTS.md — NeuralSim

> Build/test/lint commands and conventions for AI coding tools working on NeuralSim.

## Project

NeuralSim is a biologically-realistic spiking neural network simulator written in Rust.
Edition 2024. The crate name is `neural_sim` (library) with a binary also named `neural_sim`.

## Toolchain

- Rust stable (≥ 1.85 for edition 2024). Developed against 1.97.
- No extra system libraries required for the default (CPU) build.
- The `gui` feature pulls in `eframe`/`egui` and (on macOS) the system AppKit frameworks.
- The `gpu` feature pulls in `wgpu` and requires a Vulkan/Metal/DX12-capable device.

## Build / Test / Lint

```sh
# Quick build (CPU only, default features = gui)
cargo build

# Build everything (CPU + GPU + GUI)
cargo build --all-features

# Headless build (no GUI, no GPU — fastest iteration)
cargo build --no-default-features

# Run the test suite
cargo test --all-features

# Run only the unit tests in a specific module
cargo test --all-features simulation::engine

# Lint (treat warnings as errors)
cargo clippy --all-features -- -D warnings

# Format check
cargo fmt --check

# Format apply
cargo fmt

# Run the CLI demo (headless)
cargo run --no-default-features -- --cli

# Run the GUI
cargo run --release                    # uses default features (gui)

# Run an example
cargo run --example hello_brain --no-default-features
cargo run --example neuron_models --no-default-features
cargo run --example stdp_learning --no-default-features

# Benchmarks (criterion)
cargo bench
```

## Feature Flags

| Flag     | Default | Description                                    |
|----------|---------|------------------------------------------------|
| `gui`    | on      | `eframe`/`egui` real-time visualization         |
| `gpu`    | off     | `wgpu` compute backend for LIF + SpMV           |

`--no-default-features` gives a pure CPU headless build suitable for CI / servers.

## Project Layout

```
src/
├── lib.rs                  # Crate root — re-exports all modules
├── main.rs                 # Binary entry point — dispatches to gui_entry or cli
├── cli.rs                  # clap-based CLI + phased demo pipeline
├── gui_entry.rs            # Wires up the GUI when the `gui` feature is on
├── error.rs                # `NeuralSimError` / `Result` aliases
├── prelude.rs              # Convenience re-exports
├── config/                 # YAML/JSON config loading
├── neuron/                # Neuron models (LIF, Izhikevich, Hodgkin-Huxley)
├── synapse/                # Synapse types, dynamics, plasticity rules
├── network/               # CSR graph, brain builder, region templates
├── simulation/             # Engine, scheduler, GPU backend + shaders
├── io/                     # Save/load, checkpointing, text I/O
└── gui/                    # egui app + brain visualization
```

## Conventions

- **Error handling**: use `crate::error::{Result, NeuralSimError}` — do not return `Box<dyn Error>`.
- **SoA storage**: `NeuronArray` is the canonical neuron storage. New code should read/write the parallel `Vec<f64>` fields, not AoS `Vec<NeuronState>`.
- **Parallelism**: use `rayon::par_iter_mut` / `zip_eq` on the SoA slices. See `SimulationEngine::step` for the split-borrow pattern.
- **Determinism**: every `StdRng` must be seeded. The engine seed is recorded in `SimulationStats::seed`.
- **Docs**: public items should have `///` doc comments. Module-level docs go at the top of each `mod.rs`.
- **Tests**: unit tests live in `mod tests` blocks at the bottom of each file; integration tests go in `tests/integration.rs`.
- **No `unwrap()`/`expect()` in library code** except in tests. Use `?` with `NeuralSimError`.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR:
- `cargo build --release --all-features`
- `cargo test --all-features`
- `cargo clippy --all-features -- -D warnings`
- `cargo fmt --check`

Releases are built from `v*.*.*` tags via `.github/workflows/release.yml`.

## Common Tasks

- **Add a neuron model**: create `src/neuron/<name>.rs`, implement `NeuronModel`, add a variant to `NeuronModelParams`, dispatch in `SimulationEngine::step_neuron`.
- **Add a plasticity rule**: add a struct in `src/synapse/plasticity.rs`, wire it into `SimulationEngine::step` under the plasticity phase.
- **Add a brain region template**: extend `RegionTemplate` in `src/network/region.rs`.
- **Add a GPU shader**: drop a `.wgsl` file in `src/simulation/shaders/`, register it in `gpu_backend.rs`.
