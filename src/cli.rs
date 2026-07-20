//! Command-line interface for NeuralSim.
//!
//! Splits the previous monolithic `cli_demo` into focused phases:
//!   1. Build the network (from config or default demo)
//!   2. Configure plasticity
//!   3. Run the simulation
//!   4. Exercise text I/O
//!   5. Print a region activity report
//!
//! Each phase is a small function so the control flow is easy to follow
//! and each piece can be tested independently.

use clap::Parser;
use neural_sim::config::SimConfig;
use neural_sim::io::checkpoint::{CheckpointManager, StatsRecorder};
use neural_sim::io::text::{TextDecoder, TextEncoder};
use neural_sim::network::builder::BrainBuilder;
use neural_sim::neuron::NeuronModelParams;
use neural_sim::simulation::SimulationEngine;
use neural_sim::synapse::PlasticityConfig;
use neural_sim::synapse::plasticity::{ConsolidationRule, IntrinsicPlasticity, StdpRule};
use std::path::PathBuf;
use std::time::Instant;

/// Command-line arguments for NeuralSim.
#[derive(Parser, Debug)]
#[command(
    name = "neural_sim",
    version,
    about = "Biologically realistic spiking neural network simulator"
)]
pub struct CliArgs {
    /// Path to a YAML or JSON config file. If omitted, a built-in demo network is used.
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Run in terminal-only mode (skips the GUI even when the `gui` feature is enabled).
    #[arg(long)]
    pub cli: bool,

    /// Override the simulation RNG seed.
    #[arg(long)]
    pub seed: Option<u64>,
}

/// Top-level entry point invoked by `main` when the GUI is not used.
pub fn run(args: CliArgs) {
    print_banner();

    let (sim_config, network, plasticity, sim_duration, dt) = build_network(&args);
    let plasticity = configure_plasticity(plasticity, args.config.is_some());

    let mut engine = build_engine(network, plasticity, &sim_config);

    let (mut checkpoint_mgr, mut stats_recorder) = setup_recording(&sim_config);

    let elapsed = run_simulation(
        &mut engine,
        &mut checkpoint_mgr,
        &mut stats_recorder,
        sim_duration,
        dt,
    );

    exercise_text_io(&mut engine);

    print_region_report(&engine);

    save_stats(&stats_recorder, &sim_config);

    print_footer(&engine, sim_duration, dt, elapsed);
}

fn print_banner() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║        NeuralSim — Biologically Realistic       ║");
    println!("║           Spiking Neural Network Simulator       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
}

/// Build the network either from a config file or from a built-in demo layout.
fn build_network(
    args: &CliArgs,
) -> (
    SimConfig,
    neural_sim::network::Network,
    PlasticityConfig,
    f64,
    f64,
) {
    if let Some(ref cfg_path) = args.config {
        println!("  Loading config from: {}", cfg_path.display());
        let cfg = SimConfig::from_file(&cfg_path.to_string_lossy())
            .unwrap_or_else(|e| panic!("Failed to load config: {e}"));
        let _seed = args.seed.unwrap_or(cfg.seed());
        println!("  Seed: {}", _seed);
        let mut builder = BrainBuilder::new()
            .with_name("ConfigBrain")
            .with_plasticity(true);
        for (name, count, exc_ratio, params, is_input, is_output) in cfg.region_specs() {
            builder = builder.add_region(&name, count, exc_ratio, params);
            if is_input {
                builder = builder.mark_input(&name);
            }
            if is_output {
                builder = builder.mark_output(&name);
            }
        }
        for conn in &cfg.network.connections {
            builder = builder.connect_regions(
                &conn.from,
                &conn.to,
                conn.probability,
                conn.weight_scale.unwrap_or(1.0),
                None,
            );
        }
        let net = builder.build();
        let dt = cfg.dt_ms();
        let duration = cfg.duration_ms();
        let plasticity = cfg.plasticity.clone().unwrap_or_default();
        (cfg, net, plasticity, duration, dt)
    } else {
        println!("[1/5] Building brain with cortical column architecture...");
        let izh = NeuronModelParams::Izhikevich {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        };
        let builder = BrainBuilder::new()
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
        println!("  Neurons: {}", builder.neuron_count());
        println!("  Synapses: {}", builder.synapse_count());
        println!("  Regions: {:?}", builder.region_names);
        (
            SimConfig {
                network: neural_sim::config::NetworkConfig {
                    seed: args.seed,
                    regions: vec![],
                    connections: vec![],
                },
                simulation: neural_sim::config::SimulationConfig {
                    dt_ms: None,
                    duration_ms: None,
                    noise_amplitude: None,
                    use_conductance: None,
                    reward_threshold: None,
                    reward_decay: None,
                },
                plasticity: None,
                io: None,
            },
            builder,
            PlasticityConfig::default(),
            100.0,
            0.5,
        )
    }
}

/// Configure plasticity rules. For the default demo we enable a rich rule set;
/// for config-driven runs we honour whatever the config specified.
fn configure_plasticity(plasticity: PlasticityConfig, from_config: bool) -> PlasticityConfig {
    println!();
    println!("[2/5] Configuring plasticity rules...");
    let plasticity = if from_config {
        plasticity
    } else {
        PlasticityConfig {
            stdp: Some(StdpRule::default()),
            consolidation: Some(ConsolidationRule::default()),
            intrinsic: Some(IntrinsicPlasticity::default()),
            enabled: true,
            homeostatic_target_rate: 5.0,
            homeostatic_tau: 5000.0,
            ..Default::default()
        }
    };
    println!("  Plasticity: enabled={}", plasticity.enabled);
    if let Some(ref stdp) = plasticity.stdp {
        println!("  STDP: A+={:.4}, A-={:.4}", stdp.a_plus, stdp.a_minus);
    }
    if plasticity.rstdp.is_some() {
        println!("  R-STDP: enabled");
    }
    plasticity
}

/// Construct the engine and apply simulation-wide configuration from the config.
fn build_engine(
    network: neural_sim::network::Network,
    plasticity: PlasticityConfig,
    sim_config: &SimConfig,
) -> SimulationEngine {
    println!();
    println!("[3/5] Running simulation with conductance-based dynamics...");

    let noise_amp = sim_config.simulation.noise_amplitude.unwrap_or(8.0);
    let use_cond = sim_config.simulation.use_conductance.unwrap_or(true);

    let mut engine = SimulationEngine::new(network)
        .with_noise(noise_amp)
        .with_plasticity(plasticity)
        .with_conductance(use_cond);

    if let Some(threshold) = sim_config.simulation.reward_threshold {
        let decay = sim_config.simulation.reward_decay.unwrap_or(0.95);
        engine = engine.with_reward(threshold, decay);
    }

    // Enable plasticity on existing synapses.
    {
        let mut net = engine.network.write();
        for syn in net.synapses.iter_mut() {
            if engine.plasticity.enabled {
                *syn = syn.clone().with_plasticity();
            }
        }
    }

    #[cfg(feature = "gpu")]
    {
        match engine.init_gpu() {
            Ok(()) => println!("  Backend: GPU (wgpu)"),
            Err(e) => println!("  Backend: CPU (GPU init failed: {e})"),
        }
    }
    #[cfg(not(feature = "gpu"))]
    println!("  Backend: CPU");

    engine
}

fn setup_recording(sim_config: &SimConfig) -> (CheckpointManager, StatsRecorder) {
    let checkpoint_dir = sim_config
        .io
        .as_ref()
        .and_then(|io| io.checkpoint_dir.clone())
        .unwrap_or_else(|| "/tmp/neural_sim_checkpoints".to_string());
    let checkpoint_mgr = CheckpointManager::new(&checkpoint_dir);
    let stats_recorder = StatsRecorder::new(100, 10);
    (checkpoint_mgr, stats_recorder)
}

/// Run the main simulation loop and emit a per-run summary.
/// Returns the wall-clock elapsed time so callers can report throughput.
fn run_simulation(
    engine: &mut SimulationEngine,
    checkpoint_mgr: &mut CheckpointManager,
    stats_recorder: &mut StatsRecorder,
    sim_duration: f64,
    dt: f64,
) -> std::time::Duration {
    let start = Instant::now();
    let steps = (sim_duration / dt) as usize;

    for s in 0..steps {
        let _result = engine.step();
        stats_recorder.record(s as u64, &engine.stats(), engine.lfp());

        if checkpoint_mgr.should_auto_save(engine.stats().sim_time_ms) {
            checkpoint_mgr.mark_saved(engine.stats().sim_time_ms);
            if s % 500 == 0 {
                let _ = checkpoint_mgr.save_engine(engine, &format!("step_{}", s));
            }
        }
    }
    let elapsed = start.elapsed();
    let stats = engine.stats();
    println!(
        "  Simulated {} ms in {:.2}s real",
        sim_duration,
        elapsed.as_secs_f64()
    );
    println!("  Total spikes: {}", stats.total_spikes);
    println!("  Output spikes: {}", stats.output_spikes);
    println!("  Mean firing rate: {:.1} Hz", stats.mean_firing_rate);
    println!("  Synchrony index: {:.3}", stats.synchrony_index);
    println!(
        "  Weight mean: {:.4} (σ={:.4})",
        stats.weight_mean, stats.weight_std
    );
    println!("  Weight updates: {}", stats.weight_updates);
    elapsed
}

/// Demonstrate the text-encoding pipeline by encoding a string, injecting it
/// as input current, simulating briefly, and decoding output spikes.
fn exercise_text_io(engine: &mut SimulationEngine) {
    println!();
    println!("[4/5] Setting up text I/O pipeline...");
    let encoder = TextEncoder::default(500);
    let decoder = TextDecoder::from_encoder(&encoder);
    println!("  Vocabulary: {} characters", encoder.vocab_size());

    let test_input = "Hello NeuralSim";
    println!("  Encoding: '{}'", test_input);
    let input_spikes = encoder.encode(test_input, engine.stats().sim_time_ms, 50.0);
    {
        let mut net = engine.network.write();
        for &(neuron, _time) in &input_spikes {
            if neuron < net.neurons.input_current.len() {
                net.neurons.input_current[neuron] += 30.0;
            }
        }
    }
    for _ in 0..20 {
        engine.step();
    }
    let output_text = {
        let net = engine.network.read();
        let output_spikes: Vec<(usize, f64)> = (0..net.neuron_count())
            .filter(|&i| net.neurons.is_output[i])
            .flat_map(|i| {
                if net.neurons.just_spiked[i] {
                    vec![(i, net.time)]
                } else {
                    vec![]
                }
            })
            .collect();
        let sim_time = net.time;
        decoder.decode_with_threshold(&output_spikes, 0.0, sim_time, 50.0, 2)
    };
    println!("  Decoded: '{}'", output_text);
}

fn print_region_report(engine: &SimulationEngine) {
    println!();
    println!("[5/5] Region activity report:");
    let net = engine.network.read();
    for (name, count, _) in net.region_counts() {
        println!("  {}: {} neurons", name, count);
    }
    for (name, rate) in net.region_rates() {
        println!("  {} firing rate: {:.1} Hz", name, rate);
    }
}

fn save_stats(stats_recorder: &StatsRecorder, sim_config: &SimConfig) {
    let stats_path = sim_config
        .io
        .as_ref()
        .and_then(|io| io.stats_csv.clone())
        .unwrap_or_else(|| "/tmp/neural_sim_stats.csv".to_string());
    let _ = stats_recorder.save_csv(&stats_path);
    println!();
    println!("  Stats saved to {}", stats_path);
}

fn print_footer(
    engine: &SimulationEngine,
    sim_duration: f64,
    dt: f64,
    elapsed: std::time::Duration,
) {
    let total_steps = (sim_duration / dt) as usize;
    let total_nrn_steps = (engine.network.read().neuron_count() * total_steps) as f64;
    let secs = elapsed.as_secs_f64().max(1e-9);
    println!("  Throughput: {:.0} neuron-steps/s", total_nrn_steps / secs);
    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Simulation complete.                           ║");
    println!("║  Run with --cli for terminal mode.              ║");
    println!("║  Run with --features gpu for GPU acceleration.  ║");
    println!("║  Run with --config <file.yaml> for config.      ║");
    println!("╚══════════════════════════════════════════════════╝");
}
