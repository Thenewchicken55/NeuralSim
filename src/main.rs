#[cfg(feature = "gui")]
mod gui_entry;

fn main() {
    // Parse CLI arguments
    let args: Vec<String> = std::env::args().collect();
    let mut config_path: Option<String> = None;
    let mut cli_mode = false;
    let mut seed: Option<u64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                i += 1;
                config_path = Some(args[i].clone());
            }
            "--cli" => cli_mode = true,
            "--seed" => {
                i += 1;
                seed = Some(args[i].parse().expect("--seed requires a u64 integer"));
            }
            _ => {}
        }
        i += 1;
    }

    #[cfg(feature = "gui")]
    if !cli_mode {
        gui_entry::run();
        return;
    }

    cli_demo(config_path, seed);
}

fn cli_demo(config_path: Option<String>, cli_seed: Option<u64>) {
    use neural_sim::config::SimConfig;
    use neural_sim::network::builder::BrainBuilder;
    use neural_sim::simulation::SimulationEngine;
    use neural_sim::synapse::PlasticityConfig;
    use neural_sim::synapse::plasticity::{StdpRule, ConsolidationRule, IntrinsicPlasticity};
    use neural_sim::io::text::{TextEncoder, TextDecoder};
    use neural_sim::io::checkpoint::{CheckpointManager, StatsRecorder};
    use std::time::Instant;

    println!("╔══════════════════════════════════════════════════╗");
    println!("║        NeuralSim — Biologically Realistic       ║");
    println!("║           Spiking Neural Network Simulator       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    // Load config or use defaults
    let (sim_config, network, plasticity, sim_duration, dt) = if let Some(ref cfg_path) = config_path {
        println!("  Loading config from: {}", cfg_path);
        let cfg = SimConfig::from_file(cfg_path).expect("Failed to load config");
        let seed = cli_seed.unwrap_or(cfg.seed());
        println!("  Seed: {}", seed);
        let mut builder = BrainBuilder::new()
            .with_name("ConfigBrain")
            .with_plasticity(true);
        for (name, count, exc_ratio, params, is_input, is_output) in cfg.region_specs() {
            builder = builder.add_region(&name, count, exc_ratio, params);
            if is_input { builder = builder.mark_input(&name); }
            if is_output { builder = builder.mark_output(&name); }
        }
        for conn in &cfg.network.connections {
            builder = builder.connect_regions(&conn.from, &conn.to, conn.probability, conn.weight_scale.unwrap_or(1.0), None);
        }
        let net = builder.build();
        let dt = cfg.dt_ms();
        let duration = cfg.duration_ms();
        let plasticity = cfg.plasticity.clone().unwrap_or_default();
        (cfg, net, plasticity, duration, dt)
    } else {
        // Default demo config
        println!("[1/5] Building brain with cortical column architecture...");
        let builder = BrainBuilder::new()
            .with_name("DemoBrain")
            .with_plasticity(true)
            .add_region("Input", 100, 0.80, neural_sim::neuron::NeuronModelParams::Izhikevich {
                a: 0.02, b: 0.2, c: -65.0, d: 8.0,
            })
            .mark_input("Input")
            .add_cortical_column("V1", 500)
            .add_region("Output", 100, 0.80, neural_sim::neuron::NeuronModelParams::Izhikevich {
                a: 0.02, b: 0.2, c: -65.0, d: 8.0,
            })
            .mark_output("Output")
            .connect_regions("Input", "V1", 0.03, 1.0, None)
            .connect_regions("V1", "Output", 0.03, 1.0, None)
            .build();
        println!("  Neurons: {}", builder.neuron_count());
        println!("  Synapses: {}", builder.synapse_count());
        println!("  Regions: {:?}", builder.region_names);
        (SimConfig {
            network: neural_sim::config::NetworkConfig {
                seed: cli_seed,
                regions: vec![],
                connections: vec![],
            },
            simulation: neural_sim::config::SimulationConfig {
                dt_ms: None, duration_ms: None, noise_amplitude: None,
                use_conductance: None, reward_threshold: None, reward_decay: None,
            },
            plasticity: None, io: None,
        }, builder, PlasticityConfig::default(), 100.0, 0.5)
    };

    // ── Demo 2: Configure plasticity ──
    println!();
    println!("[2/5] Configuring plasticity rules...");
    let plasticity = if config_path.is_some() { plasticity } else {
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

    // ── Demo 3: Run simulation ──
    println!();
    println!("[3/5] Running simulation with conductance-based dynamics...");

    let noise_amp = sim_config.simulation.noise_amplitude.unwrap_or(8.0);
    let use_cond = sim_config.simulation.use_conductance.unwrap_or(true);

    let mut engine = SimulationEngine::new(network)
        .with_noise(noise_amp)
        .with_plasticity(plasticity)
        .with_conductance(use_cond);

    // Apply reward config from simulation
    if let Some(threshold) = sim_config.simulation.reward_threshold {
        let decay = sim_config.simulation.reward_decay.unwrap_or(0.95);
        engine = engine.with_reward(threshold, decay);
    }

    // Enable plasticity on existing synapses
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

    // Checkpoint manager
    let checkpoint_dir = sim_config.io.as_ref()
        .and_then(|io| io.checkpoint_dir.clone())
        .unwrap_or_else(|| "/tmp/neural_sim_checkpoints".to_string());
    let mut checkpoint_mgr = CheckpointManager::new(&checkpoint_dir);
    let mut stats_recorder = StatsRecorder::new(100, 10);

    // Simulate
    let start = Instant::now();
    let sim_duration = sim_duration;
    let steps = (sim_duration / dt) as usize;

    for s in 0..steps {
        let _result = engine.step();

        // Record stats periodically
        stats_recorder.record(s as u64, &engine.stats(), engine.lfp());

        // Auto-save
        if checkpoint_mgr.should_auto_save(engine.stats().sim_time_ms) {
            checkpoint_mgr.mark_saved(engine.stats().sim_time_ms);
            if s % 500 == 0 {
                let _ = checkpoint_mgr.save_engine(&engine, &format!("step_{}", s));
            }
        }
    }
    let elapsed = start.elapsed();
    let stats = engine.stats();
    println!("  Simulated {} ms in {:.2}s real", sim_duration, elapsed.as_secs_f64());
    println!("  Total spikes: {}", stats.total_spikes);
    println!("  Output spikes: {}", stats.output_spikes);
    println!("  Mean firing rate: {:.1} Hz", stats.mean_firing_rate);
    println!("  Synchrony index: {:.3}", stats.synchrony_index);
    println!("  Weight mean: {:.4} (σ={:.4})", stats.weight_mean, stats.weight_std);
    println!("  Weight updates: {}", stats.weight_updates);

    // ── Demo 4: Text I/O ──
    println!();
    println!("[4/5] Setting up text I/O pipeline...");
    let encoder = TextEncoder::default(500);
    let _decoder = TextDecoder::from_encoder(&encoder);
    println!("  Vocabulary: {} characters", encoder.vocab_size());

    // Show region rates
    println!();
    println!("[5/5] Region activity report:");
    {
        let net = engine.network.read();
        for (name, count, _) in net.region_counts() {
            println!("  {}: {} neurons", name, count);
        }
        for (name, rate) in net.region_rates() {
            println!("  {} firing rate: {:.1} Hz", name, rate);
        }
    }

    // Save recordings
    let stats_path = sim_config.io.as_ref()
        .and_then(|io| io.stats_csv.clone())
        .unwrap_or_else(|| "/tmp/neural_sim_stats.csv".to_string());
    let _ = stats_recorder.save_csv(&stats_path);
    println!();
    println!("  Stats saved to {}", stats_path);

    // Performance
    let total_steps = steps;
    let total_nrn_steps = (engine.network.read().neuron_count() * total_steps) as f64;
    println!("  Throughput: {:.0} neuron-steps/s",
        total_nrn_steps / elapsed.as_secs_f64());
    println!();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Simulation complete.                           ║");
    println!("║  Run with --cli for terminal mode.              ║");
    println!("║  Run with --features gpu for GPU acceleration.  ║");
    println!("║  Run with --config <file.yaml> for config.      ║");
    println!("╚══════════════════════════════════════════════════╝");
}

