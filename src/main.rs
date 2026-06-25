#[cfg(feature = "gui")]
mod gui_entry;

fn main() {
    #[cfg(feature = "gui")]
    if !std::env::args().any(|a| a == "--cli") {
        gui_entry::run();
        return;
    }

    cli_demo();
}

fn cli_demo() {
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

    // ── Demo 1: Brain Builder with cortical columns ──
    println!("[1/5] Building brain with cortical column architecture...");
    let network = BrainBuilder::new()
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
    println!("  Neurons: {}", network.neuron_count());
    println!("  Synapses: {}", network.synapse_count());
    println!("  Regions: {:?}", network.region_names);

    // ── Demo 2: Configure plasticity ──
    println!();
    println!("[2/5] Configuring plasticity rules...");
    let plasticity = PlasticityConfig {
        stdp: Some(StdpRule::default()),
        consolidation: Some(ConsolidationRule::default()),
        intrinsic: Some(IntrinsicPlasticity::default()),
        enabled: true,
        homeostatic_target_rate: 5.0,
        homeostatic_tau: 5000.0,
        ..Default::default()
    };
    println!("  STDP: enabled (A+={:.4}, A-={:.4})",
        plasticity.stdp.as_ref().map(|s| s.a_plus).unwrap_or(0.0),
        plasticity.stdp.as_ref().map(|s| s.a_minus).unwrap_or(0.0));

    // ── Demo 3: Run simulation ──
    println!();
    println!("[3/5] Running simulation with conductance-based dynamics...");
    let mut engine = SimulationEngine::new(network)
        .with_noise(8.0)
        .with_plasticity(plasticity)
        .with_conductance(true);

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
    let mut checkpoint_mgr = CheckpointManager::new("/tmp/neural_sim_checkpoints");
    let mut stats_recorder = StatsRecorder::new(100, 10);

    // Simulate
    let start = Instant::now();
    let sim_duration = 100.0;
    let steps = (sim_duration / engine.dt) as usize;

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
    let _ = stats_recorder.save_csv("/tmp/neural_sim_stats.csv");
    println!();
    println!("  Stats saved to /tmp/neural_sim_stats.csv");

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
    println!("╚══════════════════════════════════════════════════╝");
}

