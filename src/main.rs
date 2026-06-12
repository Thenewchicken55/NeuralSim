#[cfg(feature = "gui")]
mod gui_entry;

fn main() {
    #[cfg(feature = "gui")]
    {
        gui_entry::run();
    }

    #[cfg(not(feature = "gui"))]
    {
        cli_entry();
    }
}

#[cfg(not(feature = "gui"))]
fn cli_entry() {
    use neural_sim::network::Network;
    use neural_sim::simulation::SimulationEngine;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::time::Instant;

    println!("NeuralSim — Headless Mode");
    println!("=========================\n");

    let mut network = Network::new(5000);
    let mut rng = StdRng::seed_from_u64(42);
    network.connect_random(0.04, &mut rng);

    // Mark some neurons as output
    let n = network.neuron_count();
    let output_start = n - n / 20;
    for i in output_start..n {
        network.neurons.is_output[i] = true;
    }

    // Kickstart
    for c in network.neurons.input_current.iter_mut() {
        *c = rng.random::<f64>() * 20.0;
    }

    println!("Neurons: {}", network.neuron_count());
    println!("Synapses: {}", network.synapse_count());
    println!("Output neurons: {}", network.neurons.is_output.iter().filter(|&&x| x).count());

    let mut engine = SimulationEngine::new(network).with_noise(10.0);

    #[cfg(feature = "gpu")]
    {
        match engine.init_gpu() {
            Ok(()) => println!("Backend: GPU (wgpu)"),
            Err(e) => println!("Backend: CPU (GPU init failed: {e})"),
        }
    }
    #[cfg(not(feature = "gpu"))]
    println!("Backend: CPU (compile with --features gpu for GPU acceleration)");

    println!();

    let start = Instant::now();
    engine.simulate_ms(500.0);
    let elapsed = start.elapsed();

    let stats = engine.stats();
    let steps = (500.0 / engine.dt) as u64;
    println!("Simulation complete (500 ms simulated, {steps} steps, {:.2}s real):", elapsed.as_secs_f64());
    println!("  Total spikes: {}", stats.total_spikes);
    println!("  Output spikes: {}", stats.output_spikes);
    println!("  Peak active/step: {}", stats.active_neurons);
    if stats.total_spikes > 0 {
        println!("  Avg firing rate: {:.1} Hz",
            (stats.total_spikes as f64 / 5000.0) / (stats.sim_time_ms / 1000.0));
    }
    println!("  Throughput: {:.0} neuron-steps/s",
        (5000 * steps) as f64 / elapsed.as_secs_f64());
}
