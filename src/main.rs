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
    println!();

    let mut engine = SimulationEngine::new(network).with_noise(10.0);
    engine.simulate_ms(500.0);

    let stats = engine.stats();
    println!("Simulation complete (500 ms):");
    println!("  Total spikes: {}", stats.total_spikes);
    println!("  Output spikes: {}", stats.output_spikes);
    println!("  Peak active/step: {}", stats.active_neurons);
    if stats.total_spikes > 0 {
        println!("  Avg firing rate: {:.1} Hz",
            (stats.total_spikes as f64 / 5000.0) / (stats.sim_time_ms / 1000.0));
    }
}
