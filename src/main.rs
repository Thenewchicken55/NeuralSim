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

    println!("NeuralSim — Headless Mode");
    println!("=========================\n");

    // Create a small test network
    let mut network = Network::new(1000);

    // Connect with Erdos-Renyi random graph (p=0.05)
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(42);
    network.connect_random(0.05, &mut rng);

    println!("Neurons: {}", network.neuron_count());
    println!("Synapses: {}", network.synapse_count());
    println!();

    // Run simulation
    let mut engine = SimulationEngine::new(network);
    engine.simulate_ms(100.0);

    let stats = engine.stats();
    println!("Simulation complete:");
    println!("  Time simulated: 100 ms");
    println!("  Total spikes: {}", stats.total_spikes);
    println!("  Peak active neurons: {}", stats.active_neurons);
}
