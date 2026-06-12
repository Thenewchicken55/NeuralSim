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
    network.connect_random(0.03, &mut rng);

    // Inject initial random currents to kickstart activity
    for c in network.neurons.input_current.iter_mut() {
        *c = rng.random::<f64>() * 12.0;
    }

    println!("Neurons: {}", network.neuron_count());
    println!("Synapses: {}", network.synapse_count());
    println!();

    let mut engine = SimulationEngine::new(network).with_noise(8.0);
    engine.simulate_ms(500.0);

    let stats = engine.stats();
    println!("Simulation complete:");
    println!("  Time simulated: {:.1} ms", stats.sim_time_ms);
    println!("  Total spikes: {}", stats.total_spikes);
    println!("  Peak active neurons per step: {}", stats.active_neurons);
    if stats.total_spikes > 0 {
        println!("  Avg firing rate: {:.1} Hz",
            (stats.total_spikes as f64 / 5000.0) / (stats.sim_time_ms / 1000.0));
    }
}
