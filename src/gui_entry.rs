#[cfg(feature = "gui")]
pub fn run() {
    use neural_sim::gui::NeuralSimApp;
    use neural_sim::network::Network;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut network = Network::new(1600);
    let mut rng = StdRng::seed_from_u64(42);
    network.connect_random(0.04, &mut rng);

    // Mark the last 5% of neurons as output neurons
    let n = network.neuron_count();
    let output_start = n - n / 20;
    for i in output_start..n {
        network.neurons.is_output[i] = true;
    }

    // Kickstart a burst of random activity
    for c in network.neurons.input_current.iter_mut() {
        *c = rng.random::<f64>() * 20.0;
    }

    let _ = NeuralSimApp::run(network);
}
