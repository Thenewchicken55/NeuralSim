#[cfg(feature = "gui")]
pub fn run() {
    use neural_sim::gui::NeuralSimApp;
    use neural_sim::network::Network;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut network = Network::new(2500);
    let mut rng = StdRng::seed_from_u64(42);
    network.connect_random(0.03, &mut rng);

    // Inject initial random currents to kickstart activity
    let mut rng = StdRng::seed_from_u64(123);
    for c in network.neurons.input_current.iter_mut() {
        *c = rng.random::<f64>() * 15.0;
    }

    let _ = NeuralSimApp::run(network);
}
