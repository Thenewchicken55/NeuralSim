#[cfg(feature = "gui")]
pub fn run() {
    use neural_sim::gui::NeuralSimApp;
    use neural_sim::network::Network;
    use neural_sim::simulation::SimulationEngine;
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

    // Small gentle nudge to a few neurons, not a flood
    for _ in 0..20 {
        let idx = rng.random_range(0..n);
        network.neurons.input_current[idx] = 15.0;
    }

    #[cfg(feature = "gpu")]
    {
        let mut engine = SimulationEngine::new(network);
        match engine.init_gpu() {
            Ok(()) => {
                eprintln!("GPU backend initialized successfully");
                let _ = NeuralSimApp::run_with_engine(engine);
            }
            Err(e) => {
                eprintln!("GPU init failed ({e}), falling back to CPU");
                // Rebuild network since we moved it
                let mut fallback = Network::new(1600);
                let mut rng2 = StdRng::seed_from_u64(42);
                fallback.connect_random(0.04, &mut rng2);
                let m = fallback.neuron_count();
                let out_start = m - m / 20;
                for i in out_start..m {
                    fallback.neurons.is_output[i] = true;
                }
                for _ in 0..20 {
                    let idx = rng2.random_range(0..m);
                    fallback.neurons.input_current[idx] = 15.0;
                }
                let _ = NeuralSimApp::run(fallback);
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = NeuralSimApp::run(network);
    }
}
