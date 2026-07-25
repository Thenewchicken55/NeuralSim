#[cfg(feature = "gui")]
pub fn run() {
    use neural_sim::gui::NeuralSimApp;
    use neural_sim::network::builder::BrainBuilder;
    use neural_sim::neuron::NeuronModelParams;
    #[cfg(feature = "gpu")]
    use neural_sim::simulation::SimulationEngine;

    // Build a multi-region brain network for better visualization
    let network = BrainBuilder::new()
        .add_region("Visual Cortex", 400, 0.80, NeuronModelParams::default())
        .add_region("Auditory Cortex", 300, 0.80, NeuronModelParams::default())
        .add_region("Motor Cortex", 350, 0.80, NeuronModelParams::default())
        .add_region("Hippocampus", 200, 0.90, NeuronModelParams::default())
        .add_region("Output", 80, 0.80, NeuronModelParams::default())
        .connect_regions("Visual Cortex", "Hippocampus", 0.05, 1.0, None)
        .connect_regions("Auditory Cortex", "Hippocampus", 0.05, 1.0, None)
        .connect_regions("Motor Cortex", "Hippocampus", 0.05, 1.0, None)
        .connect_regions("Hippocampus", "Output", 0.1, 1.5, None)
        .build();


    #[cfg(feature = "gpu")]
    {
        let mut engine = SimulationEngine::new(network.clone());
        match engine.init_gpu() {
            Ok(()) => {
                eprintln!("GPU backend initialized successfully");
                let _ = NeuralSimApp::run_with_engine(engine);
            }
            Err(e) => {
                eprintln!("GPU init failed ({e}), falling back to CPU");
                // Rebuild network since we moved it
                let fallback = BrainBuilder::new()
                    .add_region("Visual Cortex", 400, 0.80, NeuronModelParams::default())
                    .add_region("Auditory Cortex", 300, 0.80, NeuronModelParams::default())
                    .add_region("Motor Cortex", 350, 0.80, NeuronModelParams::default())
                    .add_region("Hippocampus", 200, 0.90, NeuronModelParams::default())
                    .add_region("Output", 80, 0.80, NeuronModelParams::default())
                    .connect_regions("Visual Cortex", "Hippocampus", 0.05, 1.0, None)
                    .connect_regions("Auditory Cortex", "Hippocampus", 0.05, 1.0, None)
                    .connect_regions("Motor Cortex", "Hippocampus", 0.05, 1.0, None)
                    .connect_regions("Hippocampus", "Output", 0.1, 1.5, None)
                    .build();
                let _ = NeuralSimApp::run(fallback);
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = NeuralSimApp::run(network);
    }
}
