//! Minimal example: build a small brain, run it for 500ms, print stats.
//!
//! Run with:
//! ```sh
//! cargo run --example hello_brain --no-default-features
//! ```

use neural_sim::network::builder::BrainBuilder;
use neural_sim::neuron::NeuronModelParams;
use neural_sim::simulation::SimulationEngine;
use neural_sim::synapse::PlasticityConfig;
use neural_sim::synapse::plasticity::{ConsolidationRule, IntrinsicPlasticity, StdpRule};

fn main() {
    let izh = NeuronModelParams::Izhikevich {
        a: 0.02,
        b: 0.2,
        c: -65.0,
        d: 8.0,
    };

    let network = BrainBuilder::new()
        .with_name("HelloBrain")
        .with_plasticity(true)
        .add_region("Input", 100, 0.80, izh)
        .mark_input("Input")
        .add_cortical_column("V1", 500)
        .add_region("Output", 100, 0.80, izh)
        .mark_output("Output")
        .connect_regions("Input", "V1", 0.05, 1.0, None)
        .connect_regions("V1", "Output", 0.05, 1.0, None)
        .build();

    println!(
        "Built network: {} neurons, {} synapses",
        network.neuron_count(),
        network.synapse_count()
    );

    let plasticity = PlasticityConfig {
        stdp: Some(StdpRule::default()),
        consolidation: Some(ConsolidationRule::default()),
        intrinsic: Some(IntrinsicPlasticity::default()),
        enabled: true,
        homeostatic_target_rate: 5.0,
        homeostatic_tau: 5000.0,
        ..Default::default()
    };

    let mut engine = SimulationEngine::new(network)
        .with_noise(10.0)
        .with_plasticity(plasticity)
        .with_conductance(true);

    // Enable plasticity on all synapses.
    {
        let mut net = engine.network.write();
        for syn in net.synapses.iter_mut() {
            *syn = syn.clone().with_plasticity();
        }
    }

    engine.simulate_ms(500.0);

    let stats = engine.stats();
    println!("Simulated {:.1} ms", stats.sim_time_ms);
    println!("  Total spikes:    {}", stats.total_spikes);
    println!("  Output spikes:   {}", stats.output_spikes);
    println!("  Mean firing rate: {:.1} Hz", stats.mean_firing_rate);
    println!("  Synchrony:       {:.3}", stats.synchrony_index);
    println!("  Weight mean:     {:.4}", stats.weight_mean);
}
