//! Demonstrates STDP learning: a network evolves under STDP and we report
//! how the weight distribution changes over time.
//!
//! Run with:
//! ```sh
//! cargo run --example stdp_learning --no-default-features
//! ```

use neural_sim::network::builder::BrainBuilder;
use neural_sim::neuron::NeuronModelParams;
use neural_sim::simulation::SimulationEngine;
use neural_sim::synapse::plasticity::StdpTrace;
use neural_sim::synapse::{PlasticityConfig, plasticity::StdpRule};

fn snapshot_weights(engine: &SimulationEngine) -> (f64, f64, f64) {
    let net = engine.network.read();
    let ws: Vec<f64> = net.synapses.iter().map(|s| s.weight).collect();
    if ws.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = ws.iter().sum::<f64>() / ws.len() as f64;
    let var = ws.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / ws.len() as f64;
    let min = ws.iter().cloned().fold(f64::INFINITY, f64::min);
    (mean, var.sqrt(), min)
}

fn main() {
    let mut network = BrainBuilder::new()
        .add_region(
            "Cortex",
            300,
            0.80,
            NeuronModelParams::Izhikevich {
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 6.0,
            },
        )
        .build();

    // Enable STDP on every synapse.
    for syn in network.synapses.iter_mut() {
        syn.plasticity_enabled = true;
        syn.stdp_trace = Some(StdpTrace::new());
    }

    // Strong LTP/LTD so we can see the distribution shift.
    let plasticity = PlasticityConfig {
        stdp: Some(StdpRule {
            a_plus: 0.05,
            a_minus: 0.06,
            ..Default::default()
        }),
        enabled: true,
        ..Default::default()
    };

    let mut engine = SimulationEngine::new(network)
        .with_noise(15.0)
        .with_plasticity(plasticity);

    let (mean0, std0, min0) = snapshot_weights(&engine);
    println!(
        "Before:  mean={:.4}  std={:.4}  min={:.4}",
        mean0, std0, min0
    );

    let phases = [100.0, 200.0, 500.0, 1000.0];
    let mut sim_time = 0.0;
    for &target in &phases {
        engine.simulate_ms(target - sim_time);
        sim_time = target;
        let (mean, std, min) = snapshot_weights(&engine);
        let stats = engine.stats();
        println!(
            "t={:>5.0} ms  mean={:.4}  std={:.4}  min={:.4}  total_spikes={}",
            target, mean, std, min, stats.total_spikes
        );
    }
}
