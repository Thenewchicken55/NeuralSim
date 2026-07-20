//! Demonstrates different neuron models (LIF, Izhikevich, Hodgkin-Huxley)
//! by running each in isolation and comparing firing rates.
//!
//! Run with:
//! ```sh
//! cargo run --example neuron_models --no-default-features
//! ```

use neural_sim::network::Network;
use neural_sim::neuron::{NeuronModelParams, NeuronType};
use neural_sim::simulation::SimulationEngine;
use neural_sim::synapse::SynapseState;
use rand::{Rng, SeedableRng};

fn build_ring(n: usize, params: NeuronModelParams) -> Network {
    let mut net = Network::new(n);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    for i in 0..n {
        net.neurons.model_params[i] = params;
        net.neurons.neuron_type[i] = if rng.random::<f64>() < 0.8 {
            NeuronType::Excitatory
        } else {
            NeuronType::Inhibitory
        };
    }
    // Ring of excitatory connections
    for i in 0..n {
        let target = (i + 1) % n;
        net.add_synapse(SynapseState::new(
            i,
            target,
            1.0,
            neural_sim::synapse::SynapseType::AMPA,
        ));
    }
    net.finalize();
    net
}

fn run_model(label: &str, params: NeuronModelParams) {
    let net = build_ring(200, params);
    let mut engine = SimulationEngine::new(net).with_noise(12.0);
    engine.simulate_ms(300.0);
    let stats = engine.stats();
    println!(
        "{:20} total_spikes={:<6} mean_rate={:6.1} Hz  sync={:.3}",
        label, stats.total_spikes, stats.mean_firing_rate, stats.synchrony_index
    );
}

fn main() {
    println!("Comparing neuron models (200 neurons, 300 ms, ring topology):\n");
    run_model("LIF", NeuronModelParams::default());
    run_model(
        "Izhikevich",
        NeuronModelParams::Izhikevich {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        },
    );
    run_model(
        "Hodgkin-Huxley",
        NeuronModelParams::HodgkinHuxley {
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.387,
            c_m: 1.0,
        },
    );
}
