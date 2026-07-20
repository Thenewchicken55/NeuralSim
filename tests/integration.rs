use neural_sim::network::builder::BrainBuilder;
use neural_sim::neuron::NeuronModelParams;
use neural_sim::simulation::SimulationEngine;
use neural_sim::synapse::PlasticityConfig;
use neural_sim::synapse::plasticity::{ConsolidationRule, StdpRule};

#[test]
fn test_brain_builder_with_simulation() {
    // Build a multi-region network using BrainBuilder
    let network = BrainBuilder::new()
        .with_name("IntegrationTest")
        .with_plasticity(true)
        .add_region(
            "Input",
            100,
            0.80,
            NeuronModelParams::Izhikevich {
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 8.0,
            },
        )
        .mark_input("Input")
        .add_cortical_column("V1", 300)
        .add_region(
            "Output",
            100,
            0.80,
            NeuronModelParams::Izhikevich {
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 8.0,
            },
        )
        .mark_output("Output")
        .connect_regions("Input", "V1", 0.03, 1.0, None)
        .connect_regions("V1", "Output", 0.03, 1.0, None)
        .build();

    assert_eq!(network.neuron_count(), 500);
    assert!(network.synapse_count() > 100);
    assert_eq!(network.region_names.len(), 3);

    // Configure plasticity
    let plasticity = PlasticityConfig {
        stdp: Some(StdpRule::default()),
        consolidation: Some(ConsolidationRule::default()),
        enabled: true,
        homeostatic_target_rate: 5.0,
        homeostatic_tau: 5000.0,
        ..Default::default()
    };

    // Create engine with conductance-based dynamics and plasticity
    let mut engine = SimulationEngine::new(network)
        .with_noise(10.0)
        .with_plasticity(plasticity)
        .with_conductance(true);

    // Enable plasticity on all synapses
    {
        let mut net = engine.network.write();
        for syn in net.synapses.iter_mut() {
            *syn = syn.clone().with_plasticity();
        }
    }

    // Run simulation
    engine.simulate_ms(100.0);

    let stats = engine.stats();
    assert!(
        stats.sim_time_ms >= 99.0,
        "sim_time_ms={} should be near 100ms",
        stats.sim_time_ms
    );
    assert!(stats.total_spikes > 0, "Engine should produce spikes");
    assert!(stats.mean_firing_rate >= 0.0);
    assert!(stats.synchrony_index >= 0.0);
    assert!(stats.weight_mean > 0.0);
    assert!(
        stats.weight_updates > 0,
        "STDP should produce weight updates"
    );

    // Verify region-specific data
    {
        let net = engine.network.read();
        let counts = net.region_counts();
        assert_eq!(counts.len(), 3);
        assert!(counts.iter().any(|(name, _, _)| name == "Input"));
        assert!(counts.iter().any(|(name, _, _)| name == "V1"));
        assert!(counts.iter().any(|(name, _, _)| name == "Output"));

        let rates = net.region_rates();
        assert_eq!(rates.len(), 3);
    }

    let output_rate = {
        let net = engine.network.read();
        net.region_rates()
            .iter()
            .find(|(name, _)| name == "Output")
            .map(|(_, rate)| *rate)
            .unwrap_or(0.0)
    };
    assert!(
        output_rate >= 0.0,
        "Output region should report firing rate"
    );
}

#[test]
fn test_brain_builder_conductance_simulation() {
    // Smaller test verifying conductance dynamics work end-to-end
    let network = BrainBuilder::new()
        .add_region(
            "A",
            100,
            0.80,
            NeuronModelParams::Izhikevich {
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 8.0,
            },
        )
        .build();

    let mut engine = SimulationEngine::new(network)
        .with_noise(12.0)
        .with_conductance(true);

    engine.simulate_ms(50.0);

    let stats = engine.stats();
    assert!(stats.total_spikes > 0);
    assert!(
        stats.sim_time_ms >= 49.0,
        "sim_time_ms={} should be near 50ms",
        stats.sim_time_ms
    );
}

#[test]
fn test_brain_builder_stdp_weight_change() {
    // Verify that STDP actually changes weights
    use neural_sim::synapse::plasticity::StdpTrace;

    let mut network = BrainBuilder::new()
        .add_region(
            "Test",
            150,
            0.80,
            NeuronModelParams::Izhikevich {
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 6.0,
            },
        )
        .build();

    // Enable plasticity on all synapses
    for syn in network.synapses.iter_mut() {
        syn.plasticity_enabled = true;
        syn.stdp_trace = Some(StdpTrace::new());
    }

    // Record initial weights
    let initial_weights: Vec<f64> = network.synapses.iter().map(|s| s.weight).collect();

    let plasticity = PlasticityConfig {
        stdp: Some(StdpRule {
            a_plus: 0.05,
            a_minus: 0.06,
            tau_plus: 20.0,
            tau_minus: 20.0,
            weight_max: 10.0,
            weight_min: 0.0,
        }),
        enabled: true,
        homeostatic_target_rate: 5.0,
        homeostatic_tau: 5000.0,
        ..Default::default()
    };

    let mut engine = SimulationEngine::new(network)
        .with_noise(15.0)
        .with_plasticity(plasticity);

    engine.simulate_ms(200.0);

    let stats = engine.stats();
    assert!(
        stats.weight_updates > 0,
        "STDP should produce weight updates"
    );

    // Check that at least some weights changed
    let final_network = engine.network.read();
    let final_weights: Vec<f64> = final_network.synapses.iter().map(|s| s.weight).collect();
    let changed = initial_weights
        .iter()
        .zip(final_weights.iter())
        .filter(|&(a, b)| (*a - *b).abs() > 0.001)
        .count();
    assert!(changed > 0, "STDP should change at least some weights");
}
