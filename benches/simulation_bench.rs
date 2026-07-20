use criterion::{Criterion, black_box, criterion_group, criterion_main};
use neural_sim::network::NetworkBuilder;
use neural_sim::simulation::engine::SimulationEngine;

fn bench_step_throughput(c: &mut Criterion) {
    let net = NetworkBuilder::new(1000).with_default_layers().build();
    let mut engine = SimulationEngine::new(net);

    c.bench_function("step_1000_neurons", |b| {
        b.iter(|| {
            black_box(engine.step());
        })
    });
}

fn bench_plasticity_throughput(c: &mut Criterion) {
    let net = NetworkBuilder::new(500).with_default_layers().build();
    let mut engine = SimulationEngine::new(net);

    // Run a few steps to build up spike activity
    for _ in 0..100 {
        engine.step();
    }

    c.bench_function("step_500_with_plasticity", |b| {
        b.iter(|| {
            black_box(engine.step());
        })
    });
}

fn bench_network_build(c: &mut Criterion) {
    c.bench_function("build_network_10000", |b| {
        b.iter(|| {
            let net = NetworkBuilder::new(10000).with_default_layers().build();
            black_box(net);
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(100).measurement_time(std::time::Duration::from_secs(10));
    targets = bench_step_throughput, bench_plasticity_throughput, bench_network_build
);
criterion_main!(benches);
