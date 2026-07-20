//! Evolution example: evolve a population of brains to hold a target firing rate.
//!
//! Run with:
//! ```sh
//! cargo run --example evolution_demo --no-default-features
//! ```

use neural_sim::evolution::{EvolutionConfig, Population, RateHomeostasis};

fn main() {
    let config = EvolutionConfig::default();
    let mut pop = Population::random(20, &config);
    let evaluator = RateHomeostasis::new(5.0).with_lifetime(300.0);

    println!("Evolution: target 5 Hz mean firing rate, 20 brains, 20 generations\n");

    for generation in 0..20 {
        pop.evolve_generation(&evaluator, &config);
        let stats = pop.history.last().expect("history");
        println!(
            "Gen {:>2}: best={:.4}  mean={:.4}  std={:.4}  diversity={:.1}",
            generation, stats.best_fitness, stats.mean_fitness, stats.fitness_std, stats.diversity,
        );
    }

    // Save the best brain
    let _ = pop.save_best_genome("/tmp/neural_sim_evolved_brain.json");
    println!("\nBest genome saved to /tmp/neural_sim_evolved_brain.json");
}
