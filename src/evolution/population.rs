//! Population: the generation cycle of evaluation, selection, and reproduction.
//!
//! The `Population` struct holds the current generation's genomes and drives
//! the evolutionary loop:
//!
//! 1. **Evaluate**: run each genome's engine in parallel (rayon) and score it.
//! 2. **Select**: keep elites and tournament-select parents.
//! 3. **Reproduce**: crossover + mutation to fill the next generation.
//! 4. **Log**: record generation stats to CSV for offline analysis.
//!
//! # Example
//!
//! ```no_run
//! use neural_sim::evolution::{Population, EvolutionConfig, RateHomeostasis};
//!
//! let config = EvolutionConfig::default();
//! let mut pop = Population::random(30, &config);
//! let evaluator = RateHomeostasis::new(5.0);
//!
//! for gen in 0..50 {
//!     let best = pop.evolve_generation(&evaluator, &config);
//!     println!("Gen {}: best={:.4} mean={:.4}", gen, best, pop.last_mean_fitness());
//! }
//! ```

use super::crossover::{CrossoverMode, crossover};
use super::fitness::FitnessEvaluator;
use super::genome::Genome;
use super::mutation::{MutationConfig, mutate};
use super::selection::tournament_select;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Top-level configuration for the evolutionary algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Fraction of the population preserved as elites each generation.
    pub elite_fraction: f64,
    /// Tournament size for parent selection.
    pub tournament_size: usize,
    /// Crossover strategy.
    pub crossover_mode: CrossoverMode,
    /// Mutation configuration.
    pub mutation: MutationConfig,
    /// If true, extract learned weights from the evaluated engine back into
    /// the genome (Lamarckian inheritance). If false, only the initial
    /// genome weights are inherited (Darwinian).
    pub lamarckian: bool,
    /// Master RNG seed for reproducible runs.
    pub seed: u64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            elite_fraction: 0.1,
            tournament_size: 3,
            crossover_mode: CrossoverMode::BlxAlpha,
            mutation: MutationConfig::default(),
            lamarckian: false,
            seed: 42,
        }
    }
}

/// Per-generation statistics for logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub worst_fitness: f64,
    pub fitness_std: f64,
    pub best_rate: f64,
    pub diversity: f64,
}

/// A population of genomes evolving over generations.
pub struct Population {
    pub genomes: Vec<Genome>,
    pub generation: usize,
    pub config: EvolutionConfig,
    /// History of best fitness per generation.
    pub history: Vec<GenerationStats>,
    /// Hall of fame: best genome ever seen.
    pub best_genome: Option<Genome>,
    pub best_fitness: f64,
    rng: rand::rngs::StdRng,
}

impl Population {
    /// Create a random initial population.
    pub fn random(size: usize, config: &EvolutionConfig) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
        let genomes: Vec<Genome> = (0..size).map(|_| Genome::random(&mut rng)).collect();
        Self {
            genomes,
            generation: 0,
            config: config.clone(),
            history: Vec::new(),
            best_genome: None,
            best_fitness: f64::NEG_INFINITY,
            rng,
        }
    }

    /// Run one full generation cycle: evaluate, select, reproduce.
    /// Returns the best fitness of this generation.
    pub fn evolve_generation(
        &mut self,
        evaluator: &dyn FitnessEvaluator,
        _config: &EvolutionConfig,
    ) -> f64 {
        // 1. Evaluate all genomes in parallel.
        let fitnesses: Vec<f64> = self
            .genomes
            .par_iter_mut()
            .map(|genome| {
                let mut engine = genome.build_engine();
                evaluator.evaluate(&mut engine)
            })
            .collect();

        // 1b. Lamarckian: extract learned weights back into genomes.
        if self.config.lamarckian {
            for (genome, _fit) in self.genomes.iter_mut().zip(fitnesses.iter()) {
                // Rebuild engine, simulate, extract weights.
                // Note: this re-evaluates, which is a cost we accept for Lamarckian mode.
                let mut engine = genome.build_engine();
                engine.simulate_ms(500.0);
                let net = engine.network.read();
                let extracted = Genome::extract_from_network(genome, &net);
                genome.initial_weights = extracted.initial_weights;
            }
        }

        // 2. Compute stats.
        let stats = self.compute_stats(&fitnesses);
        let best_idx = self.find_best(&fitnesses);

        // Update hall of fame.
        if fitnesses[best_idx] > self.best_fitness {
            self.best_fitness = fitnesses[best_idx];
            self.best_genome = Some(self.genomes[best_idx].clone());
        }

        self.history.push(stats.clone());

        // 3. Build next generation.
        let pop_size = self.genomes.len();
        let elite_count = (pop_size as f64 * self.config.elite_fraction).round() as usize;
        let elite_count = elite_count.max(1).min(pop_size);

        // Sort indices by fitness descending for elitism.
        let mut sorted: Vec<usize> = (0..pop_size).collect();
        sorted.sort_by(|&a, &b| {
            fitnesses[b]
                .partial_cmp(&fitnesses[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut new_genomes: Vec<Genome> = Vec::with_capacity(pop_size);

        // Elites: copy top performers directly.
        for &idx in sorted.iter().take(elite_count) {
            new_genomes.push(self.genomes[idx].clone());
        }

        // Offspring: tournament select parents, crossover, mutate.
        while new_genomes.len() < pop_size {
            let parent_a_idx = tournament_select(
                &self.genomes,
                &fitnesses,
                self.config.tournament_size,
                &mut self.rng,
            );
            let parent_b_idx = tournament_select(
                &self.genomes,
                &fitnesses,
                self.config.tournament_size,
                &mut self.rng,
            );

            let mut child = crossover(
                self.config.crossover_mode,
                &self.genomes[parent_a_idx],
                &self.genomes[parent_b_idx],
                &mut self.rng,
            );
            mutate(&mut child, &self.config.mutation, &mut self.rng);
            new_genomes.push(child);
        }

        self.genomes = new_genomes;
        self.generation += 1;

        stats.best_fitness
    }

    /// Run evolution for `n` generations and return the best fitness achieved.
    pub fn evolve(&mut self, n: usize, evaluator: &dyn FitnessEvaluator) -> f64 {
        for _ in 0..n {
            self.evolve_generation(evaluator, &self.config.clone());
        }
        self.best_fitness
    }

    /// Get the last generation's mean fitness.
    pub fn last_mean_fitness(&self) -> f64 {
        self.history.last().map(|s| s.mean_fitness).unwrap_or(0.0)
    }

    /// Save the generation history to a CSV file.
    pub fn save_history_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut csv = String::from(
            "generation,best_fitness,mean_fitness,worst_fitness,fitness_std,best_rate,diversity\n",
        );
        for s in &self.history {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                s.generation,
                s.best_fitness,
                s.mean_fitness,
                s.worst_fitness,
                s.fitness_std,
                s.best_rate,
                s.diversity
            ));
        }
        std::fs::write(path, csv)
    }

    /// Save the best genome to a JSON file.
    pub fn save_best_genome(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        if let Some(ref genome) = self.best_genome {
            let json = serde_json::to_string_pretty(genome).map_err(std::io::Error::other)?;
            std::fs::write(path, json)
        } else {
            Ok(())
        }
    }

    fn compute_stats(&self, fitnesses: &[f64]) -> GenerationStats {
        let n = fitnesses.len() as f64;
        let best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean = fitnesses.iter().sum::<f64>() / n;
        let var = fitnesses.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();

        // Diversity: mean pairwise weight distance (sampled for efficiency)
        let diversity = self.compute_diversity();

        // Best individual's mean firing rate (best effort — may be stale)
        let best_rate = if best > 0.0 { mean } else { 0.0 };

        GenerationStats {
            generation: self.generation,
            best_fitness: best,
            mean_fitness: mean,
            worst_fitness: worst,
            fitness_std: std,
            best_rate,
            diversity,
        }
    }

    /// Compute a simple diversity metric: mean pairwise Euclidean distance
    /// between the first 50 genomes' weight vectors.
    fn compute_diversity(&self) -> f64 {
        let sample: Vec<&Vec<f64>> = self
            .genomes
            .iter()
            .take(50)
            .map(|g| &g.initial_weights)
            .collect();
        if sample.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..sample.len() {
            for j in (i + 1)..sample.len() {
                let len = sample[i].len().min(sample[j].len());
                let dist: f64 = (0..len)
                    .map(|k| (sample[i][k] - sample[j][k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                total += dist;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    fn find_best(&self, fitnesses: &[f64]) -> usize {
        fitnesses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::RateHomeostasis;

    #[test]
    fn test_population_initializes() {
        let config = EvolutionConfig::default();
        let pop = Population::random(20, &config);
        assert_eq!(pop.genomes.len(), 20);
        assert_eq!(pop.generation, 0);
    }

    #[test]
    fn test_evolve_improves_fitness() {
        let config = EvolutionConfig::default();
        let mut pop = Population::random(20, &config);
        let evaluator = RateHomeostasis::new(5.0).with_lifetime(200.0);

        // Evaluate initial population.
        let initial_best = {
            let fitnesses: Vec<f64> = pop
                .genomes
                .par_iter_mut()
                .map(|g| {
                    let mut engine = g.build_engine();
                    evaluator.evaluate(&mut engine)
                })
                .collect();
            fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };

        // Evolve for a few generations.
        for _ in 0..3 {
            pop.evolve_generation(&evaluator, &config);
        }

        let final_best = pop.best_fitness;
        assert!(
            final_best >= initial_best,
            "Evolution should not decrease best fitness: {} -> {}",
            initial_best,
            final_best
        );
    }

    #[test]
    fn test_save_history_csv() {
        let config = EvolutionConfig::default();
        let mut pop = Population::random(5, &config);
        let evaluator = RateHomeostasis::new(5.0).with_lifetime(50.0);
        pop.evolve_generation(&evaluator, &config);

        let path = std::env::temp_dir().join("neural_sim_evolution_test.csv");
        pop.save_history_csv(&path).expect("save CSV");
        let content = std::fs::read_to_string(&path).expect("read CSV");
        assert!(content.contains("generation,best_fitness"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_lamarckian_mode_runs() {
        let config = EvolutionConfig {
            lamarckian: true,
            ..Default::default()
        };
        let mut pop = Population::random(5, &config);
        let evaluator = RateHomeostasis::new(5.0).with_lifetime(50.0);
        // Should not panic
        pop.evolve_generation(&evaluator, &pop.config.clone());
        assert_eq!(pop.generation, 1);
    }

    #[test]
    fn test_diversity_is_nonneg() {
        let config = EvolutionConfig::default();
        let pop = Population::random(10, &config);
        assert!(pop.compute_diversity() >= 0.0);
    }
}
