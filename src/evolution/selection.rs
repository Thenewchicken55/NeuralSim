//! Selection operators for choosing parents from a population.
//!
//! - [`tournament_select`]: pick `k` random individuals and return the one
//!   with the highest fitness. Adjustable selection pressure via `k`.

use super::genome::Genome;
use rand::Rng;

/// Tournament selection: sample `k` individuals at random and return the
/// fittest among them. Higher `k` = stronger selection pressure.
///
/// Returns the index of the selected individual.
pub fn tournament_select(
    genomes: &[Genome],
    fitnesses: &[f64],
    k: usize,
    rng: &mut impl Rng,
) -> usize {
    debug_assert!(!genomes.is_empty());
    debug_assert_eq!(genomes.len(), fitnesses.len());

    let n = genomes.len();
    let k = k.min(n);
    let mut best_idx = rng.random_range(0..n);
    let mut best_fitness = fitnesses[best_idx];

    for _ in 1..k {
        let idx = rng.random_range(0..n);
        if fitnesses[idx] > best_fitness {
            best_idx = idx;
            best_fitness = fitnesses[idx];
        }
    }
    best_idx
}

/// Roulette wheel (fitness-proportionate) selection.
/// Falls back to uniform if all fitnesses are equal or negative.
pub fn roulette_select(fitnesses: &[f64], rng: &mut impl Rng) -> usize {
    let n = fitnesses.len();
    if n == 0 {
        return 0;
    }
    // Shift fitnesses to be non-negative
    let min_fit = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
    let shifted: Vec<f64> = fitnesses.iter().map(|f| f - min_fit + 1e-6).collect();
    let total: f64 = shifted.iter().sum();
    if total <= 0.0 {
        return rng.random_range(0..n);
    }
    let r = rng.random::<f64>() * total;
    let mut cumulative = 0.0;
    for (i, f) in shifted.iter().enumerate() {
        cumulative += f;
        if r <= cumulative {
            return i;
        }
    }
    n - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_tournament_returns_valid_index() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut genomes = Vec::new();
        for _ in 0..10 {
            genomes.push(Genome::random(&mut rng));
        }
        let fitnesses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let idx = tournament_select(&genomes, &fitnesses, 3, &mut rng);
        assert!(idx < 10);
    }

    #[test]
    fn test_tournament_prefers_higher_fitness() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut genomes = Vec::new();
        for _ in 0..20 {
            genomes.push(Genome::random(&mut rng));
        }
        // One individual is much better than the rest
        let fitnesses: Vec<f64> = (0..20).map(|i| if i == 5 { 100.0 } else { 1.0 }).collect();
        let mut selections = std::collections::HashMap::new();
        for _ in 0..1000 {
            let idx = tournament_select(&genomes, &fitnesses, 5, &mut rng);
            *selections.entry(idx).or_insert(0u32) += 1;
        }
        // With k=5 out of 20, P(best is in tournament) = 5/20 = 25%.
        // Expected ~250 selections of the best. Check it's significantly above uniform (50).
        let best_count = selections.get(&5).copied().unwrap_or(0);
        assert!(
            best_count > 150,
            "Tournament with k=5 should pick the best >15% of the time, got {}",
            best_count
        );
    }

    #[test]
    fn test_roulette_returns_valid_index() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let fitnesses = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        for _ in 0..100 {
            let idx = roulette_select(&fitnesses, &mut rng);
            assert!(idx < 5);
        }
    }
}
