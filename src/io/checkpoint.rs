use crate::error::{NeuralSimError, Result};
use crate::network::Network;
use crate::simulation::{SimulationEngine, SimulationStats};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

/// Checkpoint manager with metadata for resuming simulations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub name: String,
    pub sim_time_ms: f64,
    pub neuron_count: usize,
    pub synapse_count: usize,
    pub total_spikes: u64,
    pub timestamp: String,
    /// RNG seed used for this simulation run
    pub seed: u64,
}

#[derive(Clone)]
pub struct CheckpointManager {
    /// Directory to store checkpoint files
    pub checkpoint_dir: PathBuf,
    /// Interval in simulation ms between auto-save
    pub auto_save_interval_ms: f64,
    /// Last auto-save time
    last_save_time: f64,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
}

impl CheckpointManager {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        fs::create_dir_all(&dir).ok();
        Self {
            checkpoint_dir: dir,
            auto_save_interval_ms: 1000.0,
            last_save_time: 0.0,
            max_checkpoints: 10,
        }
    }

    /// Save a checkpoint with the network state and simulation stats.
    pub fn save(&self, network: &Network, stats: &SimulationStats, name: &str) -> Result<PathBuf> {
        let path = self.checkpoint_dir.join(format!("{}.json", name));
        let meta = CheckpointMetadata {
            name: name.into(),
            sim_time_ms: stats.sim_time_ms,
            neuron_count: network.neuron_count(),
            synapse_count: network.synapse_count(),
            total_spikes: stats.total_spikes,
            timestamp: chrono_now(),
            seed: stats.seed,
        };

        // Save metadata alongside
        let meta_path = self.checkpoint_dir.join(format!("{}_meta.json", name));
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(&meta_path, meta_json).map_err(|e| {
            NeuralSimError::io(
                "failed to write checkpoint metadata",
                Some(meta_path.clone()),
                e,
            )
        })?;

        // Save network
        let json = serde_json::to_string_pretty(network)?;
        fs::write(&path, json)
            .map_err(|e| NeuralSimError::io("failed to write checkpoint", Some(path.clone()), e))?;

        self.prune()?;
        Ok(path)
    }

    /// Save state from a running engine.
    pub fn save_engine(&self, engine: &SimulationEngine, name: &str) -> Result<PathBuf> {
        let net = engine.network.read();
        let stats = engine.stats.read();
        self.save(&net, &stats, name)
    }

    /// Restore from a checkpoint file.
    pub fn load(&self, name: &str) -> Result<(Network, CheckpointMetadata)> {
        let path = self.checkpoint_dir.join(format!("{}.json", name));
        let meta_path = self.checkpoint_dir.join(format!("{}_meta.json", name));

        let json = fs::read_to_string(&path)
            .map_err(|e| NeuralSimError::io("failed to read checkpoint", Some(path), e))?;
        let network: Network = serde_json::from_str(&json)?;
        let meta_json = fs::read_to_string(&meta_path).map_err(|e| {
            NeuralSimError::io("failed to read checkpoint metadata", Some(meta_path), e)
        })?;
        let meta: CheckpointMetadata = serde_json::from_str(&meta_json)?;

        Ok((network, meta))
    }

    /// Check if auto-save is needed.
    pub fn should_auto_save(&self, current_time: f64) -> bool {
        current_time - self.last_save_time >= self.auto_save_interval_ms
    }

    /// Mark auto-save as done.
    pub fn mark_saved(&mut self, current_time: f64) {
        self.last_save_time = current_time;
    }

    /// List all available checkpoints.
    pub fn list_checkpoints(&self) -> Vec<CheckpointMetadata> {
        let mut checkpoints = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json")
                    && !path.to_string_lossy().contains("_meta.")
                {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    let meta_path = self.checkpoint_dir.join(format!("{}_meta.json", name));
                    if let Ok(meta_json) = fs::read_to_string(&meta_path)
                        && let Ok(meta) = serde_json::from_str::<CheckpointMetadata>(&meta_json)
                    {
                        checkpoints.push(meta);
                    }
                }
            }
        }
        checkpoints.sort_by(|a, b| {
            b.sim_time_ms
                .partial_cmp(&a.sim_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        checkpoints
    }

    fn prune(&self) -> Result<()> {
        let mut checkpoints = self.list_checkpoints();
        if checkpoints.len() <= self.max_checkpoints {
            return Ok(());
        }
        checkpoints.sort_by(|a, b| {
            a.sim_time_ms
                .partial_cmp(&b.sim_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let to_remove = checkpoints.len() - self.max_checkpoints;
        for cp in checkpoints.iter().take(to_remove) {
            let path = self.checkpoint_dir.join(format!("{}.json", cp.name));
            let meta_path = self.checkpoint_dir.join(format!("{}_meta.json", cp.name));
            fs::remove_file(path).ok();
            fs::remove_file(meta_path).ok();
        }
        Ok(())
    }
}

fn chrono_now() -> String {
    // Simple timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:?}", dur)
}

/// Streaming stats recorder for exporting firing rates etc. to CSV.
/// Uses `VecDeque` for O(1) ring-buffer eviction.
pub struct StatsRecorder {
    records: VecDeque<StatsRecord>,
    max_records: usize,
    record_interval_steps: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsRecord {
    pub step: u64,
    pub sim_time_ms: f64,
    pub total_spikes: u64,
    pub output_spikes: u64,
    pub mean_firing_rate: f64,
    pub synchrony_index: f64,
    pub weight_mean: f64,
    pub weight_std: f64,
    pub lfp: f64,
}

impl StatsRecorder {
    pub fn new(max_records: usize, interval_steps: u64) -> Self {
        Self {
            records: VecDeque::with_capacity(max_records),
            max_records,
            record_interval_steps: interval_steps,
        }
    }

    pub fn record(&mut self, step: u64, stats: &SimulationStats, lfp: f64) {
        if !step.is_multiple_of(self.record_interval_steps) {
            return;
        }
        if self.records.len() >= self.max_records {
            self.records.pop_front();
        }
        self.records.push_back(StatsRecord {
            step,
            sim_time_ms: stats.sim_time_ms,
            total_spikes: stats.total_spikes,
            output_spikes: stats.output_spikes,
            mean_firing_rate: stats.mean_firing_rate,
            synchrony_index: stats.synchrony_index,
            weight_mean: stats.weight_mean,
            weight_std: stats.weight_std,
            lfp,
        });
    }

    pub fn to_csv(&self) -> String {
        let mut csv = String::from(
            "step,sim_time_ms,total_spikes,output_spikes,mean_firing_rate,synchrony_index,weight_mean,weight_std,lfp\n",
        );
        for r in &self.records {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{}\n",
                r.step,
                r.sim_time_ms,
                r.total_spikes,
                r.output_spikes,
                r.mean_firing_rate,
                r.synchrony_index,
                r.weight_mean,
                r.weight_std,
                r.lfp
            ));
        }
        csv
    }

    pub fn save_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref().to_path_buf();
        fs::write(&path, self.to_csv())
            .map_err(|e| NeuralSimError::io("failed to write stats CSV", Some(path), e))?;
        Ok(())
    }

    pub fn records(&self) -> &VecDeque<StatsRecord> {
        &self.records
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Network;
    use crate::simulation::SimulationStats;

    #[test]
    fn test_checkpoint_metadata() {
        let meta = CheckpointMetadata {
            name: "test".into(),
            sim_time_ms: 100.0,
            neuron_count: 1000,
            synapse_count: 50000,
            total_spikes: 42,
            timestamp: "test".into(),
            seed: 42,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let restored: CheckpointMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.name, "test");
        assert_eq!(restored.neuron_count, 1000);
        assert_eq!(restored.seed, 42);
    }

    #[test]
    fn test_stats_recorder() {
        let mut recorder = StatsRecorder::new(100, 1);
        let stats = SimulationStats {
            sim_time_ms: 10.0,
            total_spikes: 5,
            mean_firing_rate: 2.5,
            ..Default::default()
        };
        recorder.record(1, &stats, 0.1);
        assert_eq!(recorder.records().len(), 1);
        assert!(!recorder.to_csv().is_empty());
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = std::env::temp_dir().join("neural_sim_test_checkpoints");
        let _ = fs::remove_dir_all(&dir);
        let mgr = CheckpointManager::new(&dir);

        let net = Network::new(100);
        let stats = SimulationStats::default();
        mgr.save(&net, &stats, "test_cp").unwrap();

        let (loaded, meta) = mgr.load("test_cp").unwrap();
        assert_eq!(loaded.neuron_count(), 100);
        assert_eq!(meta.neuron_count, 100);

        let _ = fs::remove_dir_all(&dir);
    }
}
