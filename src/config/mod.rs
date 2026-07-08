use crate::neuron::NeuronModelParams;
use crate::synapse::PlasticityConfig;
use serde::Deserialize;

/// Top-level simulation configuration, parsed from YAML or JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct SimConfig {
    pub network: NetworkConfig,
    pub simulation: SimulationConfig,
    pub plasticity: Option<PlasticityConfig>,
    pub io: Option<IoConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NetworkConfig {
    pub seed: Option<u64>,
    pub regions: Vec<RegionConfig>,
    pub connections: Vec<RegionConnectionConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegionConfig {
    pub name: String,
    pub neuron_count: usize,
    pub excitatory_ratio: Option<f64>,
    pub model: Option<String>,
    pub model_params: Option<NeuronModelParams>,
    pub is_input: Option<bool>,
    pub is_output: Option<bool>,
    pub cortical_column: Option<CorticalColumnConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CorticalColumnConfig {
    pub enabled: bool,
    pub layer_fractions: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RegionConnectionConfig {
    pub from: String,
    pub to: String,
    pub probability: f64,
    pub weight_scale: Option<f64>,
    pub synapse_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SimulationConfig {
    pub dt_ms: Option<f64>,
    pub duration_ms: Option<f64>,
    pub noise_amplitude: Option<f64>,
    pub use_conductance: Option<bool>,
    pub reward_threshold: Option<u64>,
    pub reward_decay: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IoConfig {
    pub checkpoint_dir: Option<String>,
    pub auto_save_interval_ms: Option<f64>,
    pub stats_csv: Option<String>,
    pub text_io: Option<TextIoConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextIoConfig {
    pub input_neurons: usize,
    pub tokens_per_second: Option<f64>,
}

impl SimConfig {
    /// Load config from a YAML file path.
    pub fn from_yaml(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: SimConfig = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    /// Load config from a JSON file path.
    pub fn from_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: SimConfig = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Detect format by extension and load.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        if path.ends_with(".yaml") || path.ends_with(".yml") {
            Self::from_yaml(path)
        } else if path.ends_with(".json") {
            Self::from_json(path)
        } else {
            Err(format!("Unknown config format: {}. Use .yaml, .yml, or .json", path).into())
        }
    }

    /// Apply config to produce simulation parameters.
    pub fn seed(&self) -> u64 {
        self.network.seed.unwrap_or(42)
    }

    /// Get a builder-friendly list of regions from the config.
    pub fn region_specs(&self) -> Vec<(String, usize, f64, NeuronModelParams, bool, bool)> {
        self.network.regions.iter().map(|r| {
            let exc_ratio = r.excitatory_ratio.unwrap_or(0.8);
            let params = r.model_params.unwrap_or(NeuronModelParams::default());
            (r.name.clone(), r.neuron_count, exc_ratio, params, r.is_input.unwrap_or(false), r.is_output.unwrap_or(false))
        }).collect()
    }

    pub fn dt_ms(&self) -> f64 {
        self.simulation.dt_ms.unwrap_or(0.5)
    }

    pub fn duration_ms(&self) -> f64 {
        self.simulation.duration_ms.unwrap_or(100.0)
    }
}
