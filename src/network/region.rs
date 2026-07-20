use crate::network::connectivity::ConnectivityPattern;
use serde::{Deserialize, Serialize};

/// Configuration for a brain region template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionConfig {
    pub name: String,
    pub neuron_count: usize,
    pub excitatory_ratio: f64,
    pub connectivity: ConnectivityPattern,
    pub layers: Option<Vec<LayerConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub name: String,
    pub neuron_count: usize,
    pub excitatory_ratio: f64,
    pub connectivity: ConnectivityPattern,
}

/// Pre-built brain region templates based on known neuroanatomy
pub enum RegionTemplate {
    /// 6-layer cortical column with layer-specific connectivity
    CorticalColumn { total_neurons: usize },
    /// Thalamus <-> Cortex loop
    ThalamocorticalLoop {
        thalamus_size: usize,
        cortex_size: usize,
    },
    /// Hippocampal formation DG -> CA3 -> CA1
    HippocampalFormation {
        dg_size: usize,
        ca3_size: usize,
        ca1_size: usize,
    },
}

impl RegionTemplate {
    pub fn build(&self) -> Vec<RegionConfig> {
        match self {
            Self::CorticalColumn { total_neurons } => {
                // Layer distribution: L2/3 (15%), L4 (20%), L5 (25%), L6 (40%)
                let l23 = (*total_neurons as f64 * 0.15) as usize;
                let l4 = (*total_neurons as f64 * 0.20) as usize;
                let l5 = (*total_neurons as f64 * 0.25) as usize;
                let l6 = total_neurons - l23 - l4 - l5;

                vec![
                    RegionConfig {
                        name: "L2/3".into(),
                        neuron_count: l23,
                        excitatory_ratio: 0.85,
                        connectivity: ConnectivityPattern::ErdosRenyi { p: 0.05 },
                        layers: None,
                    },
                    RegionConfig {
                        name: "L4".into(),
                        neuron_count: l4,
                        excitatory_ratio: 0.80,
                        connectivity: ConnectivityPattern::ErdosRenyi { p: 0.08 },
                        layers: None,
                    },
                    RegionConfig {
                        name: "L5".into(),
                        neuron_count: l5,
                        excitatory_ratio: 0.75,
                        connectivity: ConnectivityPattern::ErdosRenyi { p: 0.06 },
                        layers: None,
                    },
                    RegionConfig {
                        name: "L6".into(),
                        neuron_count: l6,
                        excitatory_ratio: 0.70,
                        connectivity: ConnectivityPattern::ErdosRenyi { p: 0.04 },
                        layers: None,
                    },
                ]
            }
            Self::ThalamocorticalLoop {
                thalamus_size,
                cortex_size,
            } => {
                vec![
                    RegionConfig {
                        name: "Thalamus".into(),
                        neuron_count: *thalamus_size,
                        excitatory_ratio: 0.90,
                        connectivity: ConnectivityPattern::ErdosRenyi { p: 0.03 },
                        layers: None,
                    },
                    RegionConfig {
                        name: "Cortex".into(),
                        neuron_count: *cortex_size,
                        excitatory_ratio: 0.80,
                        connectivity: ConnectivityPattern::ErdosRenyi { p: 0.05 },
                        layers: None,
                    },
                ]
            }
            Self::HippocampalFormation {
                dg_size,
                ca3_size,
                ca1_size,
            } => {
                vec![
                    RegionConfig {
                        name: "DG".into(),
                        neuron_count: *dg_size,
                        excitatory_ratio: 0.90,
                        connectivity: ConnectivityPattern::ErdosRenyi { p: 0.01 },
                        layers: None,
                    },
                    RegionConfig {
                        name: "CA3".into(),
                        neuron_count: *ca3_size,
                        excitatory_ratio: 0.85,
                        connectivity: ConnectivityPattern::ErdosRenyi { p: 0.02 },
                        layers: None,
                    },
                    RegionConfig {
                        name: "CA1".into(),
                        neuron_count: *ca1_size,
                        excitatory_ratio: 0.80,
                        connectivity: ConnectivityPattern::ErdosRenyi { p: 0.03 },
                        layers: None,
                    },
                ]
            }
        }
    }
}
