pub mod types;
pub mod plasticity;

use serde::{Deserialize, Serialize};

pub type SynapseId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynapseType {
    AMPA,
    GABA,
    NMDA,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseState {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
    pub delay: f64,
    pub synapse_type: SynapseType,
    pub conductance: f64,
    pub is_active: bool,
}

impl SynapseState {
    pub fn new(source: usize, target: usize, weight: f64, synapse_type: SynapseType) -> Self {
        Self {
            source,
            target,
            weight,
            delay: 1.0,
            synapse_type,
            conductance: 0.0,
            is_active: true,
        }
    }
}
