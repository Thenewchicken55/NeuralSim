pub mod types;
pub mod plasticity;

use plasticity::StdpTrace;
use serde::{Deserialize, Serialize};

pub type SynapseId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynapseType {
    AMPA,
    GABA,
    GabaB,
    NMDA,
}

/// Conductance-based synapse state.
/// Tracks the real conductance value g(t) and whether Mg²⁺ block applies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductanceState {
    /// Current conductance (µS)
    pub g: f64,
    /// Time since last presynaptic spike (ms)
    pub t_since_spike: f64,
    /// Whether a spike is being processed
    pub pending_spike: bool,
}

impl Default for ConductanceState {
    fn default() -> Self {
        Self::new()
    }
}

impl ConductanceState {
    pub fn new() -> Self {
        Self { g: 0.0, t_since_spike: 0.0, pending_spike: false }
    }
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
    /// Per-synapse conductance state for alpha-function dynamics
    pub conductance_state: Option<ConductanceState>,
    /// STDP traces for online learning
    pub stdp_trace: Option<StdpTrace>,
    /// Eligibility trace for R-STDP
    pub eligibility: Option<plasticity::EligibilityTrace>,
    /// Short-term plasticity state
    pub stp: Option<plasticity::ShortTermPlasticity>,
    /// Whether plasticity is enabled on this synapse
    pub plasticity_enabled: bool,
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
            conductance_state: None,
            stdp_trace: None,
            eligibility: None,
            stp: None,
            plasticity_enabled: false,
        }
    }

    pub fn with_plasticity(mut self) -> Self {
        self.plasticity_enabled = true;
        self.stdp_trace = Some(StdpTrace::new());
        self.conductance_state = Some(ConductanceState::new());
        self
    }

    pub fn with_stp(mut self, u_se: f64, tau_rec: f64, tau_facil: f64) -> Self {
        self.stp = Some(plasticity::ShortTermPlasticity::new(u_se, tau_rec, tau_facil));
        self
    }

    pub fn with_rstdp(mut self) -> Self {
        self.plasticity_enabled = true;
        self.stdp_trace = Some(StdpTrace::new());
        self.eligibility = Some(plasticity::EligibilityTrace::new());
        self
    }
}

/// Global plasticity configuration for a simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityConfig {
    pub stdp: Option<plasticity::StdpRule>,
    pub triplet_stdp: Option<plasticity::TripletStdpRule>,
    pub bcm: Option<plasticity::BcmRule>,
    pub consolidation: Option<plasticity::ConsolidationRule>,
    pub intrinsic: Option<plasticity::IntrinsicPlasticity>,
    pub rstdp: Option<plasticity::RStdpRule>,
    pub enabled: bool,
    pub homeostatic_target_rate: f64,
    pub homeostatic_tau: f64,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            stdp: Some(plasticity::StdpRule::default()),
            triplet_stdp: None,
            bcm: None,
            consolidation: None,
            intrinsic: None,
            rstdp: None,
            enabled: true,
            homeostatic_target_rate: 5.0,
            homeostatic_tau: 5000.0,
        }
    }
}
