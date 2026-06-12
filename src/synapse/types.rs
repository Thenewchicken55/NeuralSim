use serde::{Deserialize, Serialize};

/// Time constants for each synapse type (in ms)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SynapseDynamics {
    pub rise_time: f64,
    pub decay_time: f64,
    pub reversal_potential: f64,
    pub conductance_max: f64,
}

impl SynapseDynamics {
    pub fn ampa() -> Self {
        Self {
            rise_time: 0.5,
            decay_time: 2.0,
            reversal_potential: 0.0,
            conductance_max: 1.0,
        }
    }

    pub fn gaba() -> Self {
        Self {
            rise_time: 1.0,
            decay_time: 10.0,
            reversal_potential: -70.0,
            conductance_max: 1.0,
        }
    }

    pub fn nmda() -> Self {
        Self {
            rise_time: 2.0,
            decay_time: 100.0,
            reversal_potential: 0.0,
            conductance_max: 0.5,
        }
    }

    /// Compute conductance at time t after a spike
    pub fn conductance(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        let alpha = 1.0 / self.rise_time;
        let beta = 1.0 / self.decay_time;
        let norm = (beta / alpha).powf(alpha / (beta - alpha)) - (beta / alpha).powf(beta / (beta - alpha));
        self.conductance_max * (f64::exp(-alpha * t) - f64::exp(-beta * t)) / norm
    }
}
