use serde::{Deserialize, Serialize};

/// Time constants for each synapse type (in ms)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SynapseDynamics {
    pub rise_time: f64,
    pub decay_time: f64,
    pub reversal_potential: f64,
    pub conductance_max: f64,
    /// Mg²⁺ block strength for NMDA (0.0 = no block)
    pub mg_block_strength: f64,
}

impl SynapseDynamics {
    pub fn ampa() -> Self {
        Self {
            rise_time: 0.5,
            decay_time: 2.0,
            reversal_potential: 0.0,
            conductance_max: 1.0,
            mg_block_strength: 0.0,
        }
    }

    pub fn gaba() -> Self {
        Self {
            rise_time: 1.0,
            decay_time: 10.0,
            reversal_potential: -70.0,
            conductance_max: 1.0,
            mg_block_strength: 0.0,
        }
    }

    /// GABA-B — slow metabotropic inhibition via G-protein coupled receptors.
    pub fn gaba_b() -> Self {
        Self {
            rise_time: 30.0,
            decay_time: 150.0,
            reversal_potential: -90.0,
            conductance_max: 0.5,
            mg_block_strength: 0.0,
        }
    }

    pub fn nmda() -> Self {
        Self {
            rise_time: 2.0,
            decay_time: 100.0,
            reversal_potential: 0.0,
            conductance_max: 0.5,
            mg_block_strength: 1.0,
        }
    }

    /// Compute normalized dual-exponential conductance at time t after a spike.
    /// g(t) = g_max * (exp(-t/tau_d) - exp(-t/tau_r)) / norm
    /// where norm is the peak value so that max(g(t)) = g_max.
    pub fn conductance(&self, t: f64) -> f64 {
        if t < 0.0 {
            return 0.0;
        }
        let tau_r = self.rise_time;
        let tau_d = self.decay_time;
        if tau_d <= 0.0 || tau_r <= 0.0 {
            return 0.0;
        }
        if (tau_d - tau_r).abs() < 1e-12 {
            // Alpha function for equal time constants
            let norm = tau_r * (-1.0_f64).exp();
            self.conductance_max * t * (-t / tau_r).exp() / norm
        } else {
            let t_peak = (tau_r * tau_d / (tau_d - tau_r)) * (tau_d / tau_r).ln();
            let e_d = (-t_peak / tau_d).exp();
            let e_r = (-t_peak / tau_r).exp();
            let norm = e_d - e_r;
            if norm <= 0.0 {
                return 0.0;
            }
            let g = ((-t / tau_d).exp() - (-t / tau_r).exp()) / norm;
            self.conductance_max * g.max(0.0)
        }
    }

    /// NMDA voltage-dependent Mg²⁺ block.
    /// Returns fraction of channels unblocked (0..1).
    /// g_mg(V) = 1 / (1 + [Mg²⁺]/3.57 * exp(-0.062 * V))
    pub fn nmda_mg_block(&self, membrane_potential: f64) -> f64 {
        if self.mg_block_strength <= 0.0 {
            return 1.0;
        }
        let mg_concentration = 1.0; // mM (physiological)
        1.0 / (1.0 + mg_concentration / 3.57 * (-0.062 * membrane_potential).exp())
    }

    /// Effective conductance for NMDA, accounting for voltage gating.
    pub fn effective_conductance(&self, t: f64, membrane_potential: f64) -> f64 {
        let g = self.conductance(t);
        let mg_factor = self.nmda_mg_block(membrane_potential);
        g * mg_factor
    }
}

/// Synaptic dynamics registry for looking up properties by type.
pub fn dynamics_for(syn_type: &super::SynapseType) -> SynapseDynamics {
    match syn_type {
        super::SynapseType::AMPA => SynapseDynamics::ampa(),
        super::SynapseType::GABA => SynapseDynamics::gaba(),
        super::SynapseType::GabaB => SynapseDynamics::gaba_b(),
        super::SynapseType::NMDA => SynapseDynamics::nmda(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ampa_conductance() {
        let dyns = SynapseDynamics::ampa();
        // At t=0, conductance should be 0
        let g0 = dyns.conductance(0.0);
        assert!(g0.abs() < 1e-10, "AMPA conductance at t=0 should be 0");
        // At t=1ms, should be positive
        let g1 = dyns.conductance(1.0);
        assert!(g1 > 0.0, "AMPA conductance should be positive at t=1ms");
        // At t=100ms, should be almost 0
        let g_far = dyns.conductance(100.0);
        assert!(g_far < g1, "AMPA conductance should decay");
    }

    #[test]
    fn test_nmda_mg_block_depolarized() {
        let dyns = SynapseDynamics::nmda();
        // At depolarized potential (+20mV), channels should be mostly unblocked
        let unblocked = dyns.nmda_mg_block(20.0);
        assert!(unblocked > 0.7, "NMDA should be mostly unblocked at +20mV, got {:.3}", unblocked);
        // At hyperpolarized potential (-80mV), channels should be mostly blocked
        let blocked = dyns.nmda_mg_block(-80.0);
        assert!(blocked < 0.3, "NMDA should be mostly blocked at -80mV, got {:.3}", blocked);
    }

    #[test]
    fn test_nmda_partially_blocked_at_rest() {
        let dyns = SynapseDynamics::nmda();
        // At resting potential around -65mV, NMDA should be significantly blocked
        let blocked = dyns.nmda_mg_block(-65.0);
        assert!(blocked < 0.5, "NMDA should be significantly blocked at rest, got {:.3}", blocked);
        assert!(blocked > 0.01, "NMDA should not be fully blocked at rest, got {:.3}", blocked);
    }

    #[test]
    fn test_gaba_b_slow() {
        let dyns = SynapseDynamics::gaba_b();
        // GABA-B peaks later than AMPA — at t=30ms, g should be > at t=1ms
        let g_peak = dyns.conductance(30.0);
        let g_early = dyns.conductance(1.0);
        assert!(g_peak > g_early, "GABA-B at 30ms ({:.4}) should be > at 1ms ({:.4})", g_peak, g_early);
        // Should peak somewhere reasonable
        let g_mid = dyns.conductance(60.0);
        let g_late = dyns.conductance(300.0);
        assert!(g_mid > g_late, "GABA-B should decay after peak: 60ms={:.4} > 300ms={:.4}", g_mid, g_late);
    }

    #[test]
    fn test_conductance_peak_normalization() {
        let dyns = SynapseDynamics::ampa();
        // The peak of the normalized function should be close to conductance_max
        let t_peak = (dyns.rise_time * dyns.decay_time / (dyns.decay_time - dyns.rise_time))
            * (dyns.decay_time / dyns.rise_time).ln();
        let g_peak = dyns.conductance(t_peak);
        assert!((g_peak - dyns.conductance_max).abs() < 0.01,
            "Peak conductance should be close to g_max, got {:.4} vs {:.4}", g_peak, dyns.conductance_max);
    }
}
