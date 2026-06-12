use serde::{Deserialize, Serialize};

/// Spike-Timing-Dependent Plasticity (STDP) rule.
///
/// Δw = A_plus * exp(-Δt / τ_plus)   for pre before post (Δt > 0, LTP)
/// Δw = -A_minus * exp(Δt / τ_minus)  for post before pre (Δt < 0, LTD)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StdpRule {
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub a_plus: f64,
    pub a_minus: f64,
    pub weight_min: f64,
    pub weight_max: f64,
}

impl Default for StdpRule {
    fn default() -> Self {
        Self {
            tau_plus: 20.0,
            tau_minus: 20.0,
            a_plus: 0.01,
            a_minus: 0.012,
            weight_min: 0.0,
            weight_max: 10.0,
        }
    }
}

impl StdpRule {
    pub fn weight_change(&self, dt: f64, _current_weight: f64) -> f64 {
        if dt > 0.0 {
            self.a_plus * f64::exp(-dt / self.tau_plus)
        } else {
            -self.a_minus * f64::exp(dt / self.tau_minus)
        }
    }

    pub fn apply(&self, weight: &mut f64, dt: f64) {
        let delta = self.weight_change(dt, *weight);
        *weight = (*weight + delta).clamp(self.weight_min, self.weight_max);
    }
}

/// Triplet STDP rule (more biologically accurate).
/// Incorporates interactions between three spikes (pre-post-pre or post-pre-post).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TripletStdpRule {
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub tau_y: f64,
    pub a_plus: f64,
    pub a_minus: f64,
    pub a_triplet: f64,
    pub weight_min: f64,
    pub weight_max: f64,
}

impl Default for TripletStdpRule {
    fn default() -> Self {
        Self {
            tau_plus: 16.8,
            tau_minus: 33.7,
            tau_y: 100.0,
            a_plus: 0.0005,
            a_minus: 0.0002,
            a_triplet: 0.0003,
            weight_min: 0.0,
            weight_max: 10.0,
        }
    }
}

/// Short-term plasticity (Tsodyks-Markram model).
/// Models depletion and recovery of synaptic resources.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShortTermPlasticity {
    pub u_se: f64,    // utilization of synaptic efficacy
    pub tau_rec: f64, // recovery time constant (ms)
    pub tau_facil: f64, // facilitation time constant (ms)
    pub u: f64,       // current utilization
    pub r: f64,       // available resources (fraction)
}

impl ShortTermPlasticity {
    pub fn step(&mut self, dt: f64, spike: bool) -> f64 {
        if spike {
            self.u += self.u_se * (1.0 - self.u) / self.tau_facil * dt;
            let epsp = self.u * self.r;
            self.r -= epsp;
            epsp
        } else {
            self.u += (self.u_se - self.u) / self.tau_facil * dt;
            self.r += (1.0 - self.r) / self.tau_rec * dt;
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_ltp_positive_dt() {
        let rule = StdpRule::default();
        let delta = rule.weight_change(10.0, 0.5);
        assert!(delta > 0.0, "pre-before-post should cause LTP");
    }

    #[test]
    fn test_stdp_ltd_negative_dt() {
        let rule = StdpRule::default();
        let delta = rule.weight_change(-10.0, 0.5);
        assert!(delta < 0.0, "post-before-pre should cause LTD");
    }

    #[test]
    fn test_stdp_clamps_weight() {
        let rule = StdpRule::default();
        let mut w = 0.0;
        rule.apply(&mut w, 10.0);
        assert!(w >= rule.weight_min);
    }

    #[test]
    fn test_short_term_plasticity() {
        let mut stp = ShortTermPlasticity {
            u_se: 0.5,
            tau_rec: 100.0,
            tau_facil: 1000.0,
            u: 0.5,
            r: 1.0,
        };
        let epsp = stp.step(1.0, true);
        assert!(epsp > 0.0);
        // After first spike, resources are depleted so second EPSP is smaller
        let epsp2 = stp.step(1.0, true);
        assert!(epsp2 <= epsp, "depression should reduce EPSP");
    }
}
