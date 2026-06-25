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

impl TripletStdpRule {
    pub fn weight_change(&self, dt_pre_post: f64, dt_post_pre: f64, _current_weight: f64) -> f64 {
        let ltp = self.a_plus * f64::exp(-dt_pre_post.abs() / self.tau_plus)
            + self.a_triplet * f64::exp(-dt_post_pre.abs() / self.tau_y);
        let ltd = self.a_minus * f64::exp(-dt_pre_post.abs() / self.tau_minus);
        if dt_pre_post > 0.0 { ltp } else { -ltd }
    }
}

/// Short-term plasticity (Tsodyks-Markram model).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShortTermPlasticity {
    pub u_se: f64,
    pub tau_rec: f64,
    pub tau_facil: f64,
    pub u: f64,
    pub r: f64,
}

impl ShortTermPlasticity {
    pub fn new(u_se: f64, tau_rec: f64, tau_facil: f64) -> Self {
        Self { u_se, tau_rec, tau_facil, u: u_se, r: 1.0 }
    }

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

/// Per-synapse trace state for STDP.
/// Tracks pre/post spike times for online learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdpTrace {
    pub pre_trace: f64,
    pub post_trace: f64,
    pub last_pre_spike: f64,
    pub last_post_spike: f64,
}

impl Default for StdpTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl StdpTrace {
    pub fn new() -> Self {
        Self {
            pre_trace: 0.0,
            post_trace: 0.0,
            last_pre_spike: -1e9,
            last_post_spike: -1e9,
        }
    }

    pub fn on_pre_spike(&mut self, time: f64, tau_plus: f64) {
        let dt = time - self.last_pre_spike;
        self.pre_trace = self.pre_trace * f64::exp(-dt / tau_plus) + 1.0;
        self.last_pre_spike = time;
    }

    pub fn on_post_spike(&mut self, time: f64, tau_minus: f64) {
        let dt = time - self.last_post_spike;
        self.post_trace = self.post_trace * f64::exp(-dt / tau_minus) + 1.0;
        self.last_post_spike = time;
    }

    pub fn decay(&mut self, dt: f64, tau_plus: f64, tau_minus: f64) {
        self.pre_trace *= f64::exp(-dt / tau_plus);
        self.post_trace *= f64::exp(-dt / tau_minus);
    }
}

/// Reward-modulated STDP (R-STDP).
/// Dopamine-like signal gates plasticity: Δw = η * D * e(t)
/// where D is the dopamine signal and e(t) is the eligibility trace.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RStdpRule {
    pub tau_plus: f64,
    pub tau_minus: f64,
    pub a_plus: f64,
    pub a_minus: f64,
    pub tau_eligibility: f64,
    pub learning_rate: f64,
    pub weight_min: f64,
    pub weight_max: f64,
}

impl Default for RStdpRule {
    fn default() -> Self {
        Self {
            tau_plus: 20.0,
            tau_minus: 20.0,
            a_plus: 0.01,
            a_minus: 0.012,
            tau_eligibility: 200.0,
            learning_rate: 0.1,
            weight_min: 0.0,
            weight_max: 10.0,
        }
    }
}

/// Eligibility trace for R-STDP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityTrace {
    pub value: f64,
}

impl Default for EligibilityTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl EligibilityTrace {
    pub fn new() -> Self {
        Self { value: 0.0 }
    }

    pub fn update(&mut self, dt: f64, tau_e: f64, stdp_delta: f64) {
        self.value = self.value * f64::exp(-dt / tau_e) + stdp_delta;
    }

    pub fn decay(&mut self, dt: f64, tau_e: f64) {
        self.value *= f64::exp(-dt / tau_e);
    }

    pub fn apply_reward(&mut self, reward: f64, weight: &mut f64, lr: f64, w_min: f64, w_max: f64) {
        *weight = (*weight + lr * reward * self.value).clamp(w_min, w_max);
        self.value = 0.0;
    }
}

/// Bienenstock-Cooper-Munro (BCM) rule.
/// Sliding threshold for LTD/LTP based on postsynaptic activity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BcmRule {
    pub tau_ltd: f64,
    pub tau_ltp: f64,
    pub tau_theta: f64,
    pub eta_ltd: f64,
    pub eta_ltp: f64,
    pub theta_init: f64,
    pub weight_min: f64,
    pub weight_max: f64,
}

impl Default for BcmRule {
    fn default() -> Self {
        Self {
            tau_ltd: 20.0,
            tau_ltp: 10.0,
            tau_theta: 100.0,
            eta_ltd: 0.01,
            eta_ltp: 0.01,
            theta_init: 0.5,
            weight_min: 0.0,
            weight_max: 10.0,
        }
    }
}

impl BcmRule {
    pub fn apply(&self, weight: &mut f64, pre_rate: f64, post_rate: f64, theta_m: f64, dt: f64) {
        if post_rate > theta_m {
            *weight += self.eta_ltp * pre_rate * (post_rate - theta_m) * dt;
        } else {
            *weight -= self.eta_ltd * pre_rate * (theta_m - post_rate) * dt;
        }
        *weight = weight.clamp(self.weight_min, self.weight_max);
    }

    pub fn update_threshold(&self, theta: &mut f64, post_rate: f64, dt: f64) {
        *theta += (post_rate - *theta) * dt / self.tau_theta;
    }
}

/// Synaptic consolidation — slow drift toward stable weight value.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConsolidationRule {
    pub tau_consolidation: f64,
    pub target_weight: f64,
    pub drift_rate: f64,
}

impl Default for ConsolidationRule {
    fn default() -> Self {
        Self {
            tau_consolidation: 10000.0,
            target_weight: 0.5,
            drift_rate: 0.001,
        }
    }
}

impl ConsolidationRule {
    pub fn apply(&self, weight: &mut f64, dt: f64) {
        let drift = (self.target_weight - *weight) * dt / self.tau_consolidation;
        *weight += drift * self.drift_rate;
    }
}

/// Intrinsic plasticity — neuron adjusts firing threshold toward target rate.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IntrinsicPlasticity {
    pub target_rate: f64,
    pub tau_intrinsic: f64,
    pub min_threshold: f64,
    pub max_threshold: f64,
}

impl Default for IntrinsicPlasticity {
    fn default() -> Self {
        Self {
            target_rate: 5.0,
            tau_intrinsic: 1000.0,
            min_threshold: -55.0,
            max_threshold: -40.0,
        }
    }
}

impl IntrinsicPlasticity {
    pub fn update_threshold(&self, threshold: &mut f64, firing_rate: f64, dt: f64) {
        let error = firing_rate - self.target_rate;
        *threshold += error * dt / self.tau_intrinsic;
        *threshold = threshold.clamp(self.min_threshold, self.max_threshold);
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
        let mut stp = ShortTermPlasticity::new(0.5, 100.0, 1000.0);
        let epsp = stp.step(1.0, true);
        assert!(epsp > 0.0);
        let epsp2 = stp.step(1.0, true);
        assert!(epsp2 <= epsp, "depression should reduce EPSP");
    }

    #[test]
    fn test_stdp_trace_decay() {
        let mut trace = StdpTrace::new();
        trace.on_pre_spike(0.0, 20.0);
        assert!(trace.pre_trace > 0.9);
        trace.decay(20.0, 20.0, 20.0);
        assert!((trace.pre_trace - 0.3679).abs() < 0.01);
    }

    #[test]
    fn test_eligibility_trace() {
        let mut et = EligibilityTrace::new();
        et.update(1.0, 200.0, 0.05);
        assert!(et.value > 0.0);
        let mut w = 0.5;
        et.apply_reward(1.0, &mut w, 0.1, 0.0, 10.0);
        assert!((w - 0.505).abs() < 0.001);
        assert!((et.value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bcm_threshold_update() {
        let rule = BcmRule::default();
        let mut theta = rule.theta_init;
        rule.update_threshold(&mut theta, 1.0, 100.0);
        assert!(theta > rule.theta_init);
    }

    #[test]
    fn test_consolidation_drift() {
        let rule = ConsolidationRule::default();
        let mut w = 5.0;
        rule.apply(&mut w, 1000.0);
        assert!((w - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_intrinsic_plasticity() {
        let ip = IntrinsicPlasticity::default();
        let mut th = -50.0;
        ip.update_threshold(&mut th, 50.0, 1000.0);
        assert!(th > -50.0, "high rate should raise threshold");
    }
}
