use super::{NeuronModel, NeuronModelParams, NeuronState, NeuronType};
use serde::{Deserialize, Serialize};

/// Hodgkin-Huxley conductance-based neuron model.
///
/// C_m * dV/dt = g_Na * m^3 * h * (E_Na - V) + g_K * n^4 * (E_K - V) + g_L * (E_L - V) + I
/// dm/dt = α_m(V) * (1-m) - β_m(V) * m
/// dh/dt = α_h(V) * (1-h) - β_h(V) * h
/// dn/dt = α_n(V) * (1-n) - β_n(V) * n
///
/// This is the most biologically detailed model, capturing
/// sodium and potassium channel dynamics.
///
/// Gating variables (m, h, n) are stored in NeuronState so they
/// persist across steps in the SoA layout.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HodgkinHuxleyNeuron;

impl HodgkinHuxleyNeuron {
    fn alpha_m(v: f64) -> f64 {
        0.1 * (v + 40.0) / (1.0 - f64::exp(-(v + 40.0) / 10.0))
    }

    fn beta_m(v: f64) -> f64 {
        4.0 * f64::exp(-(v + 65.0) / 18.0)
    }

    fn alpha_h(v: f64) -> f64 {
        0.07 * f64::exp(-(v + 65.0) / 20.0)
    }

    fn beta_h(v: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-(v + 35.0) / 10.0))
    }

    fn alpha_n(v: f64) -> f64 {
        0.01 * (v + 55.0) / (1.0 - f64::exp(-(v + 55.0) / 10.0))
    }

    fn beta_n(v: f64) -> f64 {
        0.125 * f64::exp(-(v + 65.0) / 80.0)
    }
}

impl NeuronModel for HodgkinHuxleyNeuron {
    fn step(&mut self, state: &mut NeuronState, dt: f64, input_current: f64) -> bool {
        let params = match &state.model_params {
            NeuronModelParams::HodgkinHuxley { .. } => &state.model_params,
            _ => return false,
        };

        let (g_na, g_k, g_l, e_na, e_k, e_l, c_m) = match params {
            NeuronModelParams::HodgkinHuxley {
                g_na,
                g_k,
                g_l,
                e_na,
                e_k,
                e_l,
                c_m,
            } => (*g_na, *g_k, *g_l, *e_na, *e_k, *e_l, *c_m),
            _ => unreachable!(),
        };

        let v = state.membrane_potential;

        // Gating variable updates (forward Euler)
        let am = Self::alpha_m(v);
        let bm = Self::beta_m(v);
        let ah = Self::alpha_h(v);
        let bh = Self::beta_h(v);
        let an = Self::alpha_n(v);
        let bn = Self::beta_n(v);

        state.hh_m += (am * (1.0 - state.hh_m) - bm * state.hh_m) * dt;
        state.hh_h += (ah * (1.0 - state.hh_h) - bh * state.hh_h) * dt;
        state.hh_n += (an * (1.0 - state.hh_n) - bn * state.hh_n) * dt;

        // Ionic currents
        let i_na = g_na * state.hh_m.powi(3) * state.hh_h * (e_na - v);
        let i_k = g_k * state.hh_n.powi(4) * (e_k - v);
        let i_l = g_l * (e_l - v);

        let dv = (input_current - i_na - i_k - i_l) / c_m * dt;
        state.membrane_potential += dv;

        if state.membrane_potential >= -55.0 {
            state.membrane_potential = -70.0;
            state.last_spike_time = 0.0;
            state.spike_count += 1;
            state.just_spiked = true;
            return true;
        }

        false
    }

    fn reset_state(&self, neuron_type: NeuronType) -> NeuronState {
        let params = NeuronModelParams::HodgkinHuxley {
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.387,
            c_m: 1.0,
        };
        let mut state = NeuronState::new(neuron_type, params);
        state.membrane_potential = -65.0;
        state.hh_m = 0.05;
        state.hh_h = 0.6;
        state.hh_n = 0.32;
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hh_resting_potential() {
        let mut neuron = HodgkinHuxleyNeuron;
        let mut state = neuron.reset_state(NeuronType::Excitatory);
        let spiked = neuron.step(&mut state, 0.01, 0.0);
        assert!(!spiked);
        assert!((state.membrane_potential - -65.0).abs() < 1.0);
    }

    #[test]
    fn test_hh_fires_with_current() {
        let mut neuron = HodgkinHuxleyNeuron;
        let mut state = neuron.reset_state(NeuronType::Excitatory);

        let mut fired = false;
        for _ in 0..500 {
            if neuron.step(&mut state, 0.01, 10.0) {
                fired = true;
                break;
            }
        }
        assert!(fired, "HH neuron should fire with sustained input current");
    }

    #[test]
    fn test_hh_gating_state_persists() {
        let mut neuron = HodgkinHuxleyNeuron;
        let mut state = neuron.reset_state(NeuronType::Excitatory);
        // After one step without input, gating should change slightly
        let (m0, h0, n0) = (state.hh_m, state.hh_h, state.hh_n);
        let _ = neuron.step(&mut state, 0.1, 0.0);
        assert!(
            state.hh_m != m0 || state.hh_h != h0 || state.hh_n != n0,
            "HH gating variables should change after a step"
        );
        // After another step, variables should continue evolving (not reset)
        let (m1, h1, n1) = (state.hh_m, state.hh_h, state.hh_n);
        let _ = neuron.step(&mut state, 0.1, 0.0);
        assert!(
            state.hh_m != m1 || state.hh_h != h1 || state.hh_n != n1,
            "HH gating variables should continue evolving across steps"
        );
    }
}
