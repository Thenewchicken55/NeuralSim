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
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HodgkinHuxleyNeuron {
    pub m: f64,
    pub h: f64,
    pub n: f64,
}

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

        self.m += (am * (1.0 - self.m) - bm * self.m) * dt;
        self.h += (ah * (1.0 - self.h) - bh * self.h) * dt;
        self.n += (an * (1.0 - self.n) - bn * self.n) * dt;

        // Ionic currents
        let i_na = g_na * self.m.powi(3) * self.h * (e_na - v);
        let i_k = g_k * self.n.powi(4) * (e_k - v);
        let i_l = g_l * (e_l - v);

        let dv = (input_current - i_na - i_k - i_l) / c_m * dt;
        state.membrane_potential += dv;

        if state.membrane_potential >= -55.0 {
            state.membrane_potential = -70.0;
            state.last_spike_time = 0.0;
            state.spike_count += 1;
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
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hh_resting_potential() {
        let mut neuron = HodgkinHuxleyNeuron { m: 0.05, h: 0.6, n: 0.32 };
        let mut state = neuron.reset_state(NeuronType::Excitatory);
        let spiked = neuron.step(&mut state, 0.01, 0.0);
        assert!(!spiked);
        assert!((state.membrane_potential - -65.0).abs() < 1.0);
    }

    #[test]
    fn test_hh_fires_with_current() {
        let mut neuron = HodgkinHuxleyNeuron { m: 0.05, h: 0.6, n: 0.32 };
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
}
