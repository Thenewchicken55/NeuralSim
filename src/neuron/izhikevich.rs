use super::{NeuronModel, NeuronModelParams, NeuronState, NeuronType};
use serde::{Deserialize, Serialize};

/// Izhikevich spiking neuron model.
///
/// dv/dt = 0.04*v^2 + 5*v + 140 - u + I
/// du/dt = a * (b*v - u)
///
/// After spike (v >= 30 mV):
///   v <- c
///   u <- u + d
///
/// This model can reproduce all known cortical firing patterns
/// (regular spiking, chattering, fast spiking, etc.) by tuning
/// parameters (a, b, c, d).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IzhikevichNeuron;

impl NeuronModel for IzhikevichNeuron {
    fn step(&mut self, state: &mut NeuronState, dt: f64, input_current: f64) -> bool {
        state.just_spiked = false;

        let params = match &state.model_params {
            NeuronModelParams::Izhikevich { .. } => &state.model_params,
            _ => return false,
        };

        let (a, b, c, d) = match params {
            NeuronModelParams::Izhikevich { a, b, c, d } => (*a, *b, *c, *d),
            _ => unreachable!(),
        };

        if state.membrane_potential >= 30.0 {
            state.membrane_potential = c;
            state.recovery_variable += d;
            state.last_spike_time = 0.0;
            state.spike_count += 1;
            state.just_spiked = true;
            return true;
        }

        let v = state.membrane_potential;
        let u = state.recovery_variable;

        let dv = (0.04 * v * v + 5.0 * v + 140.0 - u + input_current) * dt;
        let du = a * (b * v - u) * dt;

        state.membrane_potential += dv;
        state.recovery_variable += du;

        false
    }

    fn reset_state(&self, neuron_type: NeuronType) -> NeuronState {
        let params = match neuron_type {
            NeuronType::Excitatory => NeuronModelParams::Izhikevich {
                a: 0.02,
                b: 0.2,
                c: -65.0,
                d: 8.0,
            },
            NeuronType::Inhibitory => NeuronModelParams::Izhikevich {
                a: 0.02,
                b: 0.25,
                c: -65.0,
                d: 2.0,
            },
        };
        NeuronState::new(neuron_type, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_izhikevich_fires() {
        let mut state = IzhikevichNeuron.reset_state(NeuronType::Excitatory);
        state.membrane_potential = -65.0;
        let mut neuron = IzhikevichNeuron;

        let mut fired = false;
        for _ in 0..200 {
            if neuron.step(&mut state, 0.5, 10.0) {
                fired = true;
                break;
            }
        }
        assert!(fired, "Izhikevich neuron should fire with input current");
    }
}
