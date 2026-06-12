use super::{NeuronModel, NeuronModelParams, NeuronState, NeuronType};
use serde::{Deserialize, Serialize};

/// Leaky Integrate-and-Fire neuron model.
///
/// Membrane potential dynamics:
///   τ_m * dV/dt = - (V - V_rest) + R_m * I(t)
///
/// When V >= V_threshold, the neuron fires, V is reset to V_reset,
/// and the neuron enters a refractory period.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LifNeuron;

impl NeuronModel for LifNeuron {
    fn step(&mut self, state: &mut NeuronState, dt: f64, input_current: f64) -> bool {
        let params = match &state.model_params {
            NeuronModelParams::Lif { .. } => &state.model_params,
            _ => return false,
        };

        let (resting, threshold, reset, tau_m, refractory_period, input_resistance) = match params
        {
            NeuronModelParams::Lif {
                resting,
                threshold,
                reset,
                tau_m,
                refractory_period,
                input_resistance,
            } => (*resting, *threshold, *reset, *tau_m, *refractory_period, *input_resistance),
            _ => unreachable!(),
        };

        if state.refractory_counter > 0 {
            state.refractory_counter -= 1;
            return false;
        }

        let dv = (-(state.membrane_potential - resting) + input_resistance * input_current) * dt / tau_m;
        state.membrane_potential += dv;

        if state.membrane_potential >= threshold {
            state.membrane_potential = reset;
            state.refractory_counter = (refractory_period / dt).round() as i32;
            state.last_spike_time = 0.0; // caller sets absolute time
            state.spike_count += 1;
            return true;
        }

        false
    }

    fn reset_state(&self, neuron_type: NeuronType) -> NeuronState {
        NeuronState::new(neuron_type, NeuronModelParams::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_subthreshold() {
        let mut state = NeuronState::new(NeuronType::Excitatory, NeuronModelParams::default());
        let mut neuron = LifNeuron;
        let spiked = neuron.step(&mut state, 0.1, 0.0);
        assert!(!spiked);
    }

    #[test]
    fn test_lif_fires_at_threshold() {
        let mut state = NeuronState::new(NeuronType::Excitatory, NeuronModelParams::default());
        state.membrane_potential = -49.0;
        let mut neuron = LifNeuron;
        let spiked = neuron.step(&mut state, 0.1, 0.0);
        assert!(spiked);
        assert!(state.membrane_potential < -60.0);
    }

    #[test]
    fn test_lif_refractory_period() {
        let mut state = NeuronState::new(NeuronType::Excitatory, NeuronModelParams::default());
        state.membrane_potential = -49.0;
        let mut neuron = LifNeuron;
        let first = neuron.step(&mut state, 0.1, 0.0);
        assert!(first);
        let second = neuron.step(&mut state, 0.1, 1000.0);
        assert!(!second, "should be refractory");
    }
}
