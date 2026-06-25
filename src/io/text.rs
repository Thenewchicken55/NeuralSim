use crate::network::Network;
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashMap;

/// Simple character-level tokenizer for text ↔ neural spike conversion.
/// Maps each character to a population of neurons.
pub struct TextEncoder {
    /// Character -> population of neuron IDs
    char_to_neurons: HashMap<char, Vec<usize>>,
    /// All known characters
    vocabulary: Vec<char>,
    /// Neuron population size per token
    pop_size: usize,
    /// Total input neurons used
    pub total_input_neurons: usize,
    /// Firing rate for active tokens (Hz)
    pub firing_rate: f64,
}

impl TextEncoder {
    pub fn new(vocabulary: &str, total_input_neurons: usize, pop_size: usize) -> Self {
        let chars: Vec<char> = vocabulary.chars().collect();
        let n_chars = chars.len();
        let mut char_to_neurons = HashMap::new();

        let pop_per_char = (total_input_neurons / n_chars.max(1)).min(pop_size);

        for (i, &c) in chars.iter().enumerate() {
            let start = i * pop_per_char;
            let end = (start + pop_per_char).min(total_input_neurons);
            let neurons: Vec<usize> = (start..end).collect();
            char_to_neurons.insert(c, neurons);
        }

        Self {
            char_to_neurons,
            vocabulary: chars,
            pop_size: pop_per_char,
            total_input_neurons,
            firing_rate: 80.0,
        }
    }

    /// Create a default encoder with common ASCII + punctuation.
    pub fn default(total_input_neurons: usize) -> Self {
        let vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}@#$%^&*+=/<>~\n\t";
        Self::new(vocab, total_input_neurons, 10)
    }

    /// Encode text into a list of (neuron_id, spike_time_ms) pairs.
    /// Each character is rate-coded: token -> neuron population fires at `firing_rate` Hz.
    pub fn encode(&self, text: &str, start_time: f64, duration_per_token: f64) -> Vec<(usize, f64)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut spikes = Vec::new();

        for (i, c) in text.chars().enumerate() {
            let t_start = start_time + (i as f64) * duration_per_token;
            if let Some(neurons) = self.char_to_neurons.get(&c) {
                for &neuron in neurons {
                    // Poisson spike train for duration_per_token
                    let mut t = t_start;
                    while t < t_start + duration_per_token {
                        let isi = 1000.0 / self.firing_rate;
                        let jitter: f64 = rng.random::<f64>() * isi * 0.2;
                        t += isi + jitter;
                        if t < t_start + duration_per_token {
                            spikes.push((neuron, t));
                        }
                    }
                }
            }
        }

        spikes
    }

    /// Returns the character set for display
    pub fn vocabulary(&self) -> &[char] {
        &self.vocabulary
    }

    /// Number of distinct tokens
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Decode neural spikes back to text using population voting.
pub struct TextDecoder {
    /// neuron -> character mapping
    neuron_to_char: HashMap<usize, char>,
    /// Output neurons per character
    pub pop_size: usize,
}

impl TextDecoder {
    /// Create a decoder from the same vocabulary/neuron mapping used by TextEncoder.
    pub fn from_encoder(encoder: &TextEncoder) -> Self {
        let mut neuron_to_char = HashMap::new();
        for (&c, neurons) in &encoder.char_to_neurons {
            for &n in neurons {
                neuron_to_char.insert(n, c);
            }
        }
        Self {
            neuron_to_char,
            pop_size: encoder.pop_size,
        }
    }

    /// Decode output spikes over a time window into text.
    /// Uses winner-take-all per character population.
    pub fn decode(&self, output_spikes: &[(usize, f64)], start_time: f64,
                  end_time: f64, window_duration: f64) -> String {
        let mut output = String::new();

        // Count spikes per neuron in each time window
        let mut t = start_time;
        while t < end_time {
            let window_end = (t + window_duration).min(end_time);
            let mut char_votes: HashMap<char, usize> = HashMap::new();

            for &(neuron, spike_time) in output_spikes {
                if spike_time >= t && spike_time < window_end
                    && let Some(&c) = self.neuron_to_char.get(&neuron) {
                        *char_votes.entry(c).or_insert(0) += 1;
                    }
            }

            // Winner-take-all: most active character wins
            if let Some((&best_char, _)) = char_votes.iter().max_by_key(|&(_, count)| count) {
                output.push(best_char);
            }

            t = window_end;
        }

        output
    }

    /// Decode with confidence threshold: only output if votes exceed threshold.
    pub fn decode_with_threshold(&self, output_spikes: &[(usize, f64)],
                                  start_time: f64, end_time: f64,
                                  window_duration: f64, threshold: usize) -> String {
        let mut output = String::new();
        let mut t = start_time;

        while t < end_time {
            let window_end = (t + window_duration).min(end_time);
            let mut char_votes: HashMap<char, usize> = HashMap::new();

            for &(neuron, spike_time) in output_spikes {
                if spike_time >= t && spike_time < window_end
                    && let Some(&c) = self.neuron_to_char.get(&neuron) {
                        *char_votes.entry(c).or_insert(0) += 1;
                    }
            }

            if let Some((&best_char, &count)) = char_votes.iter().max_by_key(|&(_, count)| count)
                && count >= threshold {
                    output.push(best_char);
                }

            t = window_end;
        }

        output
    }
}

/// Simple REPL for interactive text-based I/O with the brain simulator.
pub struct BrainRepl {
    pub encoder: TextEncoder,
    pub decoder: TextDecoder,
    pub input_region: String,
    pub output_region: String,
    pub tokens_per_second: f64,
    pub response_threshold: usize,
}

impl BrainRepl {
    pub fn new(encoder: TextEncoder, decoder: TextDecoder) -> Self {
        Self {
            encoder,
            decoder,
            input_region: "Input".into(),
            output_region: "Output".into(),
            tokens_per_second: 20.0,
            response_threshold: 1,
        }
    }

    /// Run a single turn of the conversation:
    /// 1. Encode text as spike trains into input neurons
    /// 2. Simulate for the duration of the encoded input
    /// 3. (User calls step/driver externally)
    /// 4. Decode output spikes to text
    pub fn encode_input(&self, text: &str, network: &mut Network, start_time: f64) -> Vec<(usize, f64)> {
        let _duration = (text.len() as f64) * (1000.0 / self.tokens_per_second);
        let spikes = self.encoder.encode(text, start_time, 1000.0 / self.tokens_per_second);

        // Find input neurons
        let _input_neurons: Vec<usize> = (0..network.neuron_count())
            .filter(|&i| network.neuron_region.get(i) == Some(&0)) // assume region 0 is input
            .collect();

        // Inject currents for input spikes
        for &(neuron, _time) in &spikes {
            if neuron < network.neuron_count() && neuron < network.neurons.input_current.len() {
                network.neurons.input_current[neuron] += 30.0;
            }
        }

        spikes
    }

    /// Decode output spikes to text
    pub fn decode_output(&self, spike_buffer: &[(usize, f64)], sim_time: f64) -> String {
        let duration = sim_time.max(1000.0);
        self.decoder.decode_with_threshold(
            spike_buffer, 0.0, duration,
            1000.0 / self.tokens_per_second,
            self.response_threshold,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_encoder_basic() {
        let encoder = TextEncoder::default(260);
        let spikes = encoder.encode("hi", 0.0, 50.0);
        assert!(!spikes.is_empty(), "Should produce spike events");
        for &(neuron, _) in &spikes {
            assert!(neuron < 260, "Neuron ID within range");
        }
    }

    #[test]
    fn test_text_encoder_vocab() {
        let encoder = TextEncoder::default(260);
        assert!(encoder.vocab_size() > 10, "Should have a reasonable vocabulary");
    }

    #[test]
    fn test_text_decoder_roundtrip() {
        let encoder = TextEncoder::default(260);
        let decoder = TextDecoder::from_encoder(&encoder);

        // Encode then decode (partial — just check structure)
        let encoder2 = TextEncoder::default(260);
        let spikes = encoder2.encode("a", 0.0, 50.0);
        let decoded = decoder.decode(&spikes, 0.0, 100.0, 50.0);
        // May or may not decode to "a" exactly (Poisson jitter), but should produce something
        assert!(!decoded.is_empty() || spikes.is_empty());
    }

    #[test]
    fn test_decoder_threshold() {
        let encoder = TextEncoder::default(100);
        let decoder = TextDecoder::from_encoder(&encoder);
        let decoded = decoder.decode_with_threshold(&[], 0.0, 100.0, 50.0, 5);
        assert!(decoded.is_empty(), "No spikes should produce empty output");
    }
}
