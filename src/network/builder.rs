use crate::network::Network;
use crate::network::connectivity::ConnectivityPattern;
use crate::neuron::{NeuronModelParams, NeuronType};
use crate::synapse::{SynapseState, SynapseType};
use rand::Rng;
use rand::SeedableRng;

/// Builder for constructing multi-region brain networks with realistic connectivity.
///
/// # Example
/// ```ignore
/// use neural_sim::network::builder::BrainBuilder;
/// let mut brain = BrainBuilder::new()
///     .add_region("Cortex", 10000, 0.80, NeuronModelParams::default())
///     .add_region("Thalamus", 2000, 0.90, NeuronModelParams::default())
///     .connect_regions("Thalamus", "Cortex", 0.05, 1.5, None)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BrainBuilder {
    regions: Vec<RegionSpec>,
    connections: Vec<RegionConnection>,
    enable_plasticity: bool,
    name: String,
}

#[derive(Debug, Clone)]
struct RegionSpec {
    name: String,
    neuron_count: usize,
    excitatory_ratio: f64,
    model_params: NeuronModelParams,
    layers: Option<Vec<LayerSpec>>,
    is_input: bool,
    is_output: bool,
}

#[derive(Debug, Clone)]
struct LayerSpec {
    name: String,
    fraction: f64,
    excitatory_ratio: f64,
    _connectivity: ConnectivityPattern,
}

#[derive(Debug, Clone)]
struct RegionConnection {
    from_region: usize,
    to_region: usize,
    probability: f64,
    weight_scale: f64,
    syn_type: Option<SynapseType>,
    from_layer: Option<usize>,
    to_layer: Option<usize>,
}

impl BrainBuilder {
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
            connections: Vec::new(),
            enable_plasticity: true,
            name: "Brain".into(),
        }
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_plasticity(mut self, enable: bool) -> Self {
        self.enable_plasticity = enable;
        self
    }

    /// Add a single-region population.
    pub fn add_region(
        mut self,
        name: &str,
        count: usize,
        exc_ratio: f64,
        params: NeuronModelParams,
    ) -> Self {
        let idx = self.regions.len();
        self.regions.push(RegionSpec {
            name: name.into(),
            neuron_count: count,
            excitatory_ratio: exc_ratio,
            model_params: params,
            layers: None,
            is_input: false,
            is_output: false,
        });
        self.regions[idx].is_input = name.contains("Input") || name.contains("Sensory");
        self.regions[idx].is_output = name.contains("Output") || name.contains("Motor");
        self
    }

    /// Add a 6-layer cortical column region.
    pub fn add_cortical_column(mut self, name: &str, total_neurons: usize) -> Self {
        let fractions = [0.15, 0.20, 0.25, 0.10, 0.20, 0.10];
        let exc_ratios = [0.85, 0.80, 0.75, 0.50, 0.80, 0.70];
        let layer_names = ["L2/3", "L4", "L5", "L4_IN", "L5_IN", "L6"];
        let layer_connectivity = [
            ConnectivityPattern::ErdosRenyi { p: 0.05 },
            ConnectivityPattern::ErdosRenyi { p: 0.08 },
            ConnectivityPattern::ErdosRenyi { p: 0.06 },
            ConnectivityPattern::ErdosRenyi { p: 0.04 },
            ConnectivityPattern::ErdosRenyi { p: 0.04 },
            ConnectivityPattern::ErdosRenyi { p: 0.04 },
        ];

        let layers: Vec<LayerSpec> = fractions
            .iter()
            .enumerate()
            .map(|(i, &frac)| LayerSpec {
                name: format!("{}_{}", name, layer_names[i]),
                fraction: frac,
                excitatory_ratio: exc_ratios[i],
                _connectivity: layer_connectivity[i].clone(),
            })
            .collect();

        self.regions.push(RegionSpec {
            name: name.into(),
            neuron_count: total_neurons,
            excitatory_ratio: 0.80,
            model_params: NeuronModelParams::default(),
            layers: Some(layers),
            is_input: false,
            is_output: false,
        });
        self
    }

    /// Mark region as input (receives external stimulation).
    pub fn mark_input(mut self, name: &str) -> Self {
        if let Some(r) = self.regions.iter_mut().find(|r| r.name == name) {
            r.is_input = true;
        }
        self
    }

    /// Mark region as output (read for decoding).
    pub fn mark_output(mut self, name: &str) -> Self {
        if let Some(r) = self.regions.iter_mut().find(|r| r.name == name) {
            r.is_output = true;
        }
        self
    }

    /// Connect two regions with given probability.
    pub fn connect_regions(
        mut self,
        from: &str,
        to: &str,
        probability: f64,
        weight_scale: f64,
        syn_type: Option<SynapseType>,
    ) -> Self {
        let from_idx = self
            .regions
            .iter()
            .position(|r| r.name == from)
            .expect("from region not found");
        let to_idx = self
            .regions
            .iter()
            .position(|r| r.name == to)
            .expect("to region not found");
        self.connections.push(RegionConnection {
            from_region: from_idx,
            to_region: to_idx,
            probability,
            weight_scale,
            syn_type,
            from_layer: None,
            to_layer: None,
        });
        self
    }

    /// Connect specific layers between two regions.
    pub fn connect_layers(
        mut self,
        from_region: &str,
        from_layer: &str,
        to_region: &str,
        to_layer: &str,
        probability: f64,
        weight_scale: f64,
    ) -> Self {
        let from_idx = self
            .regions
            .iter()
            .position(|r| r.name == from_region)
            .expect("from region not found");
        let to_idx = self
            .regions
            .iter()
            .position(|r| r.name == to_region)
            .expect("to region not found");

        let from_layer_idx = self.regions[from_idx]
            .layers
            .as_ref()
            .and_then(|l| l.iter().position(|ls| ls.name.contains(from_layer)));
        let to_layer_idx = self.regions[to_idx]
            .layers
            .as_ref()
            .and_then(|l| l.iter().position(|ls| ls.name.contains(to_layer)));

        self.connections.push(RegionConnection {
            from_region: from_idx,
            to_region: to_idx,
            probability,
            weight_scale,
            syn_type: None,
            from_layer: from_layer_idx,
            to_layer: to_layer_idx,
        });
        self
    }

    /// Build the network.
    pub fn build(&self) -> Network {
        let total_neurons: usize = self.regions.iter().map(|r| r.neuron_count).sum();
        let mut net = Network::new(total_neurons);
        net.name = self.name.clone();
        net.region_names = self.regions.iter().map(|r| r.name.clone()).collect();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut offset = 0usize;

        // Assign neurons to regions with proper types
        for (rid, region) in self.regions.iter().enumerate() {
            match &region.layers {
                Some(layers) => {
                    let mut layer_offsets = Vec::new();
                    let mut current = offset;

                    for (li, layer) in layers.iter().enumerate() {
                        let count = if li == layers.len() - 1 {
                            region.neuron_count - (current - offset)
                        } else {
                            (region.neuron_count as f64 * layer.fraction) as usize
                        };
                        layer_offsets.push((current, count));
                        for i in current..current + count {
                            if i < net.neuron_count() {
                                net.neuron_region[i] = rid;
                                if rng.random::<f64>() < layer.excitatory_ratio {
                                    net.neurons.neuron_type[i] = NeuronType::Excitatory;
                                } else {
                                    net.neurons.neuron_type[i] = NeuronType::Inhibitory;
                                }
                                net.neurons.model_params[i] = region.model_params;
                                // Set Izhikevich params based on type
                                if let NeuronModelParams::Izhikevich { .. } =
                                    &net.neurons.model_params[i]
                                {
                                    match net.neurons.neuron_type[i] {
                                        NeuronType::Excitatory => {
                                            net.neurons.model_params[i] =
                                                NeuronModelParams::Izhikevich {
                                                    a: 0.02,
                                                    b: 0.2,
                                                    c: -65.0,
                                                    d: 8.0,
                                                };
                                        }
                                        NeuronType::Inhibitory => {
                                            net.neurons.model_params[i] =
                                                NeuronModelParams::Izhikevich {
                                                    a: 0.02,
                                                    b: 0.25,
                                                    c: -65.0,
                                                    d: 2.0,
                                                };
                                        }
                                    }
                                }
                                if region.is_input {
                                    net.neurons.input_current[i] = 5.0;
                                }
                                if region.is_output {
                                    net.neurons.is_output[i] = true;
                                }
                            }
                        }
                        current += count;
                    }
                    offset = current;

                    // Layer-specific within-column connectivity
                    // L4 -> L2/3 (sensory input ascends)
                    if let Some(&(l4_offset, l4_count)) = layer_offsets.get(1)
                        && let Some(&(l23_offset, l23_count)) = layer_offsets.first()
                    {
                        for si in l4_offset..l4_offset + l4_count {
                            for tj in l23_offset..l23_offset + l23_count {
                                if rng.random::<f64>() < 0.30 {
                                    net.add_synapse(Self::make_syn(si, tj, &mut rng));
                                }
                            }
                        }
                    }
                    // L2/3 -> L5
                    if let Some(&(l23_offset, l23_count)) = layer_offsets.first()
                        && let Some(&(l5_offset, l5_count)) = layer_offsets.get(2)
                    {
                        for si in l23_offset..l23_offset + l23_count {
                            for tj in l5_offset..l5_offset + l5_count {
                                if rng.random::<f64>() < 0.20 {
                                    net.add_synapse(Self::make_syn(si, tj, &mut rng));
                                }
                            }
                        }
                    }
                    // L5 -> L6
                    if let Some(&(l5_offset, l5_count)) = layer_offsets.get(2)
                        && let Some(&(l6_offset, l6_count)) = layer_offsets.get(5)
                    {
                        for si in l5_offset..l5_offset + l5_count {
                            for tj in l6_offset..l6_offset + l6_count {
                                if rng.random::<f64>() < 0.40 {
                                    net.add_synapse(Self::make_syn(si, tj, &mut rng));
                                }
                            }
                        }
                    }
                    // L4_IN (inhibitory) -> L4 (local inhibition)
                    if let Some(&(l4in_offset, l4in_count)) = layer_offsets.get(3)
                        && let Some(&(l4_offset, l4_count)) = layer_offsets.get(1)
                    {
                        for si in l4in_offset..l4in_offset + l4in_count {
                            for tj in l4_offset..l4_offset + l4_count {
                                if rng.random::<f64>() < 0.25 {
                                    let mut syn = SynapseState::new(
                                        si,
                                        tj,
                                        -rng.random::<f64>() * 2.0,
                                        SynapseType::GABA,
                                    );
                                    if self.enable_plasticity {
                                        syn = syn.with_plasticity();
                                    }
                                    net.add_synapse(syn);
                                }
                            }
                        }
                    }
                    // L5_IN (inhibitory) -> L5
                    if let Some(&(l5in_offset, l5in_count)) = layer_offsets.get(4)
                        && let Some(&(l5_offset, l5_count)) = layer_offsets.get(2)
                    {
                        for si in l5in_offset..l5in_offset + l5in_count {
                            for tj in l5_offset..l5_offset + l5_count {
                                if rng.random::<f64>() < 0.25 {
                                    let mut syn = SynapseState::new(
                                        si,
                                        tj,
                                        -rng.random::<f64>() * 2.0,
                                        SynapseType::GABA,
                                    );
                                    if self.enable_plasticity {
                                        syn = syn.with_plasticity();
                                    }
                                    net.add_synapse(syn);
                                }
                            }
                        }
                    }
                }
                None => {
                    for i in offset..offset + region.neuron_count {
                        if i < net.neuron_count() {
                            net.neuron_region[i] = rid;
                            let is_exc = rng.random::<f64>() < region.excitatory_ratio;
                            net.neurons.neuron_type[i] = if is_exc {
                                NeuronType::Excitatory
                            } else {
                                NeuronType::Inhibitory
                            };
                            net.neurons.model_params[i] = region.model_params;
                            if region.is_input {
                                net.neurons.input_current[i] = 5.0;
                            }
                            if region.is_output {
                                net.neurons.is_output[i] = true;
                            }
                        }
                    }
                    offset += region.neuron_count;
                }
            }
        }

        // Within-region random connectivity
        for rid in 0..self.regions.len() {
            for i in 0..net.neuron_count() {
                if net.neuron_region[i] != rid {
                    continue;
                }
                for j in 0..net.neuron_count() {
                    if i == j || net.neuron_region[j] != rid {
                        continue;
                    }
                    if rng.random::<f64>() < 0.02 {
                        net.add_synapse(Self::make_syn(i, j, &mut rng));
                    }
                }
            }
        }

        // Inter-region connections
        for conn in &self.connections {
            let from_neurons: Vec<usize> = (0..net.neuron_count())
                .filter(|&i| {
                    if net.neuron_region[i] != conn.from_region {
                        return false;
                    }
                    match conn.from_layer {
                        Some(layer_idx) => {
                            // Check if neuron falls in the correct layer offset range
                            let region_start = self.regions[..conn.from_region]
                                .iter()
                                .map(|r| r.neuron_count)
                                .sum::<usize>();
                            let layers = self.regions[conn.from_region].layers.as_ref().unwrap();
                            let mut layer_start = region_start;
                            for layer in layers.iter().take(layer_idx) {
                                layer_start += (layer.fraction
                                    * self.regions[conn.from_region].neuron_count as f64)
                                    as usize;
                            }
                            let layer_count = (layers[layer_idx].fraction
                                * self.regions[conn.from_region].neuron_count as f64)
                                as usize;
                            i >= layer_start && i < layer_start + layer_count
                        }
                        None => true,
                    }
                })
                .collect();
            let to_neurons: Vec<usize> = (0..net.neuron_count())
                .filter(|&i| {
                    if net.neuron_region[i] != conn.to_region {
                        return false;
                    }
                    match conn.to_layer {
                        Some(layer_idx) => {
                            let region_start = self.regions[..conn.to_region]
                                .iter()
                                .map(|r| r.neuron_count)
                                .sum::<usize>();
                            let layers = self.regions[conn.to_region].layers.as_ref().unwrap();
                            let mut layer_start = region_start;
                            for layer in layers.iter().take(layer_idx) {
                                layer_start += (layer.fraction
                                    * self.regions[conn.to_region].neuron_count as f64)
                                    as usize;
                            }
                            let layer_count = (layers[layer_idx].fraction
                                * self.regions[conn.to_region].neuron_count as f64)
                                as usize;
                            i >= layer_start && i < layer_start + layer_count
                        }
                        None => true,
                    }
                })
                .collect();

            for &si in &from_neurons {
                for &tj in &to_neurons {
                    if rng.random::<f64>() < conn.probability {
                        let st = conn.syn_type.unwrap_or_else(|| {
                            if net.neurons.neuron_type[si] == NeuronType::Excitatory {
                                SynapseType::AMPA
                            } else {
                                SynapseType::GABA
                            }
                        });
                        let mut syn = SynapseState::new(
                            si,
                            tj,
                            conn.weight_scale * (rng.random::<f64>() * 0.5 + 0.75),
                            st,
                        );
                        if self.enable_plasticity {
                            syn = syn.with_plasticity();
                        }
                        net.add_synapse(syn);
                    }
                }
            }
        }

        // Build CSR adjacency in O(N + M)
        net.finalize();
        net
    }

    fn make_syn(si: usize, tj: usize, rng: &mut impl Rng) -> SynapseState {
        // Simplified - actual type determined by caller
        SynapseState::new(si, tj, rng.random::<f64>() * 2.0 + 0.5, SynapseType::AMPA)
    }
}

impl Default for BrainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_builder_single_region() {
        let net = BrainBuilder::new()
            .add_region("Test", 1000, 0.8, NeuronModelParams::default())
            .build();
        assert_eq!(net.neuron_count(), 1000);
        assert_eq!(net.region_names.len(), 1);
    }

    #[test]
    fn test_brain_builder_multiple_regions() {
        let net = BrainBuilder::new()
            .add_region("A", 500, 0.8, NeuronModelParams::default())
            .add_region("B", 500, 0.8, NeuronModelParams::default())
            .connect_regions("A", "B", 0.05, 1.0, None)
            .build();
        assert_eq!(net.neuron_count(), 1000);
        assert!(net.synapse_count() > 0);
    }

    #[test]
    fn test_cortical_column_builder() {
        let net = BrainBuilder::new().add_cortical_column("V1", 1000).build();
        assert_eq!(net.neuron_count(), 1000);
        // Should have layer-specific within-column connections
        assert!(net.synapse_count() > 100, "should create layer connections");
    }
}
