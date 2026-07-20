use crate::network::Network;
use crate::network::builder::BrainBuilder;
use crate::neuron::NeuronModelParams;

/// Simple builder wrapping BrainBuilder for quick setup.
pub struct NetworkBuilder {
    size: usize,
    use_layers: bool,
}

impl NetworkBuilder {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            use_layers: false,
        }
    }

    pub fn with_default_layers(mut self) -> Self {
        self.use_layers = true;
        self
    }

    pub fn build(&self) -> Network {
        if self.use_layers {
            BrainBuilder::new()
                .add_cortical_column("Default", self.size)
                .build()
        } else {
            BrainBuilder::new()
                .add_region("Default", self.size, 0.8, NeuronModelParams::default())
                .build()
        }
    }
}
