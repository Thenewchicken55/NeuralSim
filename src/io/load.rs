use crate::error::{NeuralSimError, Result};
use crate::network::Network;
use std::path::Path;

/// Load a network from a pretty-printed JSON file.
pub fn load_network_json(path: impl AsRef<Path>) -> Result<Network> {
    let json = std::fs::read_to_string(path.as_ref()).map_err(|e| {
        NeuralSimError::io(
            "failed to read network JSON",
            Some(path.as_ref().to_path_buf()),
            e,
        )
    })?;
    let network: Network = serde_json::from_str(&json)?;
    Ok(network)
}

/// Load a network from a binary JSON file.
pub fn load_network_binary(path: impl AsRef<Path>) -> Result<Network> {
    let data = std::fs::read(path.as_ref()).map_err(|e| {
        NeuralSimError::io(
            "failed to read network binary",
            Some(path.as_ref().to_path_buf()),
            e,
        )
    })?;
    let network: Network = serde_json::from_slice(&data)?;
    Ok(network)
}
