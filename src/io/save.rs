use crate::error::{NeuralSimError, Result};
use crate::network::Network;
use std::path::Path;

/// Save a network to disk as pretty-printed JSON.
pub fn save_network_json(network: &Network, path: impl AsRef<Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(network)?;
    std::fs::write(path.as_ref(), json).map_err(|e| {
        NeuralSimError::io(
            "failed to write network JSON",
            Some(path.as_ref().to_path_buf()),
            e,
        )
    })?;
    Ok(())
}

/// Save a network to disk as compact binary JSON (postcard-style bytes).
pub fn save_network_binary(network: &Network, path: impl AsRef<Path>) -> Result<()> {
    let json = serde_json::to_vec(network)?;
    std::fs::write(path.as_ref(), json).map_err(|e| {
        NeuralSimError::io(
            "failed to write network binary",
            Some(path.as_ref().to_path_buf()),
            e,
        )
    })?;
    Ok(())
}
