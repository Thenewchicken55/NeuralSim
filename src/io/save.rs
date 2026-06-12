use crate::network::Network;
use std::path::Path;

pub fn save_network_json(network: &Network, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(network)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn save_network_binary(network: &Network, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_vec(network)?;
    std::fs::write(path, json)?;
    Ok(())
}
