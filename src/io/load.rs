use crate::network::Network;
use std::path::Path;

pub fn load_network_json(path: impl AsRef<Path>) -> Result<Network, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(path)?;
    let network: Network = serde_json::from_str(&json)?;
    Ok(network)
}

pub fn load_network_binary(path: impl AsRef<Path>) -> Result<Network, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let network: Network = serde_json::from_slice(&data)?;
    Ok(network)
}
