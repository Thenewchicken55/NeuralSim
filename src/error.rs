//! Error types for NeuralSim operations.
//!
//! Replaces ad-hoc `Box<dyn std::error::Error>` returns with a structured
//! enum that distinguishes I/O, serialization, configuration, and GPU failures.
//!
//! # Usage
//!
//! ```ignore
//! use neural_sim::error::{Result, NeuralSimError};
//!
//! fn load_config(path: &str) -> Result<Config> {
//!     let raw = std::fs::read_to_string(path)
//!         .map_err(|e| NeuralSimError::io("read config", Some(path.into()), e))?;
//!     Ok(serde_json::from_str(&raw)?)
//! }
//! ```

use std::path::PathBuf;
use thiserror::Error;

/// Top-level error type used by all public I/O and config functions.
#[derive(Debug, Error)]
pub enum NeuralSimError {
    #[error("I/O error: {message} (path: {path:?})")]
    Io {
        message: String,
        path: Option<PathBuf>,
        #[source]
        source: std::io::Error,
    },

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML serialization error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("config error: {0}")]
    Config(String),

    #[error("checkpoint error: {message}")]
    Checkpoint { message: String },

    #[cfg(feature = "gpu")]
    #[error("GPU backend error: {0}")]
    Gpu(#[from] crate::simulation::gpu_backend::GpuError),
}

impl NeuralSimError {
    /// Convenience constructor for I/O errors with a path.
    pub fn io(
        message: impl Into<String>,
        path: impl Into<Option<PathBuf>>,
        source: std::io::Error,
    ) -> Self {
        Self::Io {
            message: message.into(),
            path: path.into(),
            source,
        }
    }

    /// Convenience constructor for checkpoint errors.
    pub fn checkpoint(message: impl Into<String>) -> Self {
        Self::Checkpoint {
            message: message.into(),
        }
    }
}

/// Alias used pervasively across the crate for ergonomic `?` propagation.
pub type Result<T> = std::result::Result<T, NeuralSimError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = NeuralSimError::Config("missing field `seed`".into());
        assert!(format!("{e}").contains("missing field"));
    }

    #[test]
    fn test_io_error_constructor() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e = NeuralSimError::io("read failed", Some(PathBuf::from("/tmp/x")), io_err);
        assert!(format!("{e}").contains("read failed"));
        assert!(format!("{e}").contains("/tmp/x"));
    }
}
