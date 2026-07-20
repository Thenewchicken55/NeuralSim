//! Serialization, checkpointing, and text I/O.
//!
//! - [`save`] / [`load`] — JSON network save/load
//! - [`checkpoint`] — `CheckpointManager` with pruning, `StatsRecorder` for CSV export
//! - [`text`] — `TextEncoder`/`TextDecoder` for character-level text↔spike conversion

pub mod checkpoint;
pub mod load;
pub mod save;
pub mod text;
