use serde::{Deserialize, Serialize};

/// Compressed Sparse Row (CSR) adjacency matrix.
/// Memory efficient representation for large sparse graphs typical of neural networks.
///
/// Each neuron has a contiguous range [row_ptr[i], row_ptr[i+1]) of outgoing
/// synapse indices in `col_indices` / `values`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjacencyMatrix {
    /// row_ptr[i] = start index of neuron i's outgoing edges
    pub row_ptr: Vec<usize>,
    /// target neuron index for each edge
    pub col_indices: Vec<usize>,
    /// weight for each edge (or index into synapse array)
    pub values: Vec<f64>,
    /// Number of neurons
    pub n: usize,
}

impl AdjacencyMatrix {
    pub fn new(n: usize) -> Self {
        Self {
            row_ptr: vec![0; n + 1],
            col_indices: Vec::with_capacity(n * 10),
            values: Vec::with_capacity(n * 10),
            n,
        }
    }

    pub fn edges_from(&self, neuron: usize) -> &[usize] {
        let start = self.row_ptr[neuron];
        let end = self.row_ptr[neuron + 1];
        &self.col_indices[start..end]
    }

    pub fn weights_from(&self, neuron: usize) -> &[f64] {
        let start = self.row_ptr[neuron];
        let end = self.row_ptr[neuron + 1];
        &self.values[start..end]
    }

    pub fn degree(&self, neuron: usize) -> usize {
        self.row_ptr[neuron + 1] - self.row_ptr[neuron]
    }

    pub fn total_edges(&self) -> usize {
        self.col_indices.len()
    }

    pub fn add_edge(&mut self, source: usize, target: usize, weight: f64) {
        // Insert edge at the end of source's row by shifting subsequent rows
        for ptr in self.row_ptr.iter_mut().skip(source + 1) {
            *ptr += 1;
        }
        let insert_at = self.row_ptr[source + 1] - 1;
        self.col_indices.insert(insert_at, target);
        self.values.insert(insert_at, weight);
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.row_ptr.len() * std::mem::size_of::<usize>()
            + self.col_indices.len() * std::mem::size_of::<usize>()
            + self.values.len() * std::mem::size_of::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_add_edge() {
        let mut mat = AdjacencyMatrix::new(5);
        mat.add_edge(0, 3, 0.5);
        assert_eq!(mat.edges_from(0), &[3]);
        assert_eq!(mat.degree(0), 1);
        assert_eq!(mat.degree(1), 0);
    }

    #[test]
    fn test_csr_multiple_edges() {
        let mut mat = AdjacencyMatrix::new(3);
        mat.add_edge(0, 1, 1.0);
        mat.add_edge(0, 2, 2.0);
        assert_eq!(mat.edges_from(0), &[1, 2]);
        assert_eq!(mat.weights_from(0), &[1.0, 2.0]);
    }
}
