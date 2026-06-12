@group(0) @binding(0) var<storage, read> spiked: array<u32>;
@group(0) @binding(1) var<storage, read> adjacency_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> adjacency_indices: array<u32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> atomic_current: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&spiked);
    if (i >= n) { return; }

    if (spiked[i] == 0u) { return; }

    let start = adjacency_ptr[i];
    let end = adjacency_ptr[i + 1u];
    for (var idx = start; idx < end; idx = idx + 1u) {
        let target = adjacency_indices[idx];
        let w = weights[idx];
        let w_bits = bitcast<u32>(w);

        var old = atomicLoad(&atomic_current[target]);
        loop {
            let new = bitcast<u32>(bitcast<f32>(old) + w);
            let result = atomicCompareExchangeWeak(&atomic_current[target], old, new);
            if (result.exchanged) {
                break;
            }
            old = result.old_value;
        }
    }
}
