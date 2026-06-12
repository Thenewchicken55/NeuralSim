@group(0) @binding(0) var<storage, read_write> atomic_current: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> input_current: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&input_current);
    if (i >= n) { return; }

    let bits = atomicLoad(&atomic_current[i]);
    let val = bitcast<f32>(bits);
    input_current[i] = input_current[i] + val;
    atomicStore(&atomic_current[i], 0u);
}
