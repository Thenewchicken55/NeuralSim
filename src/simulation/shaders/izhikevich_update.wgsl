struct Uniforms {
    dt: f32,
    sim_time: f32,
};

@group(0) @binding(0) var<storage, read_write> membrane_potential: array<f32>;
@group(0) @binding(1) var<storage, read_write> recovery_variable: array<f32>;
@group(0) @binding(2) var<storage, read_write> refractory_counter: array<i32>;
@group(0) @binding(3) var<storage, read_write> last_spike_time: array<f32>;
@group(0) @binding(4) var<storage, read_write> spike_count: array<u32>;
@group(0) @binding(5) var<storage, read> input_current: array<f32>;
@group(0) @binding(6) var<storage, read> params: array<vec4<f32>>;  // a, b, c, d
@group(0) @binding(7) var<storage, read> is_output: array<u32>;
@group(0) @binding(8) var<storage, read_write> spiked: array<u32>;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&membrane_potential);
    if (i >= n) { return; }

    let p = params[i];
    let a = p.x;
    let b = p.y;
    let c = p.z;
    let d = p.w;

    // Check for threshold crossing from previous step
    if (membrane_potential[i] >= 30.0) {
        membrane_potential[i] = c;
        recovery_variable[i] = recovery_variable[i] + d;
        last_spike_time[i] = uniforms.sim_time;
        spike_count[i] = spike_count[i] + 1u;
        spiked[i] = 1u;
        return;
    }

    spiked[i] = 0u;

    // Izhikevich ODE step (forward Euler)
    let v = membrane_potential[i];
    let u = recovery_variable[i];
    let I = input_current[i];

    let dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I) * uniforms.dt;
    let du = a * (b * v - u) * uniforms.dt;

    membrane_potential[i] = v + dv;
    recovery_variable[i] = u + du;
}
