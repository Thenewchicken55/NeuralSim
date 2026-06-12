struct LifParams {
    resting: f32,
    threshold: f32,
    reset: f32,
    tau_m: f32,
    refractory_period: f32,
    input_resistance: f32,
};

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
@group(0) @binding(6) var<storage, read> params: array<LifParams>;
@group(0) @binding(7) var<storage, read> is_output: array<u32>;
@group(0) @binding(8) var<storage, read_write> spiked: array<u32>;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&membrane_potential);
    if (i >= n) { return; }

    let p = params[i];

    if (refractory_counter[i] > 0) {
        refractory_counter[i] = refractory_counter[i] - 1;
        membrane_potential[i] = p.resting;
        spiked[i] = 0u;
        return;
    }

    let V = membrane_potential[i];
    let I = input_current[i];
    let dV = (-(V - p.resting) + p.input_resistance * I) * uniforms.dt / p.tau_m;
    membrane_potential[i] = V + dV;

    if (membrane_potential[i] >= p.threshold) {
        membrane_potential[i] = p.reset;
        refractory_counter[i] = i32(round(p.refractory_period / uniforms.dt));
        spike_count[i] = spike_count[i] + 1u;
        last_spike_time[i] = uniforms.sim_time;
        spiked[i] = 1u;
    } else {
        spiked[i] = 0u;
    }
}
