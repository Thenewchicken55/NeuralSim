struct Uniforms {
    dt: f32,
    sim_time: f32,
};

// Hodgkin-Huxley gating variables: m, h, n
@group(0) @binding(0) var<storage, read_write> membrane_potential: array<f32>;
@group(0) @binding(1) var<storage, read_write> recovery_variable: array<f32>;
@group(0) @binding(2) var<storage, read_write> refractory_counter: array<i32>;
@group(0) @binding(3) var<storage, read_write> last_spike_time: array<f32>;
@group(0) @binding(4) var<storage, read_write> spike_count: array<u32>;
@group(0) @binding(5) var<storage, read> input_current: array<f32>;
@group(0) @binding(6) var<storage, read> params: array<vec4<f32>>; // g_na, g_k, g_l, e_na, e_k, e_l, c_m (packed)
@group(0) @binding(7) var<storage, read> is_output: array<u32>;
@group(0) @binding(8) var<storage, read_write> spiked: array<u32>;
// Gate state stored in recovery_variable as vec3:
// recovery_variable[i*3] = m, recovery_variable[i*3+1] = h, recovery_variable[i*3+2] = n
@group(0) @binding(9) var<storage, read_write> gate_m: array<f32>;
@group(0) @binding(10) var<storage, read_write> gate_h: array<f32>;
@group(0) @binding(11) var<storage, read_write> gate_n: array<f32>;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

fn alpha_m(v: f32) -> f32 {
    return 0.1 * (v + 40.0) / (1.0 - exp(-(v + 40.0) / 10.0));
}
fn beta_m(v: f32) -> f32 {
    return 4.0 * exp(-(v + 65.0) / 18.0);
}
fn alpha_h(v: f32) -> f32 {
    return 0.07 * exp(-(v + 65.0) / 20.0);
}
fn beta_h(v: f32) -> f32 {
    return 1.0 / (1.0 + exp(-(v + 35.0) / 10.0));
}
fn alpha_n(v: f32) -> f32 {
    return 0.01 * (v + 55.0) / (1.0 - exp(-(v + 55.0) / 10.0));
}
fn beta_n(v: f32) -> f32 {
    return 0.125 * exp(-(v + 65.0) / 80.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&membrane_potential);
    if (i >= n) { return; }

    let dt = uniforms.dt;
    let v = membrane_potential[i];

    if (v >= -55.0) {
        membrane_potential[i] = -70.0;
        last_spike_time[i] = uniforms.sim_time;
        spike_count[i] = spike_count[i] + 1u;
        spiked[i] = 1u;
        return;
    }
    spiked[i] = 0u;

    // Gating variable updates (forward Euler)
    let am = alpha_m(v);
    let bm = beta_m(v);
    let ah = alpha_h(v);
    let bh = beta_h(v);
    let an = alpha_n(v);
    let bn = beta_n(v);

    let m = gate_m[i];
    let h = gate_h[i];
    let n_g = gate_n[i];

    gate_m[i] = m + (am * (1.0 - m) - bm * m) * dt;
    gate_h[i] = h + (ah * (1.0 - h) - bh * h) * dt;
    gate_n[i] = n_g + (an * (1.0 - n_g) - bn * n_g) * dt;

    // Ionic currents
    let p = params[i];
    let g_na = p.x;
    let g_k = p.y;
    let g_l = p.z;
    let e_na = 50.0;
    let e_k = -77.0;
    let e_l = p.w;

    let i_na = g_na * gate_m[i] * gate_m[i] * gate_m[i] * gate_h[i] * (e_na - v);
    let i_k = g_k * gate_n[i] * gate_n[i] * gate_n[i] * gate_n[i] * (e_k - v);
    let i_l = g_l * (e_l - v);
    let I = input_current[i];

    membrane_potential[i] = v + (I - i_na - i_k - i_l) * dt;
}
