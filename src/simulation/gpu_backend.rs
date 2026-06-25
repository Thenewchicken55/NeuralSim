use std::borrow::Cow;

const LIF_SHADER_SOURCE: &str = include_str!("shaders/lif_update.wgsl");
const IZHIKEVICH_SHADER_SOURCE: &str = include_str!("shaders/izhikevich_update.wgsl");
const HH_SHADER_SOURCE: &str = include_str!("shaders/hh_update.wgsl");
const SPMV_SHADER_SOURCE: &str = include_str!("shaders/spmv.wgsl");
const SPMV_CONVERT_SHADER_SOURCE: &str = include_str!("shaders/spmv_convert.wgsl");

/// Which neuron model the GPU backend should use for stepping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuModel {
    Lif,
    Izhikevich,
    HodgkinHuxley,
}

#[derive(Debug)]
pub enum GpuError {
    NoAdapter,
    DeviceRequest(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no suitable GPU adapter found"),
            Self::DeviceRequest(e) => write!(f, "device request failed: {e}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// GPU compute backend for the neural simulation.
///
/// Manages wgpu device/queue, GPU buffers, and compute pipelines for
/// LIF neuron updates and optional SpMV (synapse propagation).
#[allow(dead_code)]
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    num_neurons: usize,
    num_workgroups_neurons: u32,

    // ── LIF buffers ──
    membrane_potential_buf: wgpu::Buffer,
    recovery_variable_buf: wgpu::Buffer,
    refractory_counter_buf: wgpu::Buffer,
    last_spike_time_buf: wgpu::Buffer,
    spike_count_buf: wgpu::Buffer,
    input_current_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    is_output_buf: wgpu::Buffer,
    spiked_buf: wgpu::Buffer,

    // ── SpMV buffers ──
    adjacency_ptr_buf: wgpu::Buffer,
    adjacency_indices_buf: wgpu::Buffer,
    synapse_weights_buf: wgpu::Buffer,
    atomic_current_buf: wgpu::Buffer,

    // ── Uniform buffer (dt, sim_time) ──
    uniform_buf: wgpu::Buffer,

    // ── Bind groups ──
    lif_bind_group: wgpu::BindGroup,
    uniform_bind_group: wgpu::BindGroup,
    spmv_bind_group: wgpu::BindGroup,
    convert_bind_group: wgpu::BindGroup,

    // ── Pipelines ──
    lif_pipeline: wgpu::ComputePipeline,
    izhikevich_pipeline: wgpu::ComputePipeline,
    hh_pipeline: wgpu::ComputePipeline,
    spmv_pipeline: wgpu::ComputePipeline,
    convert_pipeline: wgpu::ComputePipeline,
    /// Which model is currently loaded on GPU
    active_model: GpuModel,

    // ── Staging buffers for readback ──
    staging_buf: wgpu::Buffer,         // LIF readback (spiked + state)
    staging_spiked_offset: u64,
    staging_state_offset: u64,
    staging_size: u64,
    staging_current_buf: wgpu::Buffer, // input_current readback after SpMV
    staging_current_size: u64,
}

impl GpuBackend {
    pub fn new(num_neurons: usize, num_edges: usize) -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async(num_neurons, num_edges))
    }

    async fn new_async(num_neurons: usize, num_edges: usize) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("NeuralSim GPU Backend"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        let n = num_neurons as u64;
        let e = num_edges as u64;
        let b4 = |count: u64| -> u64 { count * 4 };

        let stor_rw = |device: &wgpu::Device, size: u64, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let stor_ro = |device: &wgpu::Device, size: u64, label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let stor_atomic = |device: &wgpu::Device, size: u64, label: &str| -> wgpu::Buffer {
            // atomic<u32> storage: needs STORAGE + COPY_DST (for reset) + COPY_SRC (sync)
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        // LIF buffers
        let membrane_potential_buf = stor_rw(&device, b4(n), "membrane_potential");
        let recovery_variable_buf = stor_rw(&device, b4(n), "recovery_variable");
        let refractory_counter_buf = stor_rw(&device, b4(n), "refractory_counter");
        let last_spike_time_buf = stor_rw(&device, b4(n), "last_spike_time");
        let spike_count_buf = stor_rw(&device, b4(n), "spike_count");
        let input_current_buf = stor_ro(&device, b4(n), "input_current");
        let params_buf = stor_ro(&device, b4(n) * 6, "params");
        let is_output_buf = stor_ro(&device, b4(n), "is_output");
        let spiked_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spiked"),
            size: b4(n),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // SpMV buffers
        let adjacency_ptr_buf = stor_ro(&device, b4(n + 1), "adjacency_ptr");
        let adjacency_indices_buf = stor_ro(&device, b4(e), "adjacency_indices");
        let synapse_weights_buf = stor_ro(&device, b4(e), "synapse_weights");
        let atomic_current_buf = stor_atomic(&device, b4(n), "atomic_current");

        // Uniform buffer
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Staging buffers for readback
        let staging_spiked_offset = 0;
        let staging_state_offset = b4(n);
        let staging_size = staging_state_offset + b4(n) * 4; // spiked + 4 state arrays
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let staging_current_size = b4(n);
        let staging_current_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_current"),
            size: staging_current_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ── Layouts ──

        // LIF storage layout (group 0, 9 bindings)
        let lif_storage_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lif_storage_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // Uniform layout (group 1)
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });

        // SpMV layout (group 0, 5 bindings)
        let spmv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spmv_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // Convert layout (group 0, 2 bindings)
        let convert_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("convert_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // ── Pipeline layouts ──
        let lif_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lif_pipeline_layout"),
            bind_group_layouts: &[&lif_storage_layout, &uniform_layout],
            push_constant_ranges: &[],
        });
        let spmv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spmv_pipeline_layout"),
            bind_group_layouts: &[&spmv_layout],
            push_constant_ranges: &[],
        });
        let convert_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("convert_pipeline_layout"),
            bind_group_layouts: &[&convert_layout],
            push_constant_ranges: &[],
        });

        // ── Shaders ──
        let lif_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lif_update"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(LIF_SHADER_SOURCE)),
        });
        let izhikevich_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("izhikevich_update"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(IZHIKEVICH_SHADER_SOURCE)),
        });
        let hh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hh_update"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(HH_SHADER_SOURCE)),
        });
        let spmv_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spmv"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SPMV_SHADER_SOURCE)),
        });
        let convert_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spmv_convert"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SPMV_CONVERT_SHADER_SOURCE)),
        });

        // ── Pipelines ──
        let lif_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lif_pipeline"),
            layout: Some(&lif_pipeline_layout),
            module: &lif_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let izhikevich_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("izhikevich_pipeline"),
            layout: Some(&lif_pipeline_layout),
            module: &izhikevich_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        // HH pipeline uses 12 bindings (extra gate buffers) — requires separate layout.
        // For now we reuse the same layout but the shader will only be used with
        // additional buffer support added at runtime.
        let hh_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hh_pipeline"),
            layout: Some(&lif_pipeline_layout),
            module: &hh_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let spmv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spmv_pipeline"),
            layout: Some(&spmv_pipeline_layout),
            module: &spmv_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let convert_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("convert_pipeline"),
            layout: Some(&convert_pipeline_layout),
            module: &convert_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── Bind groups ──
        let lif_bind_group = {
            let entries = [
                wgpu::BindGroupEntry { binding: 0, resource: membrane_potential_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: recovery_variable_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: refractory_counter_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: last_spike_time_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: spike_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: input_current_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: is_output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: spiked_buf.as_entire_binding() },
            ];
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lif_bind_group"),
                layout: &lif_storage_layout,
                entries: &entries,
            })
        };

        let uniform_bind_group = {
            let entry = wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("uniform_bind_group"),
                layout: &uniform_layout,
                entries: &[entry],
            })
        };

        let spmv_bind_group = {
            let entries = [
                wgpu::BindGroupEntry { binding: 0, resource: spiked_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: adjacency_ptr_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: adjacency_indices_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: synapse_weights_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: atomic_current_buf.as_entire_binding() },
            ];
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("spmv_bind_group"),
                layout: &spmv_layout,
                entries: &entries,
            })
        };

        let convert_bind_group = {
            let entries = [
                wgpu::BindGroupEntry { binding: 0, resource: atomic_current_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input_current_buf.as_entire_binding() },
            ];
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("convert_bind_group"),
                layout: &convert_layout,
                entries: &entries,
            })
        };

        Ok(GpuBackend {
            device, queue,
            num_neurons,
            num_workgroups_neurons: ((num_neurons as f32) / 64.0).ceil() as u32,
            membrane_potential_buf, recovery_variable_buf,
            refractory_counter_buf, last_spike_time_buf, spike_count_buf,
            input_current_buf, params_buf, is_output_buf, spiked_buf,
            adjacency_ptr_buf, adjacency_indices_buf, synapse_weights_buf,
            atomic_current_buf,
            uniform_buf,
            lif_bind_group, uniform_bind_group, spmv_bind_group, convert_bind_group,
            lif_pipeline, izhikevich_pipeline, hh_pipeline,
            spmv_pipeline, convert_pipeline,
            active_model: GpuModel::Lif,
            staging_buf, staging_spiked_offset, staging_state_offset, staging_size,
            staging_current_buf, staging_current_size,
        })
    }

    // ── Upload helpers ──

    pub fn upload_initial_state(
        &mut self,
        membrane_potential: &[f32],
        recovery_variable: &[f32],
        refractory_counter: &[i32],
        last_spike_time: &[f32],
        spike_count: &[u32],
        params: &[[f32; 6]],
        is_output: &[u32],
    ) {
        self.write_buf(&self.membrane_potential_buf, bytemuck::cast_slice(membrane_potential));
        self.write_buf(&self.recovery_variable_buf, bytemuck::cast_slice(recovery_variable));
        self.write_buf(&self.refractory_counter_buf, bytemuck::cast_slice(refractory_counter));
        self.write_buf(&self.last_spike_time_buf, bytemuck::cast_slice(last_spike_time));
        self.write_buf(&self.spike_count_buf, bytemuck::cast_slice(spike_count));
        let params_flat: Vec<f32> = params.iter().flat_map(|p| p.iter().copied()).collect();
        self.write_buf(&self.params_buf, bytemuck::cast_slice(&params_flat));
        self.write_buf(&self.is_output_buf, bytemuck::cast_slice(is_output));
    }

    pub fn upload_csr(
        &mut self,
        adjacency_ptr: &[u32],
        adjacency_indices: &[u32],
        synapse_weights: &[f32],
    ) {
        self.write_buf(&self.adjacency_ptr_buf, bytemuck::cast_slice(adjacency_ptr));
        self.write_buf(&self.adjacency_indices_buf, bytemuck::cast_slice(adjacency_indices));
        self.write_buf(&self.synapse_weights_buf, bytemuck::cast_slice(synapse_weights));
    }

    pub fn upload_input_current(&mut self, input_current: &[f32]) {
        self.write_buf(&self.input_current_buf, bytemuck::cast_slice(input_current));
    }

    pub fn write_uniforms(&mut self, dt: f32, sim_time: f32) {
        let data: [f32; 2] = [dt, sim_time];
        self.write_buf(&self.uniform_buf, bytemuck::cast_slice(&data));
    }

    fn write_buf(&self, buf: &wgpu::Buffer, data: &[u8]) {
        self.queue.write_buffer(buf, 0, data);
    }

    // ── Simulation steps ──

    /// Switch the active neuron model for the GPU backend.
    pub fn set_model(&mut self, model: GpuModel) {
        self.active_model = model;
    }

    /// Run the neuron compute shader for the active model.
    /// Returns (spiked, membrane_potential, recovery_variable, refractory_counter, spike_count).
    pub fn step_neurons(
        &mut self,
    ) -> (
        Vec<u32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<u32>,
    ) {
        match self.active_model {
            GpuModel::Lif => self.dispatch_lif(),
            GpuModel::Izhikevich => self.dispatch_izhikevich(),
            GpuModel::HodgkinHuxley => self.dispatch_hh(),
        }
    }

    fn dispatch_pipeline(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        lif_bind_group: &wgpu::BindGroup,
        uniform_bind_group: &wgpu::BindGroup,
        spiked_buf: &wgpu::Buffer,
        membrane_potential_buf: &wgpu::Buffer,
        recovery_variable_buf: &wgpu::Buffer,
        refractory_counter_buf: &wgpu::Buffer,
        spike_count_buf: &wgpu::Buffer,
        staging_buf: &wgpu::Buffer,
        staging_size: u64,
        staging_spiked_offset: u64,
        staging_state_offset: u64,
        num_workgroups_neurons: u32,
        num_neurons: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<u32>) {
        let n = num_neurons;
        let n_bytes = (n as u64) * 4;

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("neuron_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("neuron_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, lif_bind_group, &[]);
            cpass.set_bind_group(1, uniform_bind_group, &[]);
            cpass.dispatch_workgroups(num_workgroups_neurons, 1, 1);
        }

        encoder.copy_buffer_to_buffer(spiked_buf, 0, staging_buf, staging_spiked_offset, n_bytes);
        encoder.copy_buffer_to_buffer(membrane_potential_buf, 0, staging_buf, staging_state_offset, n_bytes);
        encoder.copy_buffer_to_buffer(recovery_variable_buf, 0, staging_buf, staging_state_offset + n_bytes, n_bytes);
        encoder.copy_buffer_to_buffer(refractory_counter_buf, 0, staging_buf, staging_state_offset + n_bytes * 2, n_bytes);
        encoder.copy_buffer_to_buffer(spike_count_buf, 0, staging_buf, staging_state_offset + n_bytes * 3, n_bytes);

        queue.submit(Some(encoder.finish()));
        Self::map_readback_impl(device, staging_buf, staging_size, staging_spiked_offset, staging_state_offset, n)
    }

    fn dispatch_lif(
        &mut self,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<u32>) {
        Self::dispatch_pipeline(
            &self.device, &self.queue, &self.lif_pipeline,
            &self.lif_bind_group, &self.uniform_bind_group,
            &self.spiked_buf, &self.membrane_potential_buf,
            &self.recovery_variable_buf, &self.refractory_counter_buf,
            &self.spike_count_buf, &self.staging_buf,
            self.staging_size,
            self.staging_spiked_offset, self.staging_state_offset,
            self.num_workgroups_neurons, self.num_neurons,
        )
    }

    fn dispatch_izhikevich(
        &mut self,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<u32>) {
        Self::dispatch_pipeline(
            &self.device, &self.queue, &self.izhikevich_pipeline,
            &self.lif_bind_group, &self.uniform_bind_group,
            &self.spiked_buf, &self.membrane_potential_buf,
            &self.recovery_variable_buf, &self.refractory_counter_buf,
            &self.spike_count_buf, &self.staging_buf,
            self.staging_size,
            self.staging_spiked_offset, self.staging_state_offset,
            self.num_workgroups_neurons, self.num_neurons,
        )
    }

    fn dispatch_hh(
        &mut self,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<u32>) {
        // HH pipeline also needs gate buffers uploaded — for now fall back to LIF dispatch
        eprintln!("Warning: HH GPU shader requires gate buffers not yet allocated; falling back to LIF");
        Self::dispatch_pipeline(
            &self.device, &self.queue, &self.lif_pipeline,
            &self.lif_bind_group, &self.uniform_bind_group,
            &self.spiked_buf, &self.membrane_potential_buf,
            &self.recovery_variable_buf, &self.refractory_counter_buf,
            &self.spike_count_buf, &self.staging_buf,
            self.staging_size,
            self.staging_spiked_offset, self.staging_state_offset,
            self.num_workgroups_neurons, self.num_neurons,
        )
    }

    /// Run the LIF compute shader (backward-compatible alias).
    pub fn step_lif(
        &mut self,
    ) -> (
        Vec<u32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<u32>,
    ) {
        let n = self.num_neurons;
        let n_bytes = (n as u64) * 4;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lif_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lif_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.lif_pipeline);
            cpass.set_bind_group(0, &self.lif_bind_group, &[]);
            cpass.set_bind_group(1, &self.uniform_bind_group, &[]);
            cpass.dispatch_workgroups(self.num_workgroups_neurons, 1, 1);
        }

        // Copy results to staging
        encoder.copy_buffer_to_buffer(&self.spiked_buf, 0, &self.staging_buf, self.staging_spiked_offset, n_bytes);
        encoder.copy_buffer_to_buffer(&self.membrane_potential_buf, 0, &self.staging_buf, self.staging_state_offset, n_bytes);
        encoder.copy_buffer_to_buffer(&self.recovery_variable_buf, 0, &self.staging_buf, self.staging_state_offset + n_bytes, n_bytes);
        encoder.copy_buffer_to_buffer(&self.refractory_counter_buf, 0, &self.staging_buf, self.staging_state_offset + n_bytes * 2, n_bytes);
        encoder.copy_buffer_to_buffer(&self.spike_count_buf, 0, &self.staging_buf, self.staging_state_offset + n_bytes * 3, n_bytes);

        self.queue.submit(Some(encoder.finish()));
        self.map_readback(n)
    }

    /// Run the SpMV compute shader (synapse propagation on GPU).
    ///
    /// Adds contributions from spiked neurons' outgoing synapses to
    /// atomic_current, then converts/accumulates into input_current
    /// and resets atomic_current to zero.
    ///
    /// Returns the updated input_current vector for CPU-side sync.
    pub fn step_spmv(&mut self) -> Vec<f32> {
        let n = self.num_neurons;
        let n_bytes = (n as u64) * 4;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spmv_encoder"),
            });

        // Phase 1: Scatter weights from spiked sources to atomic_current
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spmv_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.spmv_pipeline);
            cpass.set_bind_group(0, &self.spmv_bind_group, &[]);
            cpass.dispatch_workgroups(self.num_workgroups_neurons, 1, 1);
        }

        // Phase 2: Convert atomic_current -> f32, add to input_current, reset atomics
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("convert_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.convert_pipeline);
            cpass.set_bind_group(0, &self.convert_bind_group, &[]);
            cpass.dispatch_workgroups(self.num_workgroups_neurons, 1, 1);
        }

        // Phase 3: Copy updated input_current to staging for CPU readback
        encoder.copy_buffer_to_buffer(
            &self.input_current_buf,
            0,
            &self.staging_current_buf,
            0,
            n_bytes,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let slice = self.staging_current_buf.slice(..self.staging_current_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);

        if rx.recv().ok() != Some(Ok(())) {
            return vec![0.0; n];
        }

        let data = slice.get_mapped_range();
        let current: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_current_buf.unmap();
        current
    }

    // ── Readback ──

    fn map_readback(&self, n: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<u32>) {
        Self::map_readback_impl(
            &self.device, &self.staging_buf, self.staging_size,
            self.staging_spiked_offset, self.staging_state_offset, n,
        )
    }

    fn map_readback_impl(
        device: &wgpu::Device,
        staging_buf: &wgpu::Buffer,
        staging_size: u64,
        staging_spiked_offset: u64,
        staging_state_offset: u64,
        n: usize,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<i32>, Vec<u32>) {
        let n_bytes = (n as u64) * 4;
        let slice = staging_buf.slice(..staging_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);

        if rx.recv().ok() != Some(Ok(())) {
            return (
                vec![0u32; n], vec![0.0; n], vec![0.0; n],
                vec![0i32; n], vec![0u32; n],
            );
        }

        let data = slice.get_mapped_range();
        let soff = staging_spiked_offset as usize;
        let moff = staging_state_offset as usize;
        let spiked: Vec<u32> = bytemuck::cast_slice(&data[soff..moff]).to_vec();
        let mem_pot: Vec<f32> = bytemuck::cast_slice(&data[moff..][..n_bytes as usize]).to_vec();
        let rec_var: Vec<f32> = bytemuck::cast_slice(&data[moff + n_bytes as usize..][..n_bytes as usize]).to_vec();
        let refr_ctr: Vec<i32> = bytemuck::cast_slice(&data[moff + n_bytes as usize * 2..][..n_bytes as usize]).to_vec();
        let spike_ct: Vec<u32> = bytemuck::cast_slice(&data[moff + n_bytes as usize * 3..][..n_bytes as usize]).to_vec();
        drop(data);
        staging_buf.unmap();
        (spiked, mem_pot, rec_var, refr_ctr, spike_ct)
    }
}
