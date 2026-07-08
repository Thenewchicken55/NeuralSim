use crate::network::Network;
use crate::simulation::{SimulationEngine, StepResult};
use crate::gui::brain_viz::BrainVisualization;
use eframe::egui;
use rand::{Rng, SeedableRng};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

pub struct NeuralSimApp {
    engine: Arc<parking_lot::Mutex<SimulationEngine>>,
    _running: Arc<AtomicBool>,
    neuron_count: usize,
    grid_cols: usize,
    spike_history: Vec<f64>,
    output_history: Vec<f64>,
    lfp_history: Vec<f64>,
    firing_rate_history: Vec<f64>,
    weight_mean_history: Vec<f64>,
    history_max: usize,
    last_spike_count: u64,
    last_output_count: u64,
    fps_tracker: MovingAverage,
    last_frame: Instant,
    stimulation_strength: f64,
    noise_amplitude: f64,
    last_result: StepResult,
    neuron_flash: Vec<u8>,
    output_flash_counter: u32,
    // View settings
    show_raster: bool,
    show_lfp: bool,
    show_weights: bool,
    #[allow(dead_code)]
    show_fr_histogram: bool,
    show_region_stats: bool,
    use_conductance: bool,
    // Recording stats
    #[allow(dead_code)]
    running_since: Instant,
    // Brain visualization
    brain_viz: BrainVisualization,
    show_brain_viz: bool,
    selected_brain_region: Option<usize>,
}

struct MovingAverage {
    buffer: Vec<f64>,
    idx: usize,
    sum: f64,
}

impl MovingAverage {
    fn new(size: usize) -> Self {
        Self { buffer: vec![0.0; size], idx: 0, sum: 0.0 }
    }
    fn push(&mut self, val: f64) {
        self.sum -= self.buffer[self.idx];
        self.buffer[self.idx] = val;
        self.sum += val;
        self.idx = (self.idx + 1) % self.buffer.len();
    }
    fn average(&self) -> f64 {
        self.sum / self.buffer.len() as f64
    }
}

impl NeuralSimApp {
    pub fn new(network: Network) -> Self {
        Self::from_engine(SimulationEngine::new(network))
    }

    pub fn from_engine(engine: SimulationEngine) -> Self {
        let neuron_count = {
            let net = engine.network.read();
            net.neuron_count()
        };
        let grid_cols = (neuron_count as f64).sqrt().ceil() as usize;
        Self {
            neuron_flash: vec![0u8; neuron_count],
            engine: Arc::new(parking_lot::Mutex::new(engine)),
            _running: Arc::new(AtomicBool::new(false)),
            neuron_count,
            grid_cols,
            spike_history: Vec::with_capacity(200),
            output_history: Vec::with_capacity(200),
            lfp_history: Vec::with_capacity(200),
            firing_rate_history: Vec::with_capacity(200),
            weight_mean_history: Vec::with_capacity(200),
            history_max: 200,
            last_spike_count: 0,
            last_output_count: 0,
            fps_tracker: MovingAverage::new(60),
            last_frame: Instant::now(),
            stimulation_strength: 55.0,
            noise_amplitude: 2.0,
            last_result: StepResult::default(),
            output_flash_counter: 0,
            show_raster: true,
            show_lfp: true,
            show_weights: true,
            show_fr_histogram: true,
            show_region_stats: true,
            use_conductance: true,
            running_since: Instant::now(),
            brain_viz: BrainVisualization::new(),
            show_brain_viz: true,
            selected_brain_region: None,
        }
    }

    pub fn run(network: Network) -> eframe::Result<()> {
        let app = Self::new(network);
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1400.0, 900.0])
                .with_title("NeuralSim"),
            ..Default::default()
        };
        eframe::run_native("NeuralSim", options, Box::new(|_cc| Ok(Box::new(app))))
    }

    pub fn run_with_engine(engine: SimulationEngine) -> eframe::Result<()> {
        let app = Self::from_engine(engine);
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1400.0, 900.0])
                .with_title("NeuralSim"),
            ..Default::default()
        };
        eframe::run_native("NeuralSim", options, Box::new(|_cc| Ok(Box::new(app))))
    }

    fn neuron_color(membrane_v: f64, flash: u8, is_output: bool, region_id: usize) -> egui::Color32 {
        if flash > 0 {
            return egui::Color32::WHITE;
        }
        // Region-based color baseline
        let region_hue = (region_id as f32 * 0.15) % 1.0;
        let normalized = ((membrane_v + 70.0) / 35.0).clamp(0.0, 1.0) as f32;
        let _brightness = (normalized * 0.5 + 0.3).min(1.0);
        let r = (region_hue.sin() * 127.0 + 128.0) as u8;
        let g = ((region_hue + 0.33).sin() * 127.0 + 128.0) as u8;
        let b = ((region_hue + 0.67).sin() * 127.0 + 128.0) as u8;
        // Blend with activity
        let mix = normalized;
        let fr = (r as f32 * (1.0 - mix) + 255.0 * mix) as u8;
        let fg = (g as f32 * (1.0 - mix) + 80.0 * mix) as u8;
        let fb = (b as f32 * (1.0 - mix) + 0.0 * mix) as u8;
        if is_output {
            egui::Color32::from_rgb(fr, (fg + 80).min(255), fb)
        } else {
            egui::Color32::from_rgb(fr, fg, fb)
        }
    }
}

impl eframe::App for NeuralSimApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let frame_start = Instant::now();
        let frame_dt = frame_start.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = frame_start;

        let steps = ((frame_dt * 1000.0 / 0.5) as usize).max(1).min(20);
        let mut total_result = StepResult::default();
        {
            let mut eng = self.engine.lock();
            eng.noise_amplitude = self.noise_amplitude;
            eng.use_conductance = self.use_conductance;
            for _ in 0..steps {
                let result = eng.step();
                for &n in &result.spiking_neurons {
                    self.neuron_flash[n] = 4;
                }
                total_result.spike_count += result.spike_count;
                total_result.output_spike_count += result.output_spike_count;
                total_result.spiking_neurons.extend(&result.spiking_neurons);
                total_result.output_spiking_neurons.extend(&result.output_spiking_neurons);
            }
        }

        for f in self.neuron_flash.iter_mut() {
            *f = f.saturating_sub(1);
        }
        if total_result.output_spike_count > 0 {
            self.output_flash_counter = 10;
        } else {
            self.output_flash_counter = self.output_flash_counter.saturating_sub(1);
        }

        self.last_result = total_result;

        let stats = {
            let eng = self.engine.lock();
            eng.stats()
        };

        let lfp = {
            let eng = self.engine.lock();
            eng.lfp()
        };

        let new_spikes = stats.total_spikes - self.last_spike_count;
        let new_output = stats.output_spikes - self.last_output_count;
        self.last_spike_count = stats.total_spikes;
        self.last_output_count = stats.output_spikes;
        self.spike_history.push(new_spikes as f64);
        self.output_history.push(new_output as f64);
        self.lfp_history.push(lfp);
        self.firing_rate_history.push(stats.mean_firing_rate);
        self.weight_mean_history.push(stats.weight_mean);
        if self.spike_history.len() > self.history_max {
            self.spike_history.remove(0);
            self.output_history.remove(0);
            self.lfp_history.remove(0);
            self.firing_rate_history.remove(0);
            self.weight_mean_history.remove(0);
        }

        let fps = 1.0 / frame_start.elapsed().as_secs_f64().max(1e-6);
        self.fps_tracker.push(fps);

        // Update brain visualization
        {
            let eng = self.engine.lock();
            let net = eng.network.read();
            self.brain_viz.update_from_network(&net);
        }

        // ── Top panel ──
        egui::Panel::top("controls").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🧠 NeuralSim Brain Simulator");
                ui.separator();
                if ui.button("Reset").clicked() {
                    let mut eng = self.engine.lock();
                    *eng = SimulationEngine::new(Network::new(self.neuron_count));
                    self.last_spike_count = 0;
                    self.last_output_count = 0;
                    self.spike_history.clear();
                    self.output_history.clear();
                    self.lfp_history.clear();
                    self.firing_rate_history.clear();
                    self.weight_mean_history.clear();
                    self.neuron_flash.fill(0);
                }
                if ui.button("Burst").clicked() {
                    let mut eng = self.engine.lock();
                    let n = eng.network.read().neuron_count();
                    let mut rng = rand::rngs::StdRng::from_os_rng();
                    for _ in 0..(n / 10).max(1) {
                        let idx = rng.random_range(0..n);
                        eng.stimulate(idx, 50.0);
                    }
                }
                ui.separator();
                ui.label(format!("FPS: {:.0}", self.fps_tracker.average()));
                ui.label(format!("Neurons: {}", self.neuron_count));
                if self.output_flash_counter > 0 {
                    ui.colored_label(egui::Color32::YELLOW, format!("⚡ OUTPUT! x{}", new_output));
                }
                ui.separator();
                ui.checkbox(&mut self.show_brain_viz, "Brain View");
                ui.checkbox(&mut self.show_raster, "Raster");
                ui.checkbox(&mut self.show_lfp, "LFP");
                ui.checkbox(&mut self.show_weights, "Weights");
                ui.checkbox(&mut self.use_conductance, "Cond");
            });
        });

        // ── Central panel ──
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.columns(2, |cols| {
                // ── LEFT COLUMN: Brain Visualization + Controls ──
                cols[0].vertical(|ui| {
                    if self.show_brain_viz {
                        ui.group(|ui| {
                            ui.label("🧠 Brain Region Visualization");
                            ui.separator();
                            
                            let (response, painter) = ui.allocate_painter(
                                egui::Vec2::new(ui.available_width(), 300.0),
                                egui::Sense::click(),
                            );
                            
                            // Handle clicks on brain regions
                            if response.clicked() {
                                if let Some(click_pos) = response.interact_pointer_pos() {
                                    if let Some(region_idx) = self.brain_viz.handle_click(click_pos, response.rect) {
                                        self.selected_brain_region = Some(region_idx);
                                        
                                        // Stimulate the clicked region
                                        if let Some(region) = self.brain_viz.get_region_info(region_idx) {
                                            let region_name = region.name.clone();
                                            let mut eng = self.engine.lock();
                                            let n = eng.network.read().neuron_count();
                                            let mut rng = rand::rngs::StdRng::from_os_rng();
                                            
                                            // Stimulate random neurons in this region
                                            for _ in 0..(n / 20).max(1) {
                                                let idx = rng.random_range(0..n);
                                                eng.stimulate(idx, self.stimulation_strength);
                                            }
                                            
                                            // Update brain viz activity
                                            self.brain_viz.stimulate_region(&region_name, 0.8);
                                        }
                                    }
                                }
                            }
                            
                            // Render the brain visualization
                            let eng = self.engine.lock();
                            let net = eng.network.read();
                            self.brain_viz.render(&painter, response.rect, &net);
                            
                            ui.separator();
                            
                            // Show selected region info
                            if let Some(region_idx) = self.selected_brain_region {
                                self.brain_viz.render_info_panel(ui, region_idx);
                            } else {
                                ui.label("Click on a brain region to explore it");
                            }
                        });
                    }
                    
                    // Controls
                    ui.group(|ui| {
                        ui.label("Controls");
                        ui.separator();
                        ui.add(egui::Slider::new(&mut self.stimulation_strength, 0.0..=100.0)
                            .text("Stim Strength"));
                        ui.add(egui::Slider::new(&mut self.noise_amplitude, 0.0..=30.0)
                            .text("Noise Level"));

                        if ui.button("Stimulate Random 10").clicked() {
                            let mut eng = self.engine.lock();
                            let n = eng.network.read().neuron_count();
                            let mut rng = rand::rngs::StdRng::from_os_rng();
                            for _ in 0..10 {
                                let idx = rng.random_range(0..n);
                                eng.stimulate(idx, self.stimulation_strength);
                            }
                        }
                        
                        if ui.button("Stimulate Visual Cortex").clicked() {
                            self.brain_viz.stimulate_region("Visual Cortex", 0.8);
                        }
                        
                        if ui.button("Stimulate Motor Cortex").clicked() {
                            self.brain_viz.stimulate_region("Motor Cortex", 0.8);
                        }
                    });

                    // Stats
                    ui.group(|ui| {
                        ui.label("Statistics");
                        ui.separator();
                        ui.label(format!("Time: {:.1} ms", stats.sim_time_ms));
                        ui.label(format!("Total spikes: {}", stats.total_spikes));
                        ui.label(format!("Output spikes: {}", stats.output_spikes));
                        ui.label(format!("Spikes/frame: {}", self.last_result.spike_count));
                        let rate = {
                            let eng = self.engine.lock();
                            let n = eng.network.read().neuron_count() as f64;
                            if n > 0.0 && stats.sim_time_ms > 0.0 {
                                (stats.total_spikes as f64 / n) / (stats.sim_time_ms / 1000.0)
                            } else { 0.0 }
                        };
                        ui.label(format!("Avg rate: {:.1} Hz", rate));
                        ui.label(format!("Mean FR: {:.1} Hz", stats.mean_firing_rate));
                        ui.label(format!("Synch: {:.3}", stats.synchrony_index));
                        ui.label(format!("LFP: {:.2}", lfp));
                        ui.label(format!("Weight μ={:.3} σ={:.3}", stats.weight_mean, stats.weight_std));
                        ui.label(format!("Updates: {}", stats.weight_updates));
                    });

                    // Region stats
                    if self.show_region_stats {
                        ui.group(|ui| {
                            ui.label("Region Activity");
                            ui.separator();
                            let eng = self.engine.lock();
                            let net = eng.network.read();
                            for (name, count, spike_c) in net.region_counts() {
                                let t = net.time.max(0.001);
                                let rate = if count > 0 {
                                    spike_c as f64 / count as f64 / (t / 1000.0)
                                } else { 0.0 };
                                ui.label(format!("{}: {} nrn, {:.1} Hz", name, count, rate));
                            }
                        });
                    }

                    // Spike rate graph
                    if self.show_raster {
                        self.draw_graph(ui, "Spike Rate", &self.spike_history, egui::Color32::LIGHT_BLUE, 80.0);
                        self.draw_graph(ui, "Output Rate", &self.output_history, egui::Color32::YELLOW, 50.0);
                    }

                    // LFP graph
                    if self.show_lfp {
                        self.draw_graph(ui, "LFP", &self.lfp_history, egui::Color32::RED, 50.0);
                        self.draw_graph(ui, "Mean FR", &self.firing_rate_history, egui::Color32::GREEN, 50.0);
                    }

                    // Weight distribution
                    if self.show_weights {
                        self.draw_graph(ui, "Weight Mean", &self.weight_mean_history, egui::Color32::MAGENTA, 40.0);
                    }
                });

                // ── RIGHT COLUMN: Neural Activity + Info ──
                cols[1].vertical(|ui| {
                    ui.group(|ui| {
                        ui.label("🔬 Neural Activity Monitor");
                        ui.separator();
                        
                        // Show real-time neural activity
                        ui.label(format!("Active Neurons: {}", self.last_result.spiking_neurons.len()));
                        ui.label(format!("Output Activity: {}", self.last_result.output_spiking_neurons.len()));
                        
                        // Show neural pathway activity
                        ui.label("Active Pathways:");
                        for pathway in &self.brain_viz.pathways {
                            if pathway.activity > 0.1 {
                                let color = if pathway.is_inhibitory {
                                    egui::Color32::RED
                                } else {
                                    egui::Color32::GREEN
                                };
                                ui.colored_label(color, format!("{} → {}: {:.1}%", 
                                    pathway.from_region, pathway.to_region, pathway.activity * 100.0));
                            }
                        }
                    });
                    
                    ui.group(|ui| {
                        ui.label("🎯 Learning Progress");
                        ui.separator();
                        ui.label(format!("Weight Updates: {}", stats.weight_updates));
                        ui.label(format!("Weight Mean: {:.3}", stats.weight_mean));
                        ui.label(format!("Weight Std: {:.3}", stats.weight_std));
                        
                        // Show learning visualization
                        let progress = (stats.weight_mean * 100.0).min(100.0) as f32;
                        ui.add(egui::ProgressBar::new(progress as f32 / 100.0)
                            .text(format!("{:.1}%", progress)));
                        ui.label("Learning Progress");
                    });
                    
                    ui.group(|ui| {
                        ui.label("📊 Network Performance");
                        ui.separator();
                        ui.label(format!("Synchrony: {:.3}", stats.synchrony_index));
                        ui.label(format!("Firing Rate: {:.1} Hz", stats.mean_firing_rate));
                        
                        // Show performance meter
                        let performance = (stats.synchrony_index * 100.0).min(100.0) as f32;
                        ui.add(egui::ProgressBar::new(performance / 100.0)
                            .text(format!("{:.1}%", performance)));
                        ui.label("Network Coordination");
                    });
                    
                    // Interactive buttons for different brain functions
                    ui.group(|ui| {
                        ui.label("🧠 Brain Functions");
                        ui.separator();
                        
                        if ui.button("👁️ Visual Processing").clicked() {
                            self.brain_viz.stimulate_region("Visual Cortex", 0.9);
                            self.brain_viz.stimulate_region("Occipital Lobe", 0.7);
                        }
                        
                        if ui.button("🎵 Auditory Processing").clicked() {
                            self.brain_viz.stimulate_region("Auditory Cortex", 0.9);
                            self.brain_viz.stimulate_region("Temporal Lobe", 0.7);
                        }
                        
                        if ui.button("🏃 Motor Control").clicked() {
                            self.brain_viz.stimulate_region("Motor Cortex", 0.9);
                            self.brain_viz.stimulate_region("Cerebellum", 0.7);
                        }
                        
                        if ui.button("💾 Memory Formation").clicked() {
                            self.brain_viz.stimulate_region("Hippocampus", 0.9);
                            self.brain_viz.stimulate_region("Frontal Lobe", 0.7);
                        }
                        
                        if ui.button("😰 Emotional Response").clicked() {
                            self.brain_viz.stimulate_region("Amygdala", 0.9);
                            self.brain_viz.stimulate_region("Temporal Lobe", 0.7);
                        }
                    });
                });
            });
        });

        ui.ctx().request_repaint();
    }
}

// Helper: draw a line graph
impl NeuralSimApp {
    fn draw_graph(&self, ui: &mut egui::Ui, label: &str, data: &[f64], color: egui::Color32, height: f32) {
        ui.group(|ui| {
            ui.label(label);
            ui.separator();
            let (response, painter) = ui.allocate_painter(
                egui::Vec2::new(ui.available_width(), height),
                egui::Sense::hover(),
            );
            if data.len() >= 2 {
                let rect = response.rect;
                let max_val = data.iter().cloned().fold(0.0f64, f64::max).max(1.0);
                let w = rect.width(); let h = rect.height();

                let points: Vec<egui::Pos2> = data.iter().enumerate().map(|(i, v)| {
                    let x = rect.left() + (i as f32 / (data.len() - 1) as f32) * w;
                    let y = rect.bottom() - (*v as f32 / max_val as f32) * h;
                    egui::Pos2::new(x, y)
                }).collect();
                if points.len() > 1 {
                    painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, color)));
                }
                // Show current value
                if let Some(&last) = data.last() {
                    let text = format!("{:.2}", last);
                    painter.text(egui::Pos2::new(rect.right() - 40.0, rect.top() + 10.0),
                                 egui::Align2::RIGHT_TOP, text,
                                 egui::TextStyle::Monospace.resolve(ui.style()),
                                 color);
                }
            }
        });
    }
}
