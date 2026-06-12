use crate::network::Network;
use crate::simulation::SimulationEngine;
use eframe::egui;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

pub struct NeuralSimApp {
    engine: Arc<parking_lot::Mutex<SimulationEngine>>,
    #[allow(dead_code)]
    running: Arc<AtomicBool>,
    neuron_count: usize,
    spike_history: Vec<f64>,
    history_max: usize,
    last_spike_count: u64,
    fps_tracker: MovingAverage,
    last_frame: Instant,
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
        let neuron_count = network.neuron_count();
        let engine = SimulationEngine::new(network);
        Self {
            engine: Arc::new(parking_lot::Mutex::new(engine)),
            running: Arc::new(AtomicBool::new(false)),
            neuron_count,
            spike_history: Vec::with_capacity(200),
            history_max: 200,
            last_spike_count: 0,
            fps_tracker: MovingAverage::new(60),
            last_frame: Instant::now(),
        }
    }

    pub fn run(network: Network) -> eframe::Result<()> {
        let app = Self::new(network);
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1200.0, 800.0])
                .with_title("NeuralSim"),
            ..Default::default()
        };
        eframe::run_native("NeuralSim", options, Box::new(|_cc| Ok(Box::new(app))))
    }
}

impl eframe::App for NeuralSimApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let frame_start = Instant::now();
        let frame_dt = frame_start.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = frame_start;

        // Run simulation steps proportional to real time
        let steps = (frame_dt * 1000.0 / 0.5) as usize; // match engine dt
        {
            let mut eng = self.engine.lock();
            for _ in 0..steps.min(10) {
                eng.step();
            }
        }

        let stats = {
            let eng = self.engine.lock();
            eng.stats()
        };

        let new_spikes = stats.total_spikes - self.last_spike_count;
        self.last_spike_count = stats.total_spikes;
        self.spike_history.push(new_spikes as f64);
        if self.spike_history.len() > self.history_max {
            self.spike_history.remove(0);
        }

        let fps = 1.0 / frame_start.elapsed().as_secs_f64().max(1e-6);
        self.fps_tracker.push(fps);

        egui::Panel::top("controls").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("NeuralSim");
                ui.separator();
                if ui.button("Reset").clicked() {
                    let mut eng = self.engine.lock();
                    *eng = SimulationEngine::new(Network::new(self.neuron_count));
                    self.last_spike_count = 0;
                    self.spike_history.clear();
                }
                ui.label(format!(
                    "Neurons: {}  |  FPS: {:.0}",
                    self.neuron_count,
                    self.fps_tracker.average()
                ));
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.columns(2, |cols| {
                cols[0].vertical(|ui| {
                    ui.group(|ui| {
                        ui.label("Simulation Stats");
                        ui.separator();
                        ui.label(format!("Total spikes: {}", stats.total_spikes));
                        ui.label(format!("Peak active/step: {}", stats.active_neurons));
                        ui.label(format!("Sim time: {:.1} ms", stats.sim_time_ms));
                        let rate = {
                            let eng = self.engine.lock();
                            let n = eng.network.read().neuron_count() as f64;
                            if n > 0.0 && stats.sim_time_ms > 0.0 {
                                (stats.total_spikes as f64 / n) / (stats.sim_time_ms / 1000.0)
                            } else { 0.0 }
                        };
                        ui.label(format!("Avg firing rate: {:.2} Hz", rate));
                    });

                    ui.group(|ui| {
                        ui.label("Spike Rate (spikes/step)");
                        ui.separator();
                        let graph_height = 120.0;
                        let (response, painter) = ui.allocate_painter(
                            egui::Vec2::new(ui.available_width(), graph_height),
                            egui::Sense::hover(),
                        );
                        if self.spike_history.len() >= 2 {
                            let rect = response.rect;
                            let max_val = self.spike_history.iter().cloned().fold(0.0f64, f64::max).max(1.0);
                            let w = rect.width();
                            let h = rect.height();

                            painter.add(egui::Shape::line(
                                vec![
                                    egui::Pos2::new(rect.left(), rect.bottom()),
                                    egui::Pos2::new(rect.right(), rect.bottom()),
                                ],
                                egui::Stroke::new(1.0, egui::Color32::DARK_GRAY),
                            ));
                            painter.add(egui::Shape::line(
                                vec![
                                    egui::Pos2::new(rect.left(), rect.top()),
                                    egui::Pos2::new(rect.left(), rect.bottom()),
                                ],
                                egui::Stroke::new(1.0, egui::Color32::DARK_GRAY),
                            ));

                            let points: Vec<egui::Pos2> = self.spike_history
                                .iter()
                                .enumerate()
                                .map(|(i, v)| {
                                    let x = rect.left() + (i as f32 / (self.spike_history.len() - 1) as f32) * w;
                                    let y = rect.bottom() - (*v as f32 / max_val as f32) * h;
                                    egui::Pos2::new(x, y)
                                })
                                .collect();
                            if points.len() > 1 {
                                painter.add(egui::Shape::line(points, egui::Stroke::new(2.0, egui::Color32::LIGHT_BLUE)));
                            }
                        }
                    });
                });

                cols[1].vertical(|ui| {
                    ui.group(|ui| {
                        ui.label("Network Activity");
                        ui.separator();
                        let eng = self.engine.lock();
                        let net = eng.network.read();
                        let n = net.neuron_count();
                        let grid_cols = (n as f64).sqrt().ceil() as usize;
                        let grid_size = grid_cols.max(1);
                        let cell_size = (ui.available_width() / grid_size as f32).min(6.0).max(2.0);

                        let (response, painter) = ui.allocate_painter(
                            egui::Vec2::new(cell_size * grid_size as f32, cell_size * grid_size as f32),
                            egui::Sense::hover(),
                        );

                        let to_screen = egui::emath::RectTransform::from_to(
                            egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::Vec2::new(grid_size as f32, grid_size as f32),
                            ),
                            response.rect,
                        );

                        let display_n = n.min(grid_size * grid_size);
                        for i in 0..display_n {
                            let x = (i % grid_size) as f32;
                            let y = (i / grid_size) as f32;
                            let v = net.neurons.membrane_potential[i];
                            let normalized = ((v + 70.0) / 30.0).clamp(0.0, 1.0) as f32;
                            let color = egui::Color32::from_rgb(
                                (normalized * 255.0) as u8,
                                (normalized * 100.0) as u8,
                                (255.0 - normalized * 200.0) as u8,
                            );
                            let pos = to_screen * egui::Pos2::new(x + 0.5, y + 0.5);
                            painter.circle_filled(pos, cell_size * 0.4, color);
                        }
                    });
                });
            });
        });

        ui.ctx().request_repaint();
    }
}
