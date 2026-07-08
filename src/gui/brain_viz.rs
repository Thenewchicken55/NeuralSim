use eframe::egui::{Color32, Pos2, Rect, Vec2, Painter, Stroke, StrokeKind, FontId, Align2, Ui};
use crate::network::Network;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct BrainRegion {
    pub name: String,
    pub description: String,
    pub center: Pos2,
    pub radius: f32,
    pub color: Color32,
    pub neuron_count: usize,
    pub activity: f64,
    pub connections: Vec<String>,
    pub is_active: bool,
}

#[derive(Clone, Debug)]
pub struct NeuralPathway {
    pub from_region: String,
    pub to_region: String,
    pub strength: f64,
    pub activity: f64,
    pub is_inhibitory: bool,
}

pub struct BrainVisualization {
    pub regions: Vec<BrainRegion>,
    pub pathways: Vec<NeuralPathway>,
    pub selected_region: Option<usize>,
    pub show_labels: bool,
    pub show_activity: bool,
    pub show_pathways: bool,
    pub brain_scale: f32,
    pub brain_offset: Vec2,
    pub last_update: Instant,
    pub activity_decay: f64,
}

impl BrainVisualization {
    pub fn new() -> Self {
        let mut viz = Self {
            regions: Vec::new(),
            pathways: Vec::new(),
            selected_region: None,
            show_labels: true,
            show_activity: true,
            show_pathways: true,
            brain_scale: 1.0,
            brain_offset: Vec2::ZERO,
            last_update: Instant::now(),
            activity_decay: 0.95,
        };
        viz.initialize_brain_regions();
        viz.initialize_pathways();
        viz
    }

    fn initialize_brain_regions(&mut self) {
        // Create a simplified brain map with major regions
        self.regions = vec![
            BrainRegion {
                name: "Visual Cortex".to_string(),
                description: "Processes visual information from eyes".to_string(),
                center: Pos2::new(0.7, 0.3),
                radius: 0.15,
                color: Color32::from_rgb(100, 150, 255),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Frontal Lobe".to_string(), "Temporal Lobe".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Auditory Cortex".to_string(),
                description: "Processes sound and speech information".to_string(),
                center: Pos2::new(0.3, 0.4),
                radius: 0.12,
                color: Color32::from_rgb(150, 100, 255),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Frontal Lobe".to_string(), "Temporal Lobe".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Motor Cortex".to_string(),
                description: "Controls voluntary movements".to_string(),
                center: Pos2::new(0.5, 0.2),
                radius: 0.13,
                color: Color32::from_rgb(255, 100, 100),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Cerebellum".to_string(), "Frontal Lobe".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Frontal Lobe".to_string(),
                description: "Executive functions, decision making, planning".to_string(),
                center: Pos2::new(0.5, 0.15),
                radius: 0.18,
                color: Color32::from_rgb(255, 150, 50),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Hippocampus".to_string(), "Amygdala".to_string(), "Parietal Lobe".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Parietal Lobe".to_string(),
                description: "Processes sensory information and spatial awareness".to_string(),
                center: Pos2::new(0.5, 0.35),
                radius: 0.14,
                color: Color32::from_rgb(100, 200, 100),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Frontal Lobe".to_string(), "Occipital Lobe".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Temporal Lobe".to_string(),
                description: "Memory, language, and auditory processing".to_string(),
                center: Pos2::new(0.3, 0.5),
                radius: 0.16,
                color: Color32::from_rgb(200, 200, 100),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Hippocampus".to_string(), "Amygdala".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Occipital Lobe".to_string(),
                description: "Visual processing center".to_string(),
                center: Pos2::new(0.7, 0.5),
                radius: 0.12,
                color: Color32::from_rgb(100, 200, 200),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Visual Cortex".to_string(), "Parietal Lobe".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Hippocampus".to_string(),
                description: "Memory formation and spatial navigation".to_string(),
                center: Pos2::new(0.4, 0.6),
                radius: 0.08,
                color: Color32::from_rgb(200, 150, 100),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Frontal Lobe".to_string(), "Temporal Lobe".to_string(), "Amygdala".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Amygdala".to_string(),
                description: "Emotional processing and fear conditioning".to_string(),
                center: Pos2::new(0.5, 0.65),
                radius: 0.07,
                color: Color32::from_rgb(255, 100, 150),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Hippocampus".to_string(), "Frontal Lobe".to_string(), "Temporal Lobe".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Cerebellum".to_string(),
                description: "Motor coordination and learning".to_string(),
                center: Pos2::new(0.6, 0.75),
                radius: 0.14,
                color: Color32::from_rgb(150, 200, 150),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Motor Cortex".to_string(), "Brain Stem".to_string()],
                is_active: false,
            },
            BrainRegion {
                name: "Brain Stem".to_string(),
                description: "Vital functions and arousal".to_string(),
                center: Pos2::new(0.5, 0.85),
                radius: 0.1,
                color: Color32::from_rgb(180, 180, 180),
                neuron_count: 0,
                activity: 0.0,
                connections: vec!["Spinal Cord".to_string()],
                is_active: false,
            },
        ];
    }

    fn initialize_pathways(&mut self) {
        // Create connections between regions
        self.pathways = vec![
            NeuralPathway {
                from_region: "Visual Cortex".to_string(),
                to_region: "Occipital Lobe".to_string(),
                strength: 0.8,
                activity: 0.0,
                is_inhibitory: false,
            },
            NeuralPathway {
                from_region: "Auditory Cortex".to_string(),
                to_region: "Temporal Lobe".to_string(),
                strength: 0.7,
                activity: 0.0,
                is_inhibitory: false,
            },
            NeuralPathway {
                from_region: "Motor Cortex".to_string(),
                to_region: "Cerebellum".to_string(),
                strength: 0.9,
                activity: 0.0,
                is_inhibitory: false,
            },
            NeuralPathway {
                from_region: "Frontal Lobe".to_string(),
                to_region: "Hippocampus".to_string(),
                strength: 0.6,
                activity: 0.0,
                is_inhibitory: false,
            },
            NeuralPathway {
                from_region: "Hippocampus".to_string(),
                to_region: "Amygdala".to_string(),
                strength: 0.5,
                activity: 0.0,
                is_inhibitory: false,
            },
            NeuralPathway {
                from_region: "Amygdala".to_string(),
                to_region: "Frontal Lobe".to_string(),
                strength: 0.4,
                activity: 0.0,
                is_inhibitory: true, // Inhibitory connection
            },
            NeuralPathway {
                from_region: "Cerebellum".to_string(),
                to_region: "Motor Cortex".to_string(),
                strength: 0.7,
                activity: 0.0,
                is_inhibitory: false,
            },
        ];
    }

    pub fn update_from_network(&mut self, network: &Network) {
        let now = Instant::now();
        let _dt = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        // Update neuron counts and activity from network
        let total_neurons = network.neuron_count();
        if total_neurons == 0 {
            return;
        }

        // Distribute neurons across regions (simplified mapping)
        let neurons_per_region = total_neurons / self.regions.len();
        
        for (i, region) in self.regions.iter_mut().enumerate() {
            region.neuron_count = neurons_per_region;
            
            // Calculate activity based on region characteristics
            let base_activity = (i as f64 * 0.1).sin().abs();
            let noise = (now.elapsed().as_millis() as f64 * 0.001).sin() * 0.1;
            region.activity = (base_activity + noise).clamp(0.0, 1.0);
            
            // Decay activity
            region.activity *= self.activity_decay;
        }

        // Update pathway activity
        for pathway in &mut self.pathways {
            pathway.activity = (pathway.activity * 0.9 + pathway.strength * 0.1).clamp(0.0, 1.0);
        }
    }

    pub fn stimulate_region(&mut self, region_name: &str, strength: f64) {
        for region in &mut self.regions {
            if region.name == region_name {
                region.activity = (region.activity + strength).min(1.0);
                region.is_active = true;
                
                // Activate connected pathways
                for pathway in &mut self.pathways {
                    if pathway.from_region == region_name || pathway.to_region == region_name {
                        pathway.activity = (pathway.activity + strength * 0.5).min(1.0);
                    }
                }
                break;
            }
        }
    }

    pub fn render(&self, painter: &Painter, rect: Rect, network: &Network) {
        let center = rect.center();
        let scale = rect.width().min(rect.height()) * 0.4 * self.brain_scale;

        // Draw brain outline (simplified oval shape)
        let brain_rect = Rect::from_center_size(
            center + self.brain_offset,
            Vec2::new(scale * 2.0, scale * 1.8),
        );
        painter.rect_filled(
            brain_rect,
            20.0,
            Color32::from_rgba_unmultiplied(40, 40, 50, 200),
        );
        painter.rect_stroke(
            brain_rect,
            20.0,
            Stroke::new(2.0, Color32::from_rgb(100, 100, 120)),
            StrokeKind::Inside,
        );

        // Draw pathways first (behind regions)
        if self.show_pathways {
            self.render_pathways(painter, center, scale, network);
        }

        // Draw regions
        for (i, region) in self.regions.iter().enumerate() {
            self.render_region(painter, center, scale, region, i, network);
        }

        // Draw labels
        if self.show_labels {
            self.render_labels(painter, center, scale);
        }
    }

    fn render_region(
        &self,
        painter: &Painter,
        center: Pos2,
        scale: f32,
        region: &BrainRegion,
        index: usize,
        _network: &Network,
    ) {
        let region_center = center + Vec2::new(
            (region.center.x - 0.5) * scale * 2.0,
            (region.center.y - 0.5) * scale * 2.0,
        );
        let region_radius = region.radius * scale;

        // Region background with activity-based color intensity
        let activity_color = Color32::from_rgba_unmultiplied(
            (region.color.r() as f64 * (0.3 + region.activity * 0.7)) as u8,
            (region.color.g() as f64 * (0.3 + region.activity * 0.7)) as u8,
            (region.color.b() as f64 * (0.3 + region.activity * 0.7)) as u8,
            150,
        );

        // Draw region circle
        painter.circle_filled(
            region_center,
            region_radius,
            activity_color,
        );

        // Draw region border
        let border_color = if region.is_active {
            Color32::from_rgb(255, 255, 255)
        } else {
            Color32::from_rgb(80, 80, 100)
        };
        painter.circle_stroke(
            region_center,
            region_radius,
            Stroke::new(2.0, border_color),
        );

        // Draw activity indicator (pulsing effect when active)
        if region.activity > 0.1 {
            let pulse_radius = region_radius * (1.0 + region.activity as f32 * 0.3);
            let pulse_color = Color32::from_rgba_unmultiplied(
                region.color.r(),
                region.color.g(),
                region.color.b(),
                (region.activity * 100.0) as u8,
            );
            painter.circle_stroke(
                region_center,
                pulse_radius,
                Stroke::new(1.0, pulse_color),
            );
        }

        // Draw neuron count if region is selected
        if Some(index) == self.selected_region {
            let text = format!("{}", region.neuron_count);
            painter.text(
                region_center,
                Align2::CENTER_CENTER,
                text,
                FontId::proportional(12.0),
                Color32::WHITE,
            );
        }
    }

    fn render_pathways(
        &self,
        painter: &Painter,
        center: Pos2,
        scale: f32,
        _network: &Network,
    ) {
        for pathway in &self.pathways {
            if let (Some(from_region), Some(to_region)) = (
                self.regions.iter().find(|r| r.name == pathway.from_region),
                self.regions.iter().find(|r| r.name == pathway.to_region),
            ) {
                let from_center = center + Vec2::new(
                    (from_region.center.x - 0.5) * scale * 2.0,
                    (from_region.center.y - 0.5) * scale * 2.0,
                );
                let to_center = center + Vec2::new(
                    (to_region.center.x - 0.5) * scale * 2.0,
                    (to_region.center.y - 0.5) * scale * 2.0,
                );

                let color = if pathway.is_inhibitory {
                    Color32::from_rgba_unmultiplied(255, 100, 100, (pathway.activity * 200.0) as u8)
                } else {
                    Color32::from_rgba_unmultiplied(100, 255, 100, (pathway.activity * 200.0) as u8)
                };

                let stroke_width = (pathway.strength * 3.0) as f32;
                painter.line_segment(
                    [from_center, to_center],
                    Stroke::new(stroke_width, color),
                );

                // Draw arrow head for direction
                let direction = (to_center - from_center).normalized();
                let arrow_size = 8.0;
                let arrow_tip = to_center - direction * from_region.radius * scale;
                let arrow_left = arrow_tip + Vec2::new(-direction.y, direction.x) * arrow_size * 0.5;
                let arrow_right = arrow_tip + Vec2::new(direction.y, -direction.x) * arrow_size * 0.5;
                
                painter.line_segment(
                    [arrow_left, arrow_tip],
                    Stroke::new(2.0, color),
                );
                painter.line_segment(
                    [arrow_right, arrow_tip],
                    Stroke::new(2.0, color),
                );
            }
        }
    }

    fn render_labels(&self, painter: &Painter, center: Pos2, scale: f32) {
        for region in &self.regions {
            let region_center = center + Vec2::new(
                (region.center.x - 0.5) * scale * 2.0,
                (region.center.y - 0.5) * scale * 2.0,
            );
            let label_offset = Vec2::new(0.0, -region.radius * scale - 10.0);
            
            painter.text(
                region_center + label_offset,
                Align2::CENTER_BOTTOM,
                &region.name,
                FontId::proportional(10.0),
                Color32::from_rgb(200, 200, 220),
            );
        }
    }

    pub fn handle_click(&mut self, click_pos: Pos2, rect: Rect) -> Option<usize> {
        let center = rect.center();
        let scale = rect.width().min(rect.height()) * 0.4 * self.brain_scale;

        for (i, region) in self.regions.iter().enumerate() {
            let region_center = center + Vec2::new(
                (region.center.x - 0.5) * scale * 2.0,
                (region.center.y - 0.5) * scale * 2.0,
            );
            let region_radius = region.radius * scale;

            let distance = (click_pos - region_center).length();
            if distance <= region_radius {
                self.selected_region = Some(i);
                return Some(i);
            }
        }

        self.selected_region = None;
        None
    }

    pub fn get_region_info(&self, index: usize) -> Option<&BrainRegion> {
        self.regions.get(index)
    }

    pub fn render_info_panel(&self, ui: &mut Ui, index: usize) {
        if let Some(region) = self.regions.get(index) {
            ui.group(|ui| {
                ui.heading(&region.name);
                ui.label(&region.description);
                ui.separator();
                ui.label(format!("Neurons: {}", region.neuron_count));
                ui.label(format!("Activity: {:.1}%", region.activity * 100.0));
                ui.label(format!("Connections: {}", region.connections.len()));
                
                if !region.connections.is_empty() {
                    ui.label("Connected to:");
                    for conn in &region.connections {
                        ui.label(format!("  • {}", conn));
                    }
                }
            });
        }
    }
}