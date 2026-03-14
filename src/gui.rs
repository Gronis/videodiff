use crossbeam_channel::{unbounded, Receiver};
use eframe::egui;
use rfd::FileDialog;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Default)]
struct VideoDiffGui {
    ref_file: Option<String>,
    target_file: Option<String>,

    skip_start: f32,
    skip_end: f32,
    skip_tempo: bool,

    running: bool,
    progress: f32,

    log: Vec<String>,
    result: Option<String>,

    error: Option<String>,

    receiver: Option<Receiver<Message>>,

    child: Arc<Mutex<Option<Child>>>,
}

enum Message {
    Log(String),
    Result(String),
    Error(String),
    Finished,
}

impl VideoDiffGui {
    fn push_log(&mut self, text: String) {
        self.log.push(text.clone());

        // crude progress estimation from CLI output
        if text.contains("Extracting frames") {
            self.progress = 0.1;
        } else if text.contains("Indexing frames") {
            self.progress = 0.3;
        } else if text.contains("Comparing frames") {
            self.progress = 0.6;
        } else if text.contains("Optimizing offset") {
            self.progress = 0.8;
        } else if text.contains("[Result]") {
            self.progress = 1.0;
            self.result = Some(text);
        }
    }

    fn start_analysis(&mut self) {

        let Some(ref_file) = self.ref_file.clone() else {
            self.error = Some("Reference file not selected".into());
            return;
        };

        let Some(target_file) = self.target_file.clone() else {
            self.error = Some("Target file not selected".into());
            return;
        };

        let skip_start = self.skip_start;
        let skip_end = self.skip_end;
        let skip_tempo = self.skip_tempo;

        let (tx, rx) = unbounded();
        self.receiver = Some(rx);
        self.running = true;
        self.progress = 0.0;
        self.log.clear();
        self.result = None;
        let child_handle = self.child.clone();

        thread::spawn(move || {

            let mut cmd = Command::new("./videodiff");

            cmd.arg(ref_file)
                .arg(target_file)
                .arg("--skip-start")
                .arg(skip_start.to_string())
                .arg("--skip-end")
                .arg(skip_end.to_string());

            if skip_tempo {
                cmd.arg("--skip-tempo");
            }

            cmd.stderr(Stdio::piped());

            let mut child = match cmd.spawn() {
                Ok(child) => child,
                Err(e) => {
                    let _ = tx.send(Message::Error(format!(
                        "Failed to launch videodiff: {e}"
                    )));
                    return;
                }
            };

            let stderr = child.stderr.take();

            // store child so GUI can kill it
            if let Ok(mut slot) = child_handle.lock() {
                *slot = Some(child);
            }

            let stderr = match stderr {
                Some(s) => s,
                None => {
                    let _ = tx.send(Message::Error("Failed to read stderr".into()));
                    return;
                }
            };
            let reader = BufReader::new(stderr);

            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        if text.contains("[Result]") {
                            let _ = tx.send(Message::Result(text.clone()));
                        }

                        let _ = tx.send(Message::Log(text));
                    }
                    Err(e) => {
                        let _ = tx.send(Message::Error(format!(
                            "Error reading output: {e}"
                        )));
                    }
                }
            }

            let _ = tx.send(Message::Finished);
        });
    }
}

impl eframe::App for VideoDiffGui {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Ok(mut child) = self.child.lock() {
            if let Some(child) = child.as_mut() {
                let _ = child.kill();
            }
        }
    }

    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if let Some(rx) = &self.receiver.clone() {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Message::Log(text) => self.push_log(text),
                    Message::Result(text) => self.result = Some(text),
                    Message::Error(err) => {
                        self.error = Some(err);
                        self.running = false;
                    }
                    Message::Finished => {
                        self.running = false;
                        self.progress = 1.0;
                        if let Ok(mut slot) = self.child.lock() {
                            *slot = None;
                        }
                    }
                }
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("VideoDiff GUI");

            ui.separator();

            // reference video
            ui.horizontal(|ui| {
                ui.label("Reference video");
                if ui.button("Browse...").clicked() {
                    if let Some(file) = FileDialog::new().pick_file() {
                        self.ref_file = Some(file.display().to_string());
                    }
                }

                let mut no_file = "".to_string();
                let file = self.ref_file.as_mut().unwrap_or(&mut no_file);
                ui.add(
                    egui::TextEdit::singleline(file)
                        .desired_width(f32::INFINITY)
                        .hint_text("Click to browse or paste path here…")
                );
            });

            // target video
            ui.horizontal(|ui| {
                ui.label("Target video");
                if ui.button("Browse...").clicked() {
                    if let Some(file) = FileDialog::new().pick_file() {
                        self.target_file = Some(file.display().to_string());
                    }
                }

                let mut no_file = "".to_string();
                let file = self.target_file.as_mut().unwrap_or(&mut no_file);
                ui.add(
                    egui::TextEdit::singleline(file)
                        .desired_width(f32::INFINITY)
                        .hint_text("Click to browse or paste path here…")
                );
            });

            ui.separator();

            ui.add(
                egui::Slider::new(&mut self.skip_start, 0.0..=120.0)
                    .text("Skip start (seconds)"),
            );

            ui.add(
                egui::Slider::new(&mut self.skip_end, 0.0..=120.0)
                    .text("Skip end (seconds)"),
            );

            ui.checkbox(&mut self.skip_tempo, "Skip tempo detection");

            ui.separator();

            if !self.running {
                if ui.button("Run analysis").clicked() {
                    self.start_analysis();
                }
            } else {
                ui.label("Processing...");
                ui.add(egui::ProgressBar::new(self.progress).show_percentage());
                if let Ok(mut child) = self.child.lock() {
                    if let Some(child) = child.as_mut() {
                        if ui.button("Cancel").clicked() {
                            let _ = child.kill();
                            self.running = false;
                        }
                    }
                }
            }

            ui.separator();

            if let Some(result) = &self.result {
                ui.heading("Result");
                ui.label(result);
            }

            ui.separator();

            ui.heading("Log output");

            egui::ScrollArea::vertical()
                .max_height(250.0)
                .show(ui, |ui| {
                    for line in &self.log {
                        ui.label(line);
                    }
                });
        });

        // error popup
        if let Some(error) = &self.error.clone() {
            egui::Window::new("Error")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(error);
                    if ui.button("Close").clicked() {
                        self.error = None;
                    }
                });
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "videodiff",
        options,
        Box::new(|_| Box::new(VideoDiffGui::default())),
    )
}
