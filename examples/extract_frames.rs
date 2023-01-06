#![feature(iter_array_chunks)]

use average::{Mean, Variance};
use clap::{Arg, Command};
use embedded_graphics::{pixelcolor::Rgb888, prelude::*};
use std::thread;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    process::Output,
};
use tinytga::Tga;
use xshell::Shell;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PixelBrightness {
    data: u8,
}

impl PixelBrightness {
    fn compare(&self, other: &PixelBrightness) -> u8 {
        let cmp = self.data ^ other.data;
        ((cmp & 1) > 0) as u8
            + ((cmp & 2) > 0) as u8
            + ((cmp & 4) > 0) as u8
            + ((cmp & 8) > 0) as u8
            + ((cmp & 16) > 0) as u8
            + ((cmp & 32) > 0) as u8
            + ((cmp & 64) > 0) as u8
            + ((cmp & 128) > 0) as u8
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ImageBrightness<const W: usize, const H: usize> {
    data: [[PixelBrightness; W]; H],
}

impl<const W: usize, const H: usize> ImageBrightness<W, H> {
    fn compare(&self, other: &ImageBrightness<W, H>) -> f32 {
        let mut sum = 0.0;
        for (self_row, other_row) in self.data.iter().zip(other.data.iter()) {
            for (self_i, other_i) in self_row.iter().zip(other_row.iter()) {
                if self_i.compare(other_i) == 0 {
                    sum += 1.0
                }
            }
        }
        sum / ((W * H) as f32)
    }
}

struct Frame<const W: usize, const H: usize> {
    frame_nr: usize,
    timestamp: f32,
    hashes: Vec<[u8; W]>,
    image: ImageBrightness<W, H>,
}

impl<const W: usize, const H: usize> Hash for Frame<W, H> {
    fn hash<HF: std::hash::Hasher>(&self, state: &mut HF) {
        self.image.hash(state);
    }
}

impl<const W: usize, const H: usize> PartialEq for Frame<W, H> {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
    }
}

impl<const W: usize, const H: usize> Eq for Frame<W, H> {}

fn extract_frames<const W: usize, const H: usize>(in_file: &str, skip_start_secs: f32) -> Output {
    let filter = format!("[0:v:0]scale={W}:{H}");
    let skip_start_secs = format!("{skip_start_secs}");
    let args = vec![
        "-ss",
        &skip_start_secs,
        "-i",
        in_file,
        "-map",
        "0:v",
        "-filter_complex",
        &filter,
        "-vsync",
        "0",
        "-frame_pts",
        "true",
        // "-t", "00:10:10", // remove later
        "-c:v",
        "targa",
        "-f",
        "image2pipe",
        "-",
    ];
    let shell = Shell::new().expect("Unable to open shell");
    shell
        .cmd("ffmpeg")
        .args(args)
        .quiet()
        .output()
        .expect("Unable to execute ffmpeg")
}

fn parse_fps(bytestream: Vec<u8>) -> f32 {
    // Super ugly way to get fps for video stream:
    let dbg_output =
        String::from_utf8(bytestream).expect("Unable to read stdout from ffmpeg process");
    let fps = dbg_output
        .as_bytes()
        .windows(20)
        .filter(|chars| {
            let (c0, c1, c2, c3, c4) = (chars[15], chars[16], chars[17], chars[18], chars[19]);
            c0 == (' ' as u8)
                && c1 == ('f' as u8)
                && c2 == ('p' as u8)
                && c3 == ('s' as u8)
                && c4 == (',' as u8)
        })
        .map(|chars| {
            let mut fps: Vec<_> = chars[0..15]
                .iter()
                .map(|c| *c)
                .rev()
                .take_while(|c| *c != (',' as u8))
                .collect();
            fps.reverse();
            let fps =
                String::from_utf8(fps).expect("Unable to convert \"fps\" string to utf8 string");
            let fps: f32 = fps[1..]
                .parse()
                .expect(&format!("Unable to parse fps from \"{}\"", fps));
            fps
        })
        .next()
        .unwrap();
    // Fix inaccuracy
    if fps > 23.979 && fps < 23.981 {
        return 24000.0 / 1001.0;
    }
    fps
}

fn images_from_bytestream(data: &[u8]) -> Vec<Tga<Rgb888>> {
    let mut frames: Vec<Tga<Rgb888>> = vec![];
    let mut start: usize = 0;
    let mut end: usize = 0;

    // Find the footer of the file, then parse and put the result in
    while end + 18 < data.len() {
        while &data[end..(end + 18)] != "TRUEVISION-XFILE.\0".as_bytes() {
            end += 1;
        }
        end += 18;
        if let Ok(frame) = Tga::from_slice(&data[start..end]) {
            // println!("{}: [{}, {}]",frames.len(), start, end);
            frames.push(frame);
            start = end;
        }
    }
    frames
}

fn make_frames<const W: usize, const H: usize>(
    images: &[Tga<Rgb888>],
    fps: f32,
) -> Vec<Frame<W, H>> {
    let mut images: Vec<_> = images
        .iter()
        .map(|f| {
            let mut brightness = ImageBrightness::<W, H> {
                data: [[PixelBrightness { data: 0 }; W]; H],
            };
            let pixels: Vec<_> = f.pixels().collect();
            for x in 1..(W - 1) {
                for y in 1..(H - 1) {
                    let c = pixels[(x + 0) + (y + 0) * H].1;
                    let l = pixels[(x - 1) + (y + 0) * H].1;
                    let r = pixels[(x + 1) + (y + 0) * H].1;
                    let u = pixels[(x + 0) + (y - 1) * H].1;
                    let d = pixels[(x + 0) + (y + 1) * H].1;
                    let c = c.r() as i16 + c.b() as i16 + c.g() as i16;
                    let l = l.r() as i16 + l.b() as i16 + l.g() as i16;
                    let r = r.r() as i16 + r.b() as i16 + r.g() as i16;
                    let u = u.r() as i16 + u.b() as i16 + u.g() as i16;
                    let d = d.r() as i16 + d.b() as i16 + d.g() as i16;
                    let px = &mut brightness.data[x][y].data;

                    let l = (c - l).abs();
                    let r = (c - r).abs();
                    let u = (c - u).abs();
                    let d = (c - d).abs();

                    *px = (l > 50) as u8 * 1
                        + (l < -50) as u8 * 2
                        + (r > 50) as u8 * 4
                        + (r < -50) as u8 * 8
                        + (u > 50) as u8 * 16
                        + (u < -50) as u8 * 32
                        + (d > 50) as u8 * 64
                        + (d < -50) as u8 * 128;
                }
            }
            brightness
        })
        .collect();
    let delta_time = 1.0 / fps;
    images
        .drain(..)
        .enumerate()
        .map(|(frame_nr, image)| {
            let hashes: Vec<_> = image
                .data
                .iter()
                .map(|row| {
                    let res: [_; W] = row.iter().map(|i| i.data).array_chunks().next().unwrap();
                    res
                })
                // At least 25% of the pixels should be non 0 for an image-hash to be somewhat useful
                .filter(|h| h.iter().filter(|p| **p != 0).count() > W / 4)
                .collect();
            Frame::<W, H> {
                frame_nr,
                image,
                timestamp: frame_nr as f32 * delta_time,
                hashes,
            }
        })
        .collect()
}

fn populate_lookup_table<'a, const W: usize, const H: usize>(
    table: &mut HashMap<[u8; W], HashSet<&'a Frame<W, H>>>,
    frames: &'a [Frame<W, H>],
) {
    for frame in frames {
        let hashes = &frame.hashes;
        for hash in hashes {
            let matched_frames = match table.get_mut(hash) {
                Some(matched_frames) => matched_frames,
                _ => {
                    table.insert(*hash, HashSet::new());
                    table.get_mut(hash).unwrap()
                }
            };
            matched_frames.insert(frame);
        }
    }
}

fn lookup<'a, const W: usize, const H: usize>(
    table: &HashMap<[u8; W], HashSet<&'a Frame<W, H>>>,
    frame: &'a Frame<W, H>,
) -> HashSet<&'a Frame<W, H>> {
    let mut result = HashSet::new();
    for hash in &frame.hashes {
        let Some(hits) = table.get(hash) else { continue };
        result.extend(hits)
    }
    result
}

fn append_mean_and_variance<'a, const W: usize, const H: usize>(
    table: &HashMap<[u8; W], HashSet<&'a Frame<W, H>>>,
    frame: &'a Frame<W, H>,
) -> (&'a Frame<W, H>, (Mean, Variance)) {
    let results = lookup(&table, &frame);
    let it = results
        .iter()
        .map(|other_frame| {
            let score = frame.image.compare(&other_frame.image);
            let timestamp = other_frame.timestamp;
            (score, timestamp)
        })
        .filter(|(score, _)| *score > 0.65)
        .map(|(_, timestamp)| timestamp as f64);
    let mean: Mean = it.clone().collect();
    let variance: Variance = it.collect();
    // eprintln!("{}: {}, +-{}", frame.frame_nr, mean.mean(), variance.sample_variance());
    (frame, (mean, variance))
}

const WIDTH: usize = 20;
const HEIGHT: usize = 20;

fn main() {
    ///////////////////////////////////////////////////////////////////////////////////////////
    //  Argument parsing
    ///////////////////////////////////////////////////////////////////////////////////////////
    let args = Command::new("diffvid")
        .version("0.1.0")
        .about("Extract images from file")
        .arg(Arg::new("ref_file").required(true))
        .arg(Arg::new("target_file").required(true))
        .arg(
            clap::arg!(--"skip-start" <secs>)
                .value_parser(clap::value_parser!(f32))
                .help("Seconds to skip at start of file"),
        )
        .arg(
            clap::arg!(--"skip-end" <secs>)
                .value_parser(clap::value_parser!(f32))
                .help("Seconds to skip at end of file"),
        )
        .arg_required_else_help(true)
        .get_matches();

    let skip_start = *args.get_one::<f32>("skip-start").unwrap_or(&0.0);
    let skip_end = *args.get_one::<f32>("skip-end").unwrap_or(&0.0);
    let ref_file = args
        .get_one::<String>("ref_file")
        .expect("Reference file required")
        .clone();
    let target_file = args
        .get_one::<String>("target_file")
        .expect("Target file required")
        .clone();

    ///////////////////////////////////////////////////////////////////////////////////////////
    //  Extract and load frames from ffmpeg (in parallel)
    ///////////////////////////////////////////////////////////////////////////////////////////
    let load_frames = move |file_path: String| {
        eprintln!("Extracting frames from {file_path}...");
        let output = extract_frames::<WIDTH, HEIGHT>(&file_path, skip_start);
        let bytestream = output.stdout;
        let fps = parse_fps(output.stderr);
        eprintln!("  - Fps: {}", fps);
        eprintln!("Decoding frames...");
        let images = images_from_bytestream(&bytestream);
        eprintln!("  - Got {} images.", images.len());
        let range = 0..=(images.len() - (skip_end * fps) as usize);
        eprintln!("Computing hashes...");
        let frames = make_frames::<WIDTH, HEIGHT>(&images[range], fps);
        return frames;
    };

    let ref_file_job = thread::spawn(move || load_frames(ref_file));
    let target_file_job = thread::spawn(move || load_frames(target_file));

    let frames_ref = ref_file_job.join().unwrap();
    let frames_target: Vec<_> = target_file_job.join().unwrap();

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Index frames for faster lookup
    ///////////////////////////////////////////////////////////////////////////////////////////
    eprintln!("Indexing frames...");
    let mut table_ref: HashMap<[u8; WIDTH], HashSet<&Frame<WIDTH, HEIGHT>>> = HashMap::new();
    populate_lookup_table(&mut table_ref, &frames_ref);

    let mut table_target: HashMap<[u8; WIDTH], HashSet<&Frame<WIDTH, HEIGHT>>> = HashMap::new();
    populate_lookup_table(&mut table_target, &frames_target);

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Find good, distinct frames and the difference in time between frames from both sources
    ///////////////////////////////////////////////////////////////////////////////////////////
    eprintln!("Comparing frames (slow)...");
    let matched_frames_ref: HashMap<&Frame<WIDTH, HEIGHT>, (Mean, Variance)> = frames_ref
        .iter()
        .map(|frame| append_mean_and_variance(&table_ref, frame))
        // Remove frames with high variance or without matches
        .filter(|(_, (_, varinace))| varinace.len() > 0 && varinace.sample_variance() < 10.0)
        .collect();

    let matched_frames_target: HashMap<&Frame<WIDTH, HEIGHT>, (Mean, Variance)> = frames_ref
        .iter()
        .map(|frame| append_mean_and_variance(&table_target, frame))
        // Remove frames with high variance or without matches
        .filter(|(_, (_, varinace))| varinace.len() > 0 && varinace.sample_variance() < 10.0)
        .collect();

    // A list of tuples with the timestamp that a frame comes in each ref and target video.
    let mut ref_and_target_timestamps: Vec<_> = matched_frames_ref
        .iter()
        .flat_map(|(frame, (mean, var))| {
            let Some((mean2, var2)) = matched_frames_target.get(frame) else { return None };
            let var_factor = var.sample_variance() / var2.sample_variance();
            // If variance is similar, assume comparison of frames are good
            if var_factor < 0.57 || var_factor > 1.83 {
                return None;
            };
            Some((frame.frame_nr, (mean.mean(), mean2.mean())))
        })
        .collect();

    ref_and_target_timestamps
        .sort_by(|(self_nr, _), (other_nr, _)| self_nr.partial_cmp(&other_nr).unwrap());

    if ref_and_target_timestamps.len() < 100 {
        eprintln!(
            "Error, too few samples, only got {} similar frames.",
            ref_and_target_timestamps.len()
        );
        return;
    }
    eprintln!("  - Got {} comparable frames", ref_and_target_timestamps.len());

    // // Debug print samples
    // for (frame_nr, (ref_ts, tar_ts)) in ref_and_target_timestamps.iter() {
    //     eprintln!("    - {}:\t{:.2},   \t{:.2}", frame_nr, ref_ts, tar_ts);
    // }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Solve time difference (offset + scale) with bird/partical swarm optimization
    ///////////////////////////////////////////////////////////////////////////////////////////
    eprintln!("Optimizing offset and timestamp scale...");

    struct Bird {
        pos: (f64, f64),
        vel: (f64, f64),
        best: ((f64, f64), f64),
    }

    let evaluate_fit = |offset: f64, scale: f64| {
        let mut error = 0.0;
        
        // Add extra error weight to last 100 and first 100 frames.
        for (_, (ref_ts, target_ts)) in &ref_and_target_timestamps[0..100] {
            error += (ref_ts - target_ts * scale + offset).powi(2);
        }
        for (_, (ref_ts, target_ts)) in
            &ref_and_target_timestamps[ref_and_target_timestamps.len() - 100..]
        {
            error += (ref_ts - target_ts * scale + offset).powi(2);
        }
        for (_, (ref_ts, target_ts)) in &ref_and_target_timestamps {
            error += (ref_ts - target_ts * scale + offset).abs();
        }
        error
    };

    let sample_range = |min, max| rand::random::<f64>() * (max - min) + min;

    let spawn_bird = || Bird {
        pos: (sample_range(-10.0, 10.0), sample_range(0.99, 1.01)),
        vel: (0.0, 0.0),
        best: ((0.0, 1.0), 1_000_000.0),
    };

    let mut swarm: Vec<_> = (0..10).map(|_| spawn_bird()).collect();
    let mut best: ((f64, f64), f64) = ((0.0, 1.0), 1_000_000.0);
    let mut bird_weight = 0.6;
    let personal_weight = 0.1;
    let global_weight = 0.05;

    for _ in 0..400_000 {
        for bird in &mut swarm {
            // Calculate global and local
            let global_direction = (best.0 .0 - bird.pos.0, best.0 .1 - bird.pos.1);
            let personal_direction = (bird.best.0 .0 - bird.pos.0, bird.best.0 .1 - bird.pos.1);
            // normalize velocity
            let vel_scale = (bird.vel.0 * bird.vel.0 + bird.vel.1 * bird.vel.1).sqrt();
            let bird_vel = (
                bird.vel.0 / (vel_scale + 0.001),
                bird.vel.1 / (vel_scale + 0.001),
            );
            bird.vel = (
                bird_vel.0 * bird_weight
                    + (personal_direction.0 * personal_weight + global_direction.0 * global_weight)
                        * 0.2,
                bird_vel.1 * bird_weight * 0.01
                    + (personal_direction.1 * personal_weight + global_direction.1 * global_weight)
                        * 0.2,
            );

            bird.pos = (bird.pos.0 + bird.vel.0, bird.pos.1 + bird.vel.1);

            let score = evaluate_fit(bird.pos.0, bird.pos.1);
            let ((best_offset, best_scale), best_score) = best;
            if score.round() < best_score.round() {
                eprintln!(
                    "  - [Result] - itsoffset: {:.5}s, atempo: {:.8}, score: {:.2}",
                    -best_offset,
                    1.0 / best_scale,
                    best_score
                );
            }
            if score < best_score {
                best = (bird.pos, score)
            };
            let (_, best_score) = bird.best;
            if score < best_score {
                bird.best = (bird.pos, score)
            };
        }
        if bird_weight > 0.01 {
            bird_weight = bird_weight * 0.999;
        }
    }
    let ((best_offset, best_scale), best_score) = best;
    eprintln!(
        "  - [Result] - itsoffset: {:.5}s, atempo: {:.8}, score: {:.2}",
        -best_offset,
        1.0 / best_scale,
        best_score
    );
    if best_score as usize > ref_and_target_timestamps.len() * 2 {
        eprintln!(
            "Warning! Average frame error is over {}, which means that the fit might be bad.",
            best_score
        );
    }
}
