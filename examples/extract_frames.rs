#![feature(iter_array_chunks)]

use std::{process::Output, collections::{HashMap, HashSet}, hash::Hash};
use average::{Mean, Variance};
use embedded_graphics::{pixelcolor::Rgb888, prelude::*};
use clap::{Command, Arg};
use tinytga::{Tga};
use xshell::Shell;
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PixelBrightness {
    data: u8
}

impl PixelBrightness {
    fn compare(&self, other: &PixelBrightness) -> u8 {
        let cmp = self.data ^ other.data;
        ((cmp & 1) > 0) as u8 + ((cmp & 2) > 0) as u8 +
        ((cmp & 4) > 0) as u8 + ((cmp & 8) > 0) as u8 +
        ((cmp & 16) > 0) as u8 + ((cmp & 32) > 0) as u8 +
        ((cmp & 64) > 0) as u8 + ((cmp & 128) > 0) as u8
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ImageBrightness<const W: usize, const H: usize> {
    data: [[PixelBrightness; W]; H]
}

impl <const W: usize, const H: usize> ImageBrightness<W,H> {
    fn compare(&self, other: &ImageBrightness<W,H>) -> f32 {
        let mut sum = 0.0;
        for (self_row, other_row) in self.data.iter().zip(other.data.iter()) {
            for (self_i, other_i) in self_row.iter().zip(other_row.iter()) {
                if self_i.compare(other_i) == 0 { sum += 1.0 }
            }
        }
        sum / ((W*H) as f32)
    }
}

struct Frame<const W: usize, const H: usize> {
    timestamp: f32,
    hashes: Vec<[u8; W]>,
    image: ImageBrightness<W,H>,
}

impl <const W: usize, const H: usize>Hash for Frame<W,H> {
    fn hash<HF: std::hash::Hasher>(&self, state: &mut HF) {
        self.image.hash(state);
    }
}

impl <const W: usize, const H: usize>PartialEq for Frame<W,H> {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
    }
}

impl <const W: usize, const H: usize>Eq for Frame<W,H> {}

fn extract_frames<const W: usize, const H: usize>(in_file: &str) -> Output{
    let filter = format!("[0:v:0]scale={W}:{H}");
    let args = vec![
        "-ss", "00:02:00", // Always skip intro
        "-i", in_file, "-map", "0:v", 
        "-filter_complex", &filter, 
        "-vsync", "0", "-frame_pts", "true", 
        // "-t", "00:05:10", // remove later
        "-c:v", "targa", "-f", "image2pipe", "-"
    ];
    let shell = Shell::new().expect("Unable to open shell");
    shell.cmd("ffmpeg").args(args).quiet().output().expect("Unable to execute ffmpeg")
}

fn parse_fps(bytestream: Vec<u8>) -> f32 {
    // Super ugly way to get fps for video stream: 
    let dbg_output = String::from_utf8(bytestream).expect("Unable to read stdout from ffmpeg process");
    let fps = dbg_output.as_bytes().windows(20).filter(|chars| {
        let (c0, c1, c2, c3, c4) = (chars[15], chars[16], chars[17], chars[18], chars[19]);
        c0 == (' ' as u8) && c1 == ('f' as u8) && c2 == ('p' as u8) && c3 == ('s' as u8) && c4 == (',' as u8)
    }).map(|chars| {
        let mut fps: Vec<_> = chars[0..15].iter().map(|c| *c).rev().take_while(|c| *c != (',' as u8)).collect();
        fps.reverse();
        let fps = String::from_utf8(fps).expect("Unable to convert \"fps\" string to utf8 string");
        let fps: f32 = fps[1..].parse().expect(&format!("Unable to parse fps from \"{}\"", fps));
        fps
    }).next().unwrap();
    // Fix inaccuracy
    if fps > 23.979 && fps < 23.981 {
        return 24000.0 / 1001.0
    }
    fps
}

fn images_from_bytestream(data: &[u8]) -> Vec<Tga<Rgb888>>{
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

fn make_frames<const W: usize, const H: usize>(images: &[Tga<Rgb888>], fps: f32) -> Vec<Frame<W,H>> {
    let mut images: Vec<_> = images.iter().map(|f| {
        let mut brightness = ImageBrightness::<W,H> {
            data: [[PixelBrightness { data: 0 }; W]; H]
        };
        let pixels: Vec<_> = f.pixels().collect();
        for x in 1..(W-1){
            for y in 1..(H-1){
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

                *px   = (l > 50) as u8 *  1 + (l < -50) as u8 *   2 +
                        (r > 50) as u8 *  4 + (r < -50) as u8 *   8 +
                        (u > 50) as u8 * 16 + (u < -50) as u8 *  32 +
                        (d > 50) as u8 * 64 + (d < -50) as u8 * 128;
            }
        }
        brightness
    }).collect();
    let delta_time = 1.0 / fps;
    images.drain(..).enumerate().map(|(frame_nr, image)| {
        let hashes: Vec<_> = image.data.iter()
            .map(|row| {
                let res: [_; W] = row.iter().map(|i| i.data).array_chunks().next().unwrap();
                res
            })
            // At least 25% of the pixels should be non 0 for an image-hash to be somewhat useful
            .filter(|h| h.iter().filter(|p| **p != 0).count() > W/4) 
            .collect();
        Frame::<W,H> { image, timestamp: frame_nr as f32 * delta_time, hashes }
    }).collect()
}

fn populate_lookup_table<'a, const W: usize, const H: usize>(table: &mut HashMap<[u8; W], HashSet<&'a Frame::<W,H>>>, frames: &'a [Frame::<W,H>]){
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

fn lookup<'a, const W: usize, const H: usize>(table: & HashMap<[u8; W], HashSet<&'a Frame::<W,H>>>, frame: &'a Frame::<W,H>) -> HashSet<&'a Frame::<W,H>> {
    let mut result = HashSet::new();
    for hash in &frame.hashes {
        let Some(hits) = table.get(hash) else { continue };
        result.extend(hits)
    }
    result
}

fn append_mean_and_variance<'a, const W: usize, const H: usize>(table: & HashMap<[u8; W], HashSet<&'a Frame::<W,H>>>, frame: &'a Frame::<W,H>) -> (&'a Frame::<W,H>, (Mean, Variance)) {
    let results = lookup(&table, &frame);
    let it = results.iter()
        .map(|other_frame| {
            let score = frame.image.compare(&other_frame.image);
            let timestamp = other_frame.timestamp;
            (score, timestamp)
        })
        .filter(|(score, _)| *score > 0.5)
        .map(|(_, timestamp)| timestamp as f64);
    let mean: Mean = it.clone().collect();
    let variance: Variance = it.collect();
    // eprintln!("{}: {}, +-{}", frame.frame_nr, mean.mean(), variance.sample_variance());
    (frame, (mean, variance))
}

const WIDTH: usize = 40;
const HEIGHT: usize = 40;

fn main() {
    let args = Command::new("extract_frames")
        .bin_name("cargo run --example extract_frames")
        .version("0.1.0")
        .about("Extract images from file")
        .arg(
            Arg::new("ref_file")
            .required(true)
        )
        .arg(
            Arg::new("target_file")
            .required(true)
        )
        .after_help("Longer explanation to appear after the options when \
                    displaying the help information from --help or -h")
        .get_matches();

    let load_frames = |file_path: String| {
        eprintln!("Extracting frames from {file_path}...");
        let output = extract_frames::<WIDTH,HEIGHT>(&file_path);
        let bytestream = output.stdout;
        let fps = parse_fps(output.stderr);
        eprintln!("  - Fps: {}", fps);
        eprintln!("Decoding frames...");
        let images = images_from_bytestream(&bytestream);
        eprintln!("  - Got {} images.", images.len());
        eprintln!("Computing hashes...");
        let frames = make_frames::<WIDTH,HEIGHT>(&images, fps);
        return frames;
    };

    let ref_file = args.get_one::<String>("ref_file").expect("Reference file required").clone();
    let target_file = args.get_one::<String>("target_file").expect("Target file required").clone();

    let ref_file_job = thread::spawn(move || {
        load_frames(ref_file)
    });
    let target_file_job = thread::spawn(move || {
        load_frames(target_file)
    });

    let mut frames_ref = ref_file_job.join().unwrap();
    let max_timestamp = frames_ref.iter()
        .map(|frame| frame.timestamp)
        .fold(f32::NAN, f32::max);

    // Looks bad but is efficient due to move operation.
    let frames_ref: Vec<_> = frames_ref.drain(..).filter(|frame|
        frame.timestamp < max_timestamp - 60.0
    ).collect();

    let frames_target: Vec<_> = target_file_job.join().unwrap().drain(..).filter(|frame|
        frame.timestamp < max_timestamp - 60.0
    ).collect();

    eprintln!("Indexing frames...");
    let mut table_ref: HashMap<[u8; WIDTH], HashSet<&Frame::<WIDTH,HEIGHT>>> = HashMap::new();
    populate_lookup_table(&mut table_ref, &frames_ref);
    
    let mut table_target: HashMap<[u8; WIDTH], HashSet<&Frame::<WIDTH,HEIGHT>>> = HashMap::new();
    populate_lookup_table(&mut table_target, &frames_target);
    
    eprintln!("Comparing frames...");
    let matched_frames_ref: HashMap<&Frame::<WIDTH,HEIGHT>, (Mean, Variance)> = frames_ref.iter()
        .map(|frame| append_mean_and_variance(&table_ref, frame))
        // Remove frames with high variance or without matches  
        .filter(|(_, (_, varinace))| varinace.len() > 0 && varinace.sample_variance() < 1.0)
        .collect();

    let matched_frames_target: HashMap<&Frame::<WIDTH,HEIGHT>, (Mean, Variance)> = frames_ref.iter()
        .map(|frame| append_mean_and_variance(&table_target, frame))
        // Remove frames with high variance or without matches  
        .filter(|(_, (_, varinace))| varinace.len() > 0 && varinace.sample_variance() < 1.0)
        .collect();

    let frames_ref_with_samples: Vec<_> = matched_frames_ref
        .keys()
        .map(|frame| *frame)
        .filter(|frame| matched_frames_target.get(frame).is_some())
        .collect();
    eprintln!("  - Got {} similar frames.", frames_ref_with_samples.len());

    let sample_size = frames_ref_with_samples.len();

    let sample_frame = || {
        frames_ref_with_samples[rand::random::<usize>() % sample_size]
    };

    // Print frame timestamps:

    // let mut sorted_ref: Vec<_> = matched_frames_ref.iter().collect();
    // sorted_ref.sort_by(|(frame1, _), (frame2, _)| frame1.frame_nr.partial_cmp(&frame2.frame_nr).unwrap());

    // for (frame, (mean, variance)) in sorted_ref.iter() {
    //     let Some((mean2, variance2)) = matched_frames_target.get(*frame) else { continue };
    //     eprintln!("{}: {}, +-{} \t {}, +-{}", frame.frame_nr, mean.mean(), variance.sample_variance(), mean2.mean(), variance2.sample_variance());
    // }

    
    eprintln!("Optimizing offset and timestamp scale...");

    let mut offset = 0.0;
    let mut scale = 1.0;


    let mut offset_step = 0.002;

    // TODO: Change from this to bird swarm optimization and instead evalutate rather than "pull" in the right direction.
    // Also reduce iterations drastically in that case.
    for i in 0..4_000_000 {
        let sample1 = sample_frame();
        let mut retries = 0;
        let sample2 = loop {
            retries += 1;
            let sample2 = sample_frame();
            if (sample2.timestamp - sample1.timestamp).abs() > 0.3 * max_timestamp || retries > 10 { break sample2 }
        };
        let timestamp_ref_1 = &matched_frames_ref[sample1].0.mean();
        let timestamp_ref_2 = &matched_frames_ref[sample2].0.mean();
        let timestamp_tar_1 = &matched_frames_target[sample1].0.mean();
        let timestamp_tar_2 = &matched_frames_target[sample2].0.mean();

        // Save old offset so that we don't screw up the calculations
        let old_offset = offset;
        let old_scale = scale;
        
        // Calculate offset and use mean for both samples
        let offset_error_1 = timestamp_ref_1 - timestamp_tar_1 * old_scale + old_offset;
        let offset_error_2 = timestamp_ref_2 - timestamp_tar_2 * old_scale + old_offset;
        offset = offset - (offset_error_1 * 0.5 + offset_error_2 * 0.5).clamp(-offset_step, offset_step); 
        
        // Reduce scale adjustment strength for samples with very different offset errors (bad samples?).
        let scale_adj_strength = offset_step / 100.0 / (offset_error_1 - offset_error_2).abs().max(1.0);

        let scale_error = (timestamp_ref_1             - timestamp_ref_2             + old_offset) / 
                          (timestamp_tar_1 * old_scale - timestamp_tar_2 * old_scale + old_offset);

        scale = (1.0 - scale_adj_strength) * scale + (scale_adj_strength) * scale_error;
        
        if offset_step > 0.00005 {
            offset_step *= 0.99999;
        }

        // Printed offset is ffmpeg param "-itsoffset" for target source,
        // Printed scale is ffmpeg param "atempo" for target source
        if i % 10_000 == 0 { eprintln!("Offset: {}, Scale: {}", -offset, 1.0 / scale)};
    }
}
