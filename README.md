# videodiff
Find time offset and scaling between two versions of the same video.

Typical usage is to sync the audio of two versions of the same video.
This software compares the photage and provides an offset and scale
for adjusting the target file audio so that it matches the reference file.

A image hash algorithm is used to compare frames. The image comparison is
quite strict, so you cannot use it to sync different photage of the same
scene from different cameras.

## Requirements
- ffmpeg (runtime requirement)
- rust (if built from source)

## Use-Case:
I built this tool to sync old TV-broadcasted dubbed anime with a bluray
version to join the dubbed audio together with the improved image quality
that a bluray provides.

Because of my limited use-case, I have no idea if this software works well for
comparing other kinds of sources.

## Build
```
cargo build --release
```

## Usage
```
Usage: videodiff [OPTIONS] <ref_file> <target_file>

Arguments:
  <ref_file>
  <target_file>

Options:
      --skip-start <secs>  Seconds to skip at start of file
      --skip-end <secs>    Seconds to skip at end of file
  -h, --help               Print help information
  -V, --version            Print version information
```

An example might look like so:
```
videodiff ~/video_ref.mkv ~/video_target.mkv
```
Log:
```
Extracting frames from /Users/test/video_ref.mkv...
Extracting frames from /Users/test/video_target.mkv...
  - Fps: 25
Decoding frames...
  - Got 31888 images.
Computing hashes...
  - Fps: 23.976025
Decoding frames...
  - Got 30649 images.
Computing hashes...
Indexing frames...
Comparing frames... (slow)
  - Got 155 comparable frames
Optimizing offset and scale...
  - [Result] -        ss: 0.00000s, atempo: 1.00000000, error: 1000000.00
  - [Result] -        ss: 1.93146s, atempo: 1.00361616, error: 9510.17
  - [Result] - itsoffset: 6.48952s, atempo: 1.00187226, error: 6831.05
  - [Result] -        ss: 3.07356s, atempo: 0.99189467, error: 4045.38
  - [Result] - itsoffset: 5.11055s, atempo: 1.00167032, error: 3877.90
  - [Result] - itsoffset: 4.51322s, atempo: 1.00166945, error: 2812.60
  - [Result] - itsoffset: 3.91662s, atempo: 1.00166944, error: 1950.80
  - [Result] - itsoffset: 3.32061s, atempo: 1.00166944, error: 1292.43
  - [Result] - itsoffset: 2.72520s, atempo: 1.00166944, error: 842.11
  - [Result] - itsoffset: 2.13039s, atempo: 1.00166944, error: 620.97
  - [Result] - itsoffset: 1.53617s, atempo: 1.00166944, error: 620.45
  - [Result] -        ss: 0.77154s, atempo: 0.99892964, error: 365.63
  - [Result] -        ss: 0.18084s, atempo: 0.99892991, error: 68.28
  - [Result] - itsoffset: 0.20129s, atempo: 0.99892991, error: 31.95
  - [Result] - itsoffset: 0.04572s, atempo: 0.99892991, error: 26.72
  - [Result] - itsoffset: 0.15528s, atempo: 0.99901764, error: 24.67
```

This can be used with ffmpeg to sync like so:

```
ffmpeg -i ~/video_ref.mkv -itsoffset 0.15528 -i ~/video_target.mkv -filter:a:1 "atempo=0.99901764" -map 0 -map 1:a -c:v copy -c:a:0 copy -c:a:1 libopus -b:a:1 96k ~/out.mkv
```

## Notes
More options should probably be tunable via cli.

This is very much untested software, use at own risk.