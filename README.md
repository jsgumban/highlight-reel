# Highlight Reel

Scores every half-second of a long video (sports session, vlog footage, gaming VOD, family event, screen recording) for excitement across five independent signals — loudness, audio novelty, speech-band energy, visual motion, and hard-cut density — fuses them into one timeline, and emits ranked highlight clips plus a chronological supercut, so an hour of footage shrinks to the two minutes worth watching.

## Setup (once)

`./run.sh` bootstraps everything on first run: it finds a Python >= 3.11 on your PATH, creates `.venv`, and installs the requirements. No system ffmpeg needed either — the `imageio-ffmpeg` package ships a static binary (set `FFMPEG_BIN` to point at your own build instead).

To set up manually:

```
git clone https://github.com/jsgumban/highlight-reel.git && cd highlight-reel
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```
./run.sh /path/to/video.mp4
```

A 90-second reel of a padel session:

```
./run.sh session.mp4 --preset sports --target 90
```

`--preset` picks a weighting table from `config.toml` (`general`, `sports`, `talk`, `gaming`); `--target` overrides `target_total_s`. Or run the stages individually:

```
.venv/bin/python analyze.py /path/to/video.mp4
.venv/bin/python select.py  /path/to/video.mp4 --preset talk --target 300
.venv/bin/python render.py  /path/to/video.mp4
```

Outputs land in `outputs/<video_basename>/`:

| File | What it is |
| --- | --- |
| `clips/highlight_NN_t<start>s.mp4` | ranked highlight clips (01 = most exciting) |
| `supercut.mp4` | the whole video minus the dead parts, chronological |
| `timeline.png` | fused score + per-signal breakdown with selected spans shaded (open this first to sanity-check the selection) |
| `chapters.txt` | YouTube-style chapter list mapping supercut time → source time |
| `contactsheet.jpg` | thumbnail grid of the chosen highlights |
| `segments.json` / `signals.npz` | machine-readable sidecars |

## Tuning

If it keeps too much or too little, edit `config.toml` and re-run. Analysis is the slow step; selection is instant, so iterate with select → check timeline → tweak (analyze once, then re-run `select.py` and `render.py` only).

| Setting | Effect |
| --- | --- |
| `percentile_floor` | Score percentile that seeds candidates. Default 80. Drop toward 70 if good moments are getting missed; raise toward 90 to stay strict. |
| `target_total_s` | Total highlight duration to aim for. 0 keeps everything above threshold. Also `--target` per run. |
| `min_clip_s` / `max_clip_s` | Segment duration bounds. Raise `min_clip_s` if clips feel like blink-and-miss fragments; lower `max_clip_s` if one long stretch eats the whole budget. |
| `merge_gap_s` | How close two excited stretches must be to fuse into one segment. Default 3s. |
| `boundary_snap_s` | How far cut points may slide toward calm moments. Widen if cuts land mid-sentence. |
| `smooth_s` | Smoothing of the fused score. Larger = fewer, longer segments; smaller = choppier, more reactive. |
| `[presets.*]` weights | Per-signal weights. Music inflates `onset` — lower it. Screen recordings live on `motion`. Voice-driven footage wants `speech` up. |

## How it works

1. `ffmpeg` pipes the audio to Python as 22-kHz mono PCM and the video as tiny 4-fps grayscale frames.
2. Five signals are scored per half-second hop: `loudness` (RMS in dB), `onset` (spectral flux — audio novelty), `speech` (300–3400 Hz band energy), `motion` (mean frame difference), `scenecut` (hard-cut density).
3. Each signal becomes a robust z-score (median + MAD, clipped) so raw scales don't matter, then all five fuse into one excitement score using the active preset's weights.
4. The fused score is gaussian-smoothed and thresholded at `percentile_floor`; runs above threshold merge across small gaps, snap their edges into calm moments, and get pre/post roll.
5. Segments are ranked by `0.7·peak + 0.3·mean` score and greedily accepted until the target duration is filled.
6. Selected segments are cut with `ffmpeg -c copy` (fast, no re-encode), concatenated into the supercut, charted, and thumbnailed.

## Notes & limits

- Clips are cut with stream-copy, so they snap to the nearest keyframe boundary. Set `reencode = true` in `[render]` for frame-accurate (much slower) cuts.
- No-audio footage (screen recordings) works fine: the audio signals go to zero and motion carries the score. Audio-only files work too, the other way around.
- Music-heavy footage inflates `onset` — every beat reads as novelty. Lower its weight in your preset.
- The supercut is chronological; the clips are ranked. `highlight_01` is the most exciting moment, not the first.

## Project layout

```
highlight-reel/
├── README.md
├── requirements.txt
├── config.toml
├── common.py
├── analyze.py
├── select.py
├── render.py
├── run.sh
├── .venv/
├── LOG.md
└── outputs/<video_basename>/
    ├── signals.npz
    ├── meta.json
    ├── segments.json
    ├── timeline.png
    ├── supercut.mp4
    ├── chapters.txt
    ├── contactsheet.jpg
    └── clips/highlight_NN_t<start>s.mp4
```
