# Motion Extraction and Curation

This directory contains the production `merge_aios_wilor.py`, `align_views.py`, and `fuse_poses.py` snapshot used for the regenerated How2Sign-MV3D release, plus portable entry points. `source_snapshot.json` records source hashes. Raw estimator weights, SMPL-X/MANO files, textures, and videos are not bundled.

## 1. Extract Raw Predictions (GPU)

Install [AiOS](https://github.com/ttxskk/AiOS) and [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) with their dependencies and licensed model assets. Use their respective environments. `aios_adapter/` contains the frozen inference adaptations, with the upstream AiOS license retained. The WiLoR adapter uses the installed pipeline's batched `predict_with_bboxes` interface.

Write one full video path per line in `videos.txt`. Run in a Slurm GPU allocation:

```bash
python pipeline/extract_aios.py \
  --aios-root /path/to/AiOS --checkpoint /path/to/aios_checkpoint.pth \
  --videos videos.txt --output raw/aios --batch-size 16
```

WiLoR accepts a JSON list of records with a `videos` dictionary:

```json
[{"videos": {"front": "/path/clip_front.mp4", "side": "/path/clip_side.mp4"}}]
```

```bash
WILOR_MINI_ROOT=/path/to/WiLoR-mini \
python pipeline/extract_wilor.py --records videos.json --out raw/wilor --batch-size 32
```

The models are initialized once per invocation. AiOS emits `<video_stem>_amass.npz`; WiLoR emits `<video_stem>_hands.npz` and decoding metadata. Frames are padded to 16:9 without frame-rate resampling. WiLoR uses detector confidence 0.3 and float16. Read actual FPS from decoded-video metadata, not a nominal container frame count. Install FFprobe for independent displayed-frame-count checks.

## 2. Fuse and Curate (CPU)

```bash
pip install -r pipeline/requirements_cpu.txt
python pipeline/curate.py \
  --aios-front raw/aios/clip_front_amass.npz \
  --wilor-front raw/wilor/clip_front_hands.npz \
  --aios-side raw/aios/clip_side_amass.npz \
  --wilor-side raw/wilor/clip_side_hands.npz \
  --fps 24 --sample-id clip --split train --out final/clip.npz
```

Use the two videos' common actual FPS; different view frame rates require explicit preprocessing, not this frame-offset alignment. At least eight decoded frames are required. The wrapper runs both per-view fusions, temporal/spatial alignment, **enabled non-sign pruning**, cross-view hand correction, and **enabled coif3 level-2 body wavelet smoothing**. PyWavelets 1.8.0 is required; there is no silent Savitzky-Golay fallback. It writes one final frontal-coordinate SMPL-X sequence plus a JSON trace. The optional `--keep-side` output is diagnostic, not a second training sample.

Hand repair jointly checks both views and neighboring frames. The frontal body coordinate representation is retained for the final output. The joint-frame matrices are stored as small JSON constants, with the SMPL-X finger order `index, middle, pinky, ring, thumb`; the wrapper recreates a local NPZ cache without shipping model vertices. To recompute the constants from a separately licensed neutral SMPL-X model, use `solve_hand_coordinate_frames.py --model /path/SMPLX_NEUTRAL.npz --out /path/frames`.

## Fields and Parameters

The exact thresholds are in the source snapshot, not hidden in a trained repair model. Important settings include a temporal search over +/-8 frames, four synthetic motion-blend frames per internal splice, cross-view finger outlier factor 1.15, collapse ratio 0.65, neighbor checks at +/-1 frame, and coif3 level-2 body smoothing. Non-sign detection uses the production activity/height/finger rules and preserves the original sentence label; it is not word-level transcript realignment.

`poses` contains SMPL-X local axis-angle rotations in radians; `betas` contains fitted shape parameters. `source_frame_index`, `source_padding_mask`, `synthetic_bridge_mask`, and `wilor_hand_valid_mask` retain temporal/validity bookkeeping. `trans` may be a placeholder; `cam_trans` is not a calibrated motion-capture camera trajectory. Saved tracking keypoints are not recomputed FK from repaired poses.

To process a single-view dataset, call `merge_aios_wilor.py` with `--aios_dir`, `--wilor_dir`, and `--output_dir`. That operation does not provide cross-view correction; it must not be described as a multi-view reconstruction.

## Scope

The release wrapper is intentionally separate from cluster-specific queue supervisors. The mathematical core and bookkeeping hooks are frozen from the production run; portable argument parsing and model-path configuration are release additions. No new GPU inference is required to read the published final motion files.
