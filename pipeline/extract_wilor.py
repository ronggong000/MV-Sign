"""Stream WiLoR frames without PNGs or persistent raw hand meshes."""
import argparse
import os
import shutil
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

if os.environ.get('WILOR_MINI_ROOT'):
    sys.path.insert(0, os.environ['WILOR_MINI_ROOT'])
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


def validate_frame_count(path, advertised, decoded):
    """A stream-copy edit list can retain hidden preroll in nb_frames."""
    report = dict(advertised_frames=advertised, decoded_frames=decoded)
    if advertised > 0 and advertised == decoded:
        report['validation'] = 'opencv count matches container'
        return report
    local_probe = Path(sys.executable).with_name('ffprobe')
    probe = str(local_probe) if local_probe.is_file() else shutil.which('ffprobe')
    result = subprocess.run([str(probe), '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                             '-show_entries', 'stream=nb_frames,nb_read_frames', '-of', 'json', str(path)],
                            capture_output=True, text=True, timeout=120)
    if result.returncode or result.stderr.strip():
        raise RuntimeError('Independent decoder rejected video: ' + result.stderr[-1000:])
    stream = json.loads(result.stdout)['streams'][0]
    expected = int(stream['nb_read_frames'])
    if expected != decoded:
        raise RuntimeError(f'Decoder disagreement: ffprobe={expected}, OpenCV={decoded}')
    report.update(validation='independent ffprobe confirms displayed frame count',
                  ffprobe_decoded_frames=expected,
                  container_note='nb_frames may include nondisplayed edit-list preroll')
    return report


def padded_rgb(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if abs(w / h - 16 / 9) < 1e-4:
        return rgb
    height, width = (int(w / (16 / 9)), w) if w / h > 16 / 9 else (h, int(h * (16 / 9)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - h) // 2, (width - w) // 2
    canvas[y:y + h, x:x + w] = rgb
    return canvas


def process(pipe, path, out, batch):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video ' + str(path))
    meta = dict(fps=float(cap.get(cv2.CAP_PROP_FPS)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    advertised = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    parts = []
    try:
        while True:
            frames = []
            for _ in range(batch):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(padded_rgb(frame))
            if not frames:
                break
            n = len(frames)
            values = dict(global_orient=np.zeros((n, 2, 3), np.float32),
                          hand_pose=np.zeros((n, 2, 45), np.float32),
                          pred_keypoints_3d=np.zeros((n, 2, 21, 3), np.float32),
                          valid_mask=np.zeros((n, 2), bool), detection_mask=np.zeros((n, 2), bool),
                          hand_scores=np.zeros((n, 2), np.float32))
            boxes = np.full((n, 2, 4), np.nan, np.float32)
            detections = pipe.hand_detector(frames, conf=0.3, verbose=False, batch=batch)
            if len(detections) != n:
                raise RuntimeError('Detector returned wrong frame count')
            for i, detection in enumerate(detections):
                for box in detection.boxes:
                    hand, score = int(box.cls.item()), float(box.conf.item())
                    if hand not in (0, 1):
                        raise RuntimeError('Unexpected hand class')
                    if score > values['hand_scores'][i, hand]:
                        boxes[i, hand] = box.xyxy.cpu().numpy().reshape(4)
                        values['hand_scores'][i, hand] = score
                        values['detection_mask'][i, hand] = True
            pairs = [(i, h) for i in range(n) for h in range(2) if values['detection_mask'][i, h]]
            for start in range(0, len(pairs), batch):
                chunk = pairs[start:start + batch]
                predictions = pipe.predict_with_bboxes([frames[i] for i, h in chunk],
                              np.array([boxes[i, h] for i, h in chunk]), [h for i, h in chunk])
                if len(predictions) != len(chunk):
                    raise RuntimeError('Wrong number of hand fits')
                for (i, h), prediction in zip(chunk, predictions):
                    data = prediction.get('wilor_preds') if prediction else None
                    keys = ('global_orient', 'hand_pose', 'pred_keypoints_3d')
                    if data is None or not all(k in data and np.isfinite(data[k]).all() for k in keys):
                        continue
                    for key in keys:
                        values[key][i, h] = np.asarray(data[key]).reshape(values[key][i, h].shape)
                    values['valid_mask'][i, h] = True
            parts.append(values)
    finally:
        cap.release()
    if not parts:
        raise RuntimeError('Video decoded zero frames')
    arrays = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    n = len(arrays['global_orient'])
    meta['frame_count_validation'] = validate_frame_count(path, advertised, n)
    if meta['fps'] <= 0:
        raise RuntimeError('Invalid source FPS')
    np.savez_compressed(out / (path.stem + '_hands.npz'), **arrays)
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    meta.update(frames=n, source_sha256=checksum.hexdigest(),
                detected=arrays['detection_mask'].sum(axis=0).tolist(),
                valid=arrays['valid_mask'].sum(axis=0).tolist())
    (out / (path.stem + '.json')).write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(dict(video=path.name, **meta)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU inference must run in a Slurm job')
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = WiLorHandPose3dEstimationPipeline(device=torch.device('cuda'), dtype=torch.float16, verbose=False)
    for record in json.loads(args.records.read_text()):
        for video in record['videos'].values():
            process(pipe, Path(video), args.out, args.batch_size)


if __name__ == '__main__':
    main()
