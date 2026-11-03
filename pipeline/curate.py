"""Fuse two views of saved AiOS/WiLoR outputs into one final SMPL-X sequence."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent


def ensure_hand_frames():
    data = json.loads((HERE / 'hand_coordinate_frames.json').read_text())
    matrices = np.asarray(data['T_target_to_smplx'], dtype=np.float64)
    if matrices.shape != (2, 15, 3, 3) or not np.allclose(matrices.swapaxes(-1, -2) @ matrices, np.eye(3), atol=1e-8):
        raise ValueError('Invalid hand coordinate constants')
    if not np.allclose(np.linalg.det(matrices), 1.0, atol=1e-8):
        raise ValueError('Hand frames must be right handed')
    path = HERE / 'hand_coordinate_frames.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored['T_target_to_smplx'], matrices) or stored['finger_names'].tolist() != data['finger_names']:
                raise ValueError('Existing hand coordinate cache differs from release constants')
    else:
        np.savez(path, T_target_to_smplx=matrices, finger_names=np.asarray(data['finger_names']))


def run(args):
    ensure_hand_frames()
    import align_views as align
    import merge_aios_wilor as merge
    from release_hooks import require_wavelet, install_trace_hooks, validate_npz
    require_wavelet({'wavelet_version': '1.8.0'})
    if args.out.exists() and not args.overwrite:
        raise FileExistsError('Output exists; use --overwrite explicitly')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = {'idle_runs_inclusive': []}
    restore = install_trace_hooks(align, trace)
    try:
        with tempfile.TemporaryDirectory(prefix='mvsign_', dir=args.out.parent) as tmp:
            work = Path(tmp)
            merged = {}
            for view in ('front', 'side'):
                body, hand = getattr(args, 'aios_' + view), getattr(args, 'wilor_' + view)
                with np.load(body, allow_pickle=False) as data, np.load(hand, allow_pickle=False) as hands:
                    n = len(data['poses'])
                    if n < 8 or len(hands['global_orient']) != n:
                        raise ValueError(view + ': unequal source lengths or fewer than 8 frames')
                dest = work / (view + '_merged.npz')
                merge.merge_files(body, hand, dest)
                if not dest.exists():
                    raise RuntimeError(view + ': merge did not produce an output')
                with np.load(dest, allow_pickle=False) as data:
                    values = {k: data[k].copy() for k in data.files}
                values.update(source_frame_index=np.arange(n, dtype=np.int64),
                              source_padding_mask=np.zeros(n, dtype=bool),
                              synthetic_bridge_mask=np.zeros(n, dtype=bool),
                              mocap_framerate=np.asarray(args.fps))
                merged[view] = values
            length = max(len(x['poses']) for x in merged.values())
            keys = ('poses', 'trans', 'cam_trans', 'smpl_kp3d', 'smplx_expr', 'smplx_joint_proj',
                    'wilor_hand_valid_mask', 'source_frame_index', 'source_padding_mask', 'synthetic_bridge_mask')
            for view, data in merged.items():
                n = len(data['poses'])
                for key in keys:
                    if key in data:
                        if len(data[key]) != n:
                            raise ValueError('Stale temporal field: ' + key)
                        data[key] = np.concatenate([data[key], np.repeat(data[key][-1:], length - n, axis=0)])
                data['source_padding_mask'][n:] = True
                np.savez_compressed(work / (view + '_merged.npz'), **data)
            align.process_pair(work / 'front_merged.npz', work / 'side_merged.npz',
                               work / 'front.npz', work / 'side.npz', False,
                               enable_hand_fix=True, disable_idle_trim=False)
            with np.load(work / 'front.npz', allow_pickle=False) as data:
                final = {k: data[k].copy() for k in data.files}
            fingerprint = hashlib.sha256(b''.join((HERE / name).read_bytes() for name in
                                      ('merge_aios_wilor.py', 'align_views.py', 'fuse_poses.py', 'hand_coordinate_frames.json'))).hexdigest()
            final.update(view='front', sample_id=args.sample_id, split=args.split,
                         pipeline_fingerprint=fingerprint,
                         body_smoothing_backend=trace['body_smoothing']['backend'],
                         body_smoothing_version=trace['body_smoothing']['version'],
                         body_smoothing_wavelet=trace['body_smoothing']['wavelet'],
                         body_smoothing_level=trace['body_smoothing']['level'],
                         smpl_kp3d_semantics='fused tracking keypoints; not recomputed FK after pose repair')
            np.savez_compressed(work / 'release.npz', **final)
            frames = validate_npz(work / 'release.npz')
            # Replace only after output validation, leaving no partial result on failure.
            (work / 'release.npz').replace(args.out)
            args.out.with_suffix('.json').write_text(json.dumps({'sample_id': args.sample_id, 'split': args.split,
                'frames': frames, 'fps': args.fps, 'trace': trace, 'pipeline_fingerprint': fingerprint,
                'output': 'one final sequence in frontal body coordinate representation'}, indent=2) + '\n')
            if args.keep_side:
                with np.load(work / 'side.npz', allow_pickle=False) as data:
                    np.savez_compressed(args.out.with_name(args.out.stem + '_side_debug.npz'), **dict(data))
            print('FINAL_SEQUENCE', args.out, 'frames', frames)
    finally:
        restore()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('aios-front', 'wilor-front', 'aios-side', 'wilor-side'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fps', type=float, required=True, help='Actual shared decoded video FPS; no resampling is performed')
    p.add_argument('--sample-id', default='sample')
    p.add_argument('--split', default='unspecified')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--keep-side', action='store_true', help='Optional diagnostic output; not an additional training sample')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if not 0 < args.fps <= 240:
        raise ValueError('Invalid frame rate')
    run(args)
