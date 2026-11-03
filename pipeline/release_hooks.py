"""Production validation and temporal-bookkeeping hooks, extracted unchanged."""
import inspect
class CuratedOut(ValueError):
    """A clip excluded by the existing curation policy, not a runtime failure."""

def require_wavelet(config=None):
    try:
        import pywt
        pywt.Wavelet('coif3')
    except Exception as error:
        raise RuntimeError('PyWavelets is required; refusing silent Savitzky-Golay fallback') from error
    if config and config.get('wavelet_version') and pywt.__version__ != config['wavelet_version']:
        raise RuntimeError('PyWavelets version differs from frozen configuration')
    return dict(backend='PyWavelets', version=pywt.__version__)

def validate_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        n = len(data['poses'])
        if n < 1 or data['poses'].shape != (n, 55, 3):
            raise ValueError('Invalid poses or empty curated sequence')
        for key in ('poses', 'trans', 'cam_trans', 'smpl_kp3d', 'betas'):
            if key not in data or not np.isfinite(data[key]).all():
                raise ValueError('Missing/nonfinite ' + key)
        for key in ('trans', 'cam_trans', 'smpl_kp3d', 'smplx_expr', 'smplx_joint_proj',
                    'wilor_hand_valid_mask', 'source_frame_index', 'synthetic_bridge_mask', 'source_padding_mask'):
            if key in data and data[key].shape[0] != n:
                raise ValueError('Stale temporal field: ' + key)
        return n

def install_trace_hooks(align, trace):
    import numpy as np
    original_find = align.find_temporal_offset_on_aligned_data
    original_idle = align.detect_idle_v3
    original_drop = align.drop_idle_segments_with_bridge
    original_sync = align.create_synced_data_with_padding
    original_smooth = align.fix_body_rotations_spikes_and_smooth

    def smooth(*args, **kwargs):
        backend = require_wavelet()
        bound = inspect.signature(original_smooth).bind(*args, **kwargs)
        bound.apply_defaults()
        if not bound.arguments['wavelet_denoise']:
            raise RuntimeError('Expected body wavelet denoising to be enabled')
        bound.arguments['fallback_savgol'] = False
        bound.arguments['verbose'] = True
        trace['body_smoothing'] = dict(backend, wavelet=bound.arguments['wavelet_name'],
                                       level=bound.arguments['wavelet_level'],
                                       joints=list(bound.arguments['joints_body']), fallback_allowed=False)
        return original_smooth(*bound.args, **bound.kwargs)

    def find(*args, **kwargs):
        value = original_find(*args, **kwargs)
        trace['temporal_offset'] = int(value)
        return value

    def idle(*args, **kwargs):
        value = original_idle(*args, **kwargs)
        trace['idle_runs_inclusive'] = [[int(a), int(b)] for a, b in value]
        return value

    def sync(data, offset, target_len):
        result = original_sync(data, offset, target_len)
        # A static 10-coefficient beta vector is not a time series in a 10-frame clip.
        if 'betas' in data and data['betas'].ndim == 1:
            result['betas'] = data['betas'].copy()
        indices = np.arange(target_len) - offset
        clipped = np.clip(indices, 0, len(data['poses']) - 1)
        result['source_padding_mask'] = data['source_padding_mask'][clipped] | (indices != clipped)
        return result

    def drop(front, side, runs, bridge_frames=4):
        n = len(front['poses'])
        merged = []
        for a, b in sorted(runs):
            a, b = max(0, int(a)), min(n - 1, int(b))
            if a > b:
                continue
            if merged and a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        keep, cursor = [], 0
        for a, b in merged:
            if cursor < a:
                keep.append((cursor, a))
            cursor = b + 1
        if cursor < n:
            keep.append((cursor, n))
        if not keep:
            raise CuratedOut('non-sign removal excluded the entire sequence')
        result = original_drop(front, side, runs, bridge_frames=bridge_frames)
        for source, dest in zip((front, side), result):
            indices, synthetic, padding, valid = [], [], [], []
            for i, (a, b) in enumerate(keep):
                indices.extend(source['source_frame_index'][a:b])
                synthetic.extend([False] * (b - a))
                padding.extend(source['source_padding_mask'][a:b])
                valid.extend(source['wilor_hand_valid_mask'][a:b])
                if i < len(keep) - 1:
                    indices.extend([-1] * bridge_frames)
                    synthetic.extend([True] * bridge_frames)
                    padding.extend([False] * bridge_frames)
                    valid.extend([[False, False]] * bridge_frames)
            dest['source_frame_index'] = np.asarray(indices, dtype=np.int64)
            dest['synthetic_bridge_mask'] = np.asarray(synthetic, dtype=bool)
            dest['source_padding_mask'] = np.asarray(padding, dtype=bool)
            dest['wilor_hand_valid_mask'] = np.asarray(valid, dtype=bool)
            assert len(indices) == len(dest['poses'])
        return result

    align.find_temporal_offset_on_aligned_data = find
    align.detect_idle_v3 = idle
    align.create_synced_data_with_padding = sync
    align.drop_idle_segments_with_bridge = drop
    align.fix_body_rotations_spikes_and_smooth = smooth
    return lambda: restore_hooks(align, original_find, original_idle, original_sync, original_drop, original_smooth)

def restore_hooks(align, find, idle, sync, drop, smooth=None):
    align.find_temporal_offset_on_aligned_data = find
    align.detect_idle_v3 = idle
    align.create_synced_data_with_padding = sync
    align.drop_idle_segments_with_bridge = drop
    if smooth is not None:
        align.fix_body_rotations_spikes_and_smooth = smooth
