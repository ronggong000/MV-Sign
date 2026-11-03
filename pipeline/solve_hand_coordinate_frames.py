"""Derive geometric finger frames from the SMPL-X rest skeleton, on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


FINGERS = ('index', 'middle', 'pinky', 'ring', 'thumb')
TIP_IDS = {'left': (4933, 5058, 5286, 5169, 5361),
           'right': (7669, 7794, 8022, 7905, 8079)}


def unit(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError('Degenerate geometric direction')
    return v / norm


def fk(joints, parents, local_rotations):
    """SMPL-X zero-rest-rotation skeleton, with local pose rotations."""
    positions = np.empty_like(joints)
    global_rotations = np.empty_like(local_rotations)
    for i, parent in enumerate(parents):
        if parent < 0:
            positions[i] = joints[i]
            global_rotations[i] = local_rotations[i]
        else:
            positions[i] = positions[parent] + global_rotations[parent] @ (joints[i] - joints[parent])
            global_rotations[i] = global_rotations[parent] @ local_rotations[i]
    return positions, global_rotations


def child_position(joint, child, tip, joints, parents, rotations):
    positions, global_rotations = fk(joints, parents, rotations)
    if child is not None:
        return positions[child], positions[joint]
    return positions[joint] + global_rotations[joint] @ (tip - joints[joint]), positions[joint]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('data/SMPLX_NEUTRAL.npz'))
    parser.add_argument('--out', type=Path, default=Path('paper_assets/hand_coordinate_frames_cpu'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.model, allow_pickle=True) as model:
        vertices = np.array(model['v_template'], dtype=np.float64)
        regressor = model['J_regressor']
        if regressor.ndim == 0:
            regressor = regressor.item()
        joints = np.asarray(regressor @ vertices, dtype=np.float64)
        parents = model['kintree_table'][0].astype(np.int64)
    parents[0] = -1
    assert joints.shape == (55, 3)
    down = -unit(joints[12] - joints[0])
    identities = np.broadcast_to(np.eye(3), (55, 3, 3)).copy()
    matrices = np.empty((2, 15, 3, 3))
    eulers = np.empty((2, 15, 3))
    normals = []
    records, trajectories = [], {}
    angles = np.deg2rad(np.linspace(0, 90, 19))
    rng = np.random.default_rng(20261006)
    for h, hand in enumerate(('left', 'right')):
        base, wrist = (25, 20) if h == 0 else (40, 21)
        palm_points = joints[[wrist, base, base + 3, base + 6, base + 9]]
        _, _, vh = np.linalg.svd(palm_points - palm_points.mean(axis=0))
        normal = vh[-1]
        if normal @ down < 0:
            normal = -normal
        normals.append(normal)
        for f, name in enumerate(FINGERS):
            tip = vertices[TIP_IDS[hand][f]]
            for k in range(3):
                joint = base + f * 3 + k
                child = joint + 1 if k < 2 else None
                endpoint = joints[child] if child is not None else tip
                direction = unit(endpoint - joints[joint])
                palmward = unit(normal - direction * (normal @ direction))
                x = unit(np.cross(direction, palmward))
                z = unit(np.cross(x, direction))
                frame = np.column_stack((x, direction, z))
                matrices[h, f * 3 + k] = frame
                # Match fuse_poses: stored [x,y,z], matrix = Ry @ Rx @ Rz.
                angles_zxy = Rotation.from_matrix(frame).as_euler('zxy')
                eulers[h, f * 3 + k] = angles_zxy[[1, 2, 0]]
                path = []
                for angle in angles:
                    rotations = identities.copy()
                    rotations[joint] = frame @ Rotation.from_rotvec([angle, 0, 0]).as_matrix() @ frame.T
                    point, pivot = child_position(joint, child, tip, joints, parents, rotations)
                    path.append(point - pivot)
                path = np.array(path)
                length = np.linalg.norm(endpoint - joints[joint])
                expected = length * (np.cos(angles)[:, None] * direction + np.sin(angles)[:, None] * palmward)
                test_rotations = Rotation.from_rotvec(rng.normal(size=(100, 3))).as_matrix()
                in_target = frame.T @ test_rotations @ frame
                restored = frame @ in_target @ frame.T
                # Independent axis probes: X bends, Y twists, Z splays.
                probe_errors = []
                a = np.deg2rad(10)
                for axis in range(3):
                    rv = np.zeros(3)
                    rv[axis] = a
                    rotations = identities.copy()
                    rotations[joint] = frame @ Rotation.from_rotvec(rv).as_matrix() @ frame.T
                    point, pivot = child_position(joint, child, tip, joints, parents, rotations)
                    target = (length * (np.cos(a) * direction + np.sin(a) * palmward),
                              length * direction,
                              length * (np.cos(a) * direction - np.sin(a) * x))[axis]
                    probe_errors.append(float(np.linalg.norm(point - pivot - target)))
                record = dict(hand=hand, finger=name, segment=k + 1, pose_joint=joint,
                              matrix_index=f * 3 + k, tip_vertex=TIP_IDS[hand][f],
                              determinant=float(np.linalg.det(frame)),
                              orthogonality_error=float(np.max(np.abs(frame.T @ frame - np.eye(3)))),
                              roundtrip_error=float(np.max(np.abs(restored - test_rotations))),
                              trajectory_error_m=float(np.max(np.linalg.norm(path - expected, axis=1))),
                              lateral_drift_m=float(np.max(np.abs(path @ x))),
                              positive_bend_palmward_m=float((path[1] - path[0]) @ palmward),
                              positive_bend_body_down_m=float((path[1] - path[0]) @ down),
                              axis_probe_errors_m=probe_errors)
                records.append(record)
                trajectories[hand, name, k] = (path, joints[joint], x, direction, palmward)
    assert all(r['trajectory_error_m'] < 1e-10 and r['lateral_drift_m'] < 1e-10
               and r['roundtrip_error'] < 1e-10 and r['orthogonality_error'] < 1e-10
               and r['positive_bend_palmward_m'] > 0 and r['positive_bend_body_down_m'] > 0
               and max(r['axis_probe_errors_m']) < 1e-10 for r in records)
    reconstructed = Rotation.from_euler('zxy', eulers.reshape(-1, 3)[:, [2, 0, 1]]).as_matrix()
    euler_error = float(np.max(np.abs(reconstructed - matrices.reshape(-1, 3, 3))))
    assert euler_error < 1e-10
    np.savez(args.out / 'hand_coordinate_frames.npz', T_target_to_smplx=matrices,
             euler_xyz_zxy_rad=eulers, rest_joints=joints, parents=parents,
             palmward_normals=np.array(normals), finger_names=np.array(FINGERS))
    report = dict(model=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  device='cpu', beta='zero', hand_mean='not applied; rest template',
                  convention='T columns = target axes in SMPL-X rest coordinates; R_target=T.T@R_local@T',
                  axes='X positive palmward flexion; Y distal bone direction; Z=X cross Y',
                  palm_sign='palm-plane normal oriented toward negative pelvis-to-neck direction',
                  euler_reconstruction_error=euler_error, joints=records)
    (args.out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    draw(args.out, joints, trajectories)
    lines = ['# CPU geometric hand coordinate frames', '',
             'These are geometric candidate frames, not experimentally measured anatomical joint axes.',
             'No pipeline scripts have been changed. Both hands and all 15 pose joints per hand are included.', '',
             '## Definition', '', report['convention'], '', report['axes'], '',
             'Palm plane: least-squares plane through wrist and four non-thumb MCP joints.',
             'Its sign is chosen toward body-down in the rest template, as requested; this is a convention, not a palm-surface measurement.',
             'Each joint Y follows its outgoing bone; terminal joints use the official SMPL-X fingertip vertex.',
             'Rest skeleton is J_regressor @ v_template, beta=0, with no hand mean.',
             'The same finger can have slightly different frames at its three joints.', '',
             '## CPU verification', '',
             '30 joints: single-joint FK sweeps 0..90 degrees, positive flexion sign, constant radius,',
             'zero lateral drift, independent X/Y/Z probes, rotation round trips and Euler reconstruction.',
             f'Max trajectory error: {max(r["trajectory_error_m"] for r in records):.3e} m.',
             f'Max lateral drift: {max(r["lateral_drift_m"] for r in records):.3e} m.',
             f'Max rotation roundtrip error: {max(r["roundtrip_error"] for r in records):.3e}.',
             f'Euler reconstruction error: {euler_error:.3e}.', '',
             'These verify the stated geometric definition and transform direction; they do not establish biological correctness.',
             'Thumb opposition cannot be inferred uniquely from the rest skeleton. Its frame here uses the same palm-plane convention.',
             'Inspect the thumb visually before using it for repair; the current pipeline only repairs four non-thumb fingers.',
             'Mesh skinning, pose corrective deformation, personalised beta and pipeline repair behavior are not tested by skeleton FK.', '',
             '## Mapping', '', '| Finger | Local hand indices | Left pose indices | Right pose indices |',
             '|---|---|---|---|']
    for i, name in enumerate(FINGERS):
        lines.append(f'| {name} | {i*3}..{i*3+2} | {25+i*3}..{27+i*3} | {40+i*3}..{42+i*3} |')
    lines += ['', '## Reproduce', '', '```bash',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \\',
              '/home/smuk0019/ar85_scratch2/singyu/miniconda3/envs/osx/bin/python \\',
              'scripts/solve_hand_coordinate_frames_cpu.py', '```', '']
    (args.out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'out': str(args.out), 'validated_joints': len(records),
                      'max_trajectory_error_m': max(r['trajectory_error_m'] for r in records),
                      'euler_error': euler_error}, indent=2))


def draw(out, joints, trajectories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 8))
    colors = dict(zip(FINGERS, ('#0072b2', '#009e73', '#cc79a7', '#e69f00', '#d55e00')))
    for h, hand in enumerate(('left', 'right')):
        ax = fig.add_subplot(1, 2, h + 1, projection='3d')
        all_points = []
        for name in FINGERS:
            for k in range(3):
                path, pivot, x, d, p = trajectories[hand, name, k]
                curve = (path + pivot) * 1000
                all_points.extend(curve)
                ax.plot(*curve.T, color=colors[name], label=name if k == 0 else None)
                ax.plot(*np.stack((pivot, pivot + path[0])).T * 1000, color=colors[name], alpha=0.45)
                ax.quiver(*(pivot * 1000), *(x * 12), color=colors[name], linewidth=1)
        points = np.array(all_points)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = np.max(np.ptp(points, axis=0)) / 2 * 1.15
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set(xlabel='SMPL-X X (mm)', ylabel='SMPL-X Y (mm)', zlabel='SMPL-X Z (mm)',
               title=f'{hand}: isolated joint +X sweeps, 0..90 deg')
        ax.view_init(elev=25, azim=-65 if h == 0 else -115)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'joint_flexion_trajectories.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
