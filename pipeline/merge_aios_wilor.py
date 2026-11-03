import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation, Slerp

import os

# # --- 全局加载 Relaxed Hand Pose 参考数据 ---
# # 这个数据来自Blender插件，用于将flat hand pose转换为relaxed pose
# try:
#     HAND_POSES_PATH = Path(__file__).parent / 'smplx_handposes.npz'
#     with np.load(HAND_POSES_PATH, allow_pickle=True) as data:
#         hand_poses_data = data["hand_poses"].item()
#         (left_hand_pose_relaxed, right_hand_pose_relaxed) = hand_poses_data["relaxed"]
#         # 将左右手参考姿态合并并 reshape
#         RELAXED_HAND_REF = np.concatenate(
#             (left_hand_pose_relaxed, right_hand_pose_relaxed)
#         ).reshape(2, 15, 3) # Shape: (2, 15, 3), Index 0:Left, 1:Right
#     print("✅ Successfully loaded relaxed hand pose reference from 'data/smplx_handposes.npz'")
# except FileNotFoundError:
#     print(f"❌ ERROR: Cannot find 'data/smplx_handposes.npz'.")
#     print("Please create a 'data' folder next to this script and place the file inside.")
#     RELAXED_HAND_REF = None
# except Exception as e:
#     print(f"❌ ERROR: Failed to load or process 'smplx_handposes.npz': {e}")
#     RELAXED_HAND_REF = None

# --- SMPL-X 骨骼结构定义 (保持不变) ---
PARENTS = {
    16: 13, 13: 9, 18: 16, 20: 18, 17: 14, 14: 9, 19: 17, 21: 19,
    9: 6, 6: 3, 3: 0, 12: 9, 15: 12,
}
def interpolate_pose_sequence(full_pose_sequence: np.ndarray) -> np.ndarray:
    """
    对单只手的完整姿态序列 (手腕+手指) 进行插值 (V3 - 处理开头和结尾的空洞)。
    full_pose_sequence: (N, 16, 3) 形状的轴角数据
    """
    num_frames, num_joints, _ = full_pose_sequence.shape
    
    # 1. 创建有效性 Mask: 以手腕旋转是否为零作为判断依据
    is_valid = np.abs(full_pose_sequence[:, 0, :]).sum(axis=1) > 1e-6
    if is_valid.all():
        return full_pose_sequence # 无需插值

    # 2. 将所有姿态转换为旋转矩阵
    rot_mats = Rotation.from_rotvec(full_pose_sequence.reshape(-1, 3)).as_matrix().reshape(num_frames, num_joints, 3, 3)
    
    interpolated_mats = rot_mats.copy()
    
    # 3. 找到所有无效帧的连续块
    invalid_indices = np.where(~is_valid)[0]
    gaps = np.split(invalid_indices, np.where(np.diff(invalid_indices) != 1)[0] + 1)
    
    # 4. 遍历每个关节，分别进行插值或填充
    for joint_idx in range(num_joints):
        joint_rot_mats = rot_mats[:, joint_idx, :, :]
        
        for gap in gaps:
            if len(gap) == 0:
                continue
            
            start_frame = gap[0] - 1
            end_frame = gap[-1] + 1
            
            # --- START: 核心修正逻辑 ---
            # 情况一：空洞在中间，两端都有有效帧 -> 进行Slerp插值
            if start_frame >= 0 and end_frame < num_frames and is_valid[start_frame] and is_valid[end_frame]:
                key_rots = Rotation.from_matrix([
                    joint_rot_mats[start_frame],
                    joint_rot_mats[end_frame]
                ])
                key_times = [start_frame, end_frame]
                slerp = Slerp(key_times, key_rots)
                interp_rots = slerp(gap)
                interpolated_mats[gap, joint_idx, :, :] = interp_rots.as_matrix()
            
            # 情况二：空洞在视频开头，只有后面有有效帧 -> 向前填充
            elif start_frame < 0 and end_frame < num_frames and is_valid[end_frame]:
                fill_value = joint_rot_mats[end_frame]
                interpolated_mats[gap, joint_idx, :, :] = fill_value

            # 情况三：空洞在视频结尾，只有前面有有效帧 -> 向后填充
            elif end_frame >= num_frames and start_frame >= 0 and is_valid[start_frame]:
                fill_value = joint_rot_mats[start_frame]
                interpolated_mats[gap, joint_idx, :, :] = fill_value
            # --- END: 核心修正逻辑 ---

    # 5. 将插值后的旋转矩阵序列转换回轴角格式
    return Rotation.from_matrix(interpolated_mats.reshape(-1, 3, 3)).as_rotvec().reshape(num_frames, num_joints, 3)

def interpolate_keypoints_sequence(keypoints: np.ndarray, is_valid: np.ndarray) -> np.ndarray:
    """
    对3D关节点序列进行线性插值，以填充无效帧 (V1 - 直接坐标插值)。
    keypoints: (N, num_joints, 3) 形状的坐标数据
    is_valid: (N,) 形状的布尔掩码，标记有效帧
    """
    if is_valid.all():
        return keypoints  # 无需插值

    num_frames, num_joints, _ = keypoints.shape
    interpolated_kps = keypoints.copy()

    # 找到所有无效帧的连续块
    invalid_indices = np.where(~is_valid)[0]
    gaps = np.split(invalid_indices, np.where(np.diff(invalid_indices) != 1)[0] + 1)

    for gap in gaps:
        if len(gap) == 0:
            continue
        
        start_frame = gap[0] - 1
        end_frame = gap[-1] + 1

        # 情况一：空洞在中间 -> 进行线性插值 (LERP)
        if start_frame >= 0 and end_frame < num_frames and is_valid[start_frame] and is_valid[end_frame]:
            start_kps = keypoints[start_frame]  # Shape: (num_joints, 3)
            end_kps = keypoints[end_frame]      # Shape: (num_joints, 3)
            
            for i, frame_idx in enumerate(gap):
                alpha = (i + 1) / (len(gap) + 1)
                interpolated_kps[frame_idx] = (1 - alpha) * start_kps + alpha * end_kps

        # 情况二：空洞在视频开头 -> 向前填充
        elif start_frame < 0 and end_frame < num_frames and is_valid[end_frame]:
            fill_value = keypoints[end_frame]
            interpolated_kps[gap] = fill_value

        # 情况三：空洞在视频结尾 -> 向后填充
        elif end_frame >= num_frames and start_frame >= 0 and is_valid[start_frame]:
            fill_value = keypoints[start_frame]
            interpolated_kps[gap] = fill_value
            
    return interpolated_kps
def forward_kinematics(poses):
    num_frames = poses.shape[0]
    local_rot_mats = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(num_frames, -1, 3, 3)
    global_rot_mats = np.zeros_like(local_rot_mats)
    global_rot_mats[:, 0] = local_rot_mats[:, 0]
    num_joints = poses.shape[1]
    
    for i in range(1, num_joints):
        parent_idx = PARENTS.get(i)
        if parent_idx is not None:
            global_rot_mats[:, i] = np.einsum('fij,fjk->fik', global_rot_mats[:, parent_idx], local_rot_mats[:, i])
        else:
            global_rot_mats[:, i] = local_rot_mats[:, i]
    return global_rot_mats

def merge_files(aios_path, wilor_path, output_path):
    #if RELAXED_HAND_REF is None: return
    try:
        aios_data, wilor_data = np.load(aios_path), np.load(wilor_path)
    except Exception as e:
        print(f"\nError loading files for {os.path.basename(aios_path)}: {e}")
        return

    final_data = dict(aios_data)
    aios_poses = aios_data['poses']
    num_frames = aios_poses.shape[0]
    # --- 功能1: 检查并跳过帧数不足的文件 ---
    if num_frames < 8:
        # 加一个换行符 \n 防止被 tqdm 进度条覆盖
        print(f"\n[!] Skipping {os.path.basename(aios_path)}: Frame count ({num_frames}) is less than the minimum of 8.")
        return  # 直接退出函数，不处理此文件
    # --- 准备原始数据 ---
    wilor_wrist_poses_aa = wilor_data['global_orient']
    wilor_finger_poses_aa = wilor_data['hand_pose'].reshape(num_frames, 2, 15, 3)
    wilor_kp3d_raw = wilor_data.get('pred_keypoints_3d')

    if wilor_wrist_poses_aa.shape[0] != num_frames:
        print(f"\nWarning: Frame count mismatch for {os.path.basename(aios_path)}. Skipping file.")
        return

    # =============================================================================
    # --- 新逻辑核心: 数据质量验证与重建 ---
    # =============================================================================
    print(f"      -> Verifying and reconstructing hand data for {os.path.basename(aios_path)}...")

    # --- 步骤 1: 计算所有需要的“度量指标” ---
    
    # 指标1: 存在性 (Presence)
    is_present_left = np.abs(wilor_wrist_poses_aa[:, 0, :]).sum(axis=1) > 1e-6
    is_present_right = np.abs(wilor_wrist_poses_aa[:, 1, :]).sum(axis=1) > 1e-6

    # 准备旋转对象
    aios_global_rotations = forward_kinematics(aios_poses)
    aios_wrist_rot_left = Rotation.from_matrix(aios_global_rotations[:, 20])
    aios_wrist_rot_right = Rotation.from_matrix(aios_global_rotations[:, 21])
    wilor_wrist_rot_left = Rotation.from_rotvec(wilor_wrist_poses_aa[:, 0, :])
    wilor_wrist_rot_right = Rotation.from_rotvec(wilor_wrist_poses_aa[:, 1, :])
    
    # 指标2: 姿态差异 (Orientation Difference)
    diff_rot_left = aios_wrist_rot_left.inv() * wilor_wrist_rot_left
    diff_rot_right = aios_wrist_rot_right.inv() * wilor_wrist_rot_right
    diff_angle_left = diff_rot_left.magnitude()
    diff_angle_right = diff_rot_right.magnitude()

    # 指标3 & 4: 自身角速度 (Angular Velocity)
    # 通过计算连续帧之间的角度距离来估算
    vel_wilor_left = (wilor_wrist_rot_left[:-1].inv() * wilor_wrist_rot_left[1:]).magnitude()
    vel_wilor_right = (wilor_wrist_rot_right[:-1].inv() * wilor_wrist_rot_right[1:]).magnitude()
    vel_aios_left = (aios_wrist_rot_left[:-1].inv() * aios_wrist_rot_left[1:]).magnitude()
    vel_aios_right = (aios_wrist_rot_right[:-1].inv() * aios_wrist_rot_right[1:]).magnitude()
    # 补齐第一帧的速度（假设为0）
    vel_wilor_left = np.insert(vel_wilor_left, 0, 0)
    vel_wilor_right = np.insert(vel_wilor_right, 0, 0)
    vel_aios_left = np.insert(vel_aios_left, 0, 0)
    vel_aios_right = np.insert(vel_aios_right, 0, 0)

    # --- 步骤 2: 定义“损坏”规则并生成“损坏掩码” ---
    # 阈值设定 (可根据需要调整)
    DIFF_ANGLE_THRESHOLD = np.pi / 1.5  # 120度
    VEL_WILOR_THRESHOLD = np.pi / 3.0   # 60度/帧
    VEL_AIOS_THRESHOLD = np.pi / 12.0   # 15度/帧

    # 应用规则
    is_damaged_left = ~is_present_left | \
                      (diff_angle_left > DIFF_ANGLE_THRESHOLD) | \
                      ((vel_wilor_left > VEL_WILOR_THRESHOLD) & (vel_aios_left < VEL_AIOS_THRESHOLD))
    
    is_damaged_right = ~is_present_right | \
                       (diff_angle_right > DIFF_ANGLE_THRESHOLD) | \
                       ((vel_wilor_right > VEL_WILOR_THRESHOLD) & (vel_aios_right < VEL_AIOS_THRESHOLD))

    # --- 步骤 3: (可选但推荐) 扩展损坏区域 ---
    # 为了安全起见，我们将损坏区域向前后各扩展一帧
    from scipy.ndimage import binary_dilation
    
    damaged_indices_left = np.where(is_damaged_left)[0]
    damaged_indices_right = np.where(is_damaged_right)[0]
    
    if len(damaged_indices_left) > 0 and len(damaged_indices_left) < num_frames:
        is_damaged_left = binary_dilation(is_damaged_left, iterations=1)
        print(f"        Identified and expanded {is_damaged_left.sum()} bad frames for LEFT hand.")
        
    if len(damaged_indices_right) > 0 and len(damaged_indices_right) < num_frames:
        is_damaged_right = binary_dilation(is_damaged_right, iterations=1)
        print(f"        Identified and expanded {is_damaged_right.sum()} bad frames for RIGHT hand.")

    # --- 步骤 4: 插值重建 ---
    # 生成最终的有效性掩码
    final_is_valid_left = ~is_damaged_left
    final_is_valid_right = ~is_damaged_right
    has_valid_left = bool(final_is_valid_left.any())
    has_valid_right = bool(final_is_valid_right.any())
    
    # A: 重建 Poses
    # 将被判定为损坏的帧的poses数据清零，以便插值函数识别
    wilor_wrist_poses_aa_clean = wilor_wrist_poses_aa.copy()
    wilor_wrist_poses_aa_clean[is_damaged_left, 0, :] = 0
    wilor_wrist_poses_aa_clean[is_damaged_right, 1, :] = 0
    
    wilor_full_flat_pose_to_interp = np.concatenate([
        wilor_wrist_poses_aa_clean[:, :, np.newaxis, :],
        wilor_finger_poses_aa
    ], axis=2)
    
    # 现在传入插值函数的序列，其“空洞”是我们主动制造出来的损坏区域
    left_hand_interp = interpolate_pose_sequence(wilor_full_flat_pose_to_interp[:, 0, :, :])
    right_hand_interp = interpolate_pose_sequence(wilor_full_flat_pose_to_interp[:, 1, :, :])
    wilor_full_flat_interp = np.stack([left_hand_interp, right_hand_interp], axis=1)

    # B: 重建 3D Keypoints
    wilor_kp3d_interp = None
    if wilor_kp3d_raw is not None:
        # 这里我们直接使用最终的有效性掩码进行插值，因为插值函数内部会处理
        wilor_kp3d_left_interp = interpolate_keypoints_sequence(wilor_kp3d_raw[:, 0, :, :], final_is_valid_left)
        wilor_kp3d_right_interp = interpolate_keypoints_sequence(wilor_kp3d_raw[:, 1, :, :], final_is_valid_right)

    # =============================================================================
    # --- 后续合并逻辑 (使用重建后的干净数据) ---
    # =============================================================================
    new_poses = aios_poses.copy()

    # 3. 转换手指姿态
    interp_wrist_pose = wilor_full_flat_interp[:, :, 0, :]
    interp_finger_pose = wilor_full_flat_interp[:, :, 1:, :]
    #relaxed_ref_expanded = np.tile(RELAXED_HAND_REF, (num_frames, 1, 1, 1))
    #relax_based_finger_pose = interp_finger_pose - relaxed_ref_expanded
    relax_based_finger_pose = interp_finger_pose
    if has_valid_left:
        new_poses[:, 25:40, :] = relax_based_finger_pose[:, 0, :, :]
    else:
        print("        LEFT hand has no valid WiLoR frames; keeping AiOS wrist/fingers.")
    if has_valid_right:
        new_poses[:, 40:55, :] = relax_based_finger_pose[:, 1, :, :]
    else:
        print("        RIGHT hand has no valid WiLoR frames; keeping AiOS wrist/fingers.")

    # 4. 替换手腕姿态
    target_rot_left_wrist = Rotation.from_rotvec(interp_wrist_pose[:, 0, :])
    target_rot_right_wrist = Rotation.from_rotvec(interp_wrist_pose[:, 1, :])
    new_local_rot_left_wrist = Rotation.from_matrix(aios_global_rotations[:, 18]).inv() * target_rot_left_wrist
    new_local_rot_right_wrist = Rotation.from_matrix(aios_global_rotations[:, 19]).inv() * target_rot_right_wrist
    if has_valid_left:
        new_poses[:, 20, :] = new_local_rot_left_wrist.as_rotvec()
    if has_valid_right:
        new_poses[:, 21, :] = new_local_rot_right_wrist.as_rotvec()
    final_data['poses'] = new_poses
    final_data['wilor_hand_valid_mask'] = np.stack([final_is_valid_left, final_is_valid_right], axis=1)
    
    # 6. 合并 3D Hand Keypoints
    if 'smpl_kp3d' in final_data and wilor_kp3d_raw is not None:
        new_kp3d = final_data['smpl_kp3d'].copy()
        aios_wrist_left = new_kp3d[:, 20, :]
        aios_wrist_right = new_kp3d[:, 21, :]
        wilor_wrist_left = wilor_kp3d_left_interp[:, 0, :]
        wilor_wrist_right = wilor_kp3d_right_interp[:, 0, :]
        offset_left = aios_wrist_left - wilor_wrist_left
        offset_right = aios_wrist_right - wilor_wrist_right

        left_hand_map = [
            (slice(1, 5), [37, 38, 39, 66]), (slice(5, 9), [25, 26, 27, 67]),
            (slice(9, 13), [28, 29, 30, 68]), (slice(13, 17), [34, 35, 36, 69]),
            (slice(17, 21), [31, 32, 33, 70]),
        ]
        right_hand_map = [
            (slice(1, 5), [52, 53, 54, 71]), (slice(5, 9), [40, 41, 42, 72]),
            (slice(9, 13), [43, 44, 45, 73]), (slice(13, 17), [49, 50, 51, 74]),
            (slice(17, 21), [46, 47, 48, 75]),
        ]

        if has_valid_left:
            for source_slice, dest_indices in left_hand_map:
                source_kps = wilor_kp3d_left_interp[:, source_slice, :]
                aligned_kps = source_kps + offset_left[:, np.newaxis, :]
                new_kp3d[:, dest_indices, :] = aligned_kps
        
        if has_valid_right:
            for source_slice, dest_indices in right_hand_map:
                source_kps = wilor_kp3d_right_interp[:, source_slice, :]
                aligned_kps = source_kps + offset_right[:, np.newaxis, :]
                new_kp3d[:, dest_indices, :] = aligned_kps

        final_data['smpl_kp3d'] = new_kp3d
    
    # 7. 最终保存
    np.savez_compressed(output_path, **final_data)


def main():
    parser = argparse.ArgumentParser(description="Merge AiOS body poses with WiLoR hand poses.")
    parser.add_argument('--aios_dir', type=str, required=True, help="Directory containing AiOS's _amass.npz files.")
    parser.add_argument('--wilor_dir', type=str, required=True, help="Directory containing WiLoR's _hands.npz files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the merged _aioswilor.npz files.")
    args = parser.parse_args()

    aios_dir = Path(args.aios_dir)
    wilor_dir = Path(args.wilor_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Finding matching files...")
    # 这个文件匹配逻辑已经很健壮，能处理你的文件名
    aios_files = {p.stem.replace('_amass', ''): p for p in aios_dir.glob('*_amass.npz')}
    wilor_files = {p.stem.replace('_hands', ''): p for p in wilor_dir.glob('*_hands.npz')}

    common_basenames = sorted(list(aios_files.keys() & wilor_files.keys()))
    print(f"Found {len(common_basenames)} total matching file pairs.")
    
    files_to_process = []
    for basename in common_basenames:
        output_filename = f"{basename}_aioswilor.npz"
        output_path = output_dir / output_filename
        if not output_path.exists():
            files_to_process.append(basename)

    if not files_to_process:
        print("All matched files have already been merged. Nothing to do.")
        return

    print(f"{len(common_basenames) - len(files_to_process)} files already merged. Processing {len(files_to_process)} new files.")

    for basename in tqdm(common_basenames, desc="Merging files"):
        aios_path = aios_files[basename]
        wilor_path = wilor_files[basename]
        output_filename = f"{basename}_aioswilor.npz"
        output_path = output_dir / output_filename
        merge_files(aios_path, wilor_path, output_path)

    print("\nMerging process complete.")
    print(f"Merged files saved to: {output_dir}")

if __name__ == '__main__':
    main()
