import numpy as np
import argparse
from scipy.spatial.transform import Rotation
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection # 用于绘制相机锥体
except ImportError:
    plt = None
    Axes3D = None
    Poly3DCollection = None
import torch
from fuse_poses import batch_euler2mat_zxy, batch_mat2euler_zxy, rodrigues_to_rotation_matrix, rotation_matrix_to_rodrigues


# --- 骨架连接定义，用于可视化 (保持不变) ---
SMPL_BODY_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (18, 20), (19, 21), # 修正：(16, 18), (17, 19) 是手部，可能没有
]

def align_views_with_pelvis_correction(front_data: dict, side_data: dict) -> dict:
    """
    (V2 - 稳定版) 执行三阶段空间对齐。
    此版本通过计算平均旋转来消除抖动。
    """
    print("Running 3-stage spatial alignment (Rotation -> Centroid -> Pelvis)...")
    poses_front = front_data['poses']
    poses_side = side_data['poses']
    kp3d_front = front_data['smpl_kp3d']
    kp3d_side = side_data['smpl_kp3d']
    trans_front = front_data['trans']
    trans_side = side_data['trans']
    num_frames = poses_front.shape[0]
    
    # --- 步骤 1: (不变) 先计算出每一帧独立的旋转修正 ---
    per_frame_R_corrections = np.zeros((num_frames, 3, 3))
    for i in range(num_frames):
        R_front = Rotation.from_rotvec(poses_front[i, 0]).as_matrix()
        R_side = Rotation.from_rotvec(poses_side[i, 0]).as_matrix()
        per_frame_R_corrections[i] = R_front @ R_side.T

    # --- (核心修正) 步骤 2: 计算单一的、稳定的“平均旋转” ---
    # 通过对所有逐帧旋转矩阵求和并进行SVD分解，可以找到最优的平均旋转
    print("    -> Stabilizing rotation by calculating the mean rotation...")
    R_sum = np.sum(per_frame_R_corrections, axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_correction_avg = U @ Vt
    # 确保它是一个右手坐标系的旋转矩阵
    if np.linalg.det(R_correction_avg) < 0:
        Vt[-1, :] *= -1
        R_correction_avg = U @ Vt

    # --- 步骤 3: 使用这个稳定的平均旋转进行后续所有计算 ---
    kp3d_side_rotated = np.zeros_like(kp3d_side)
    for i in range(num_frames):
        # (修改) 使用平均旋转
        kp3d_side_rotated[i] = (R_correction_avg @ kp3d_side[i].T).T
    
    global_centroid_front = kp3d_front.mean(axis=(0, 1))
    global_centroid_side_rotated = kp3d_side_rotated.mean(axis=(0, 1))
    global_t_correction = global_centroid_front - global_centroid_side_rotated
    
    pelvis_offsets_per_frame = []
    for i in range(num_frames):
        pelvis_pos_front = kp3d_front[i, 0]
        # (修改) 使用平均旋转
        pelvis_pos_side_coarse_aligned = (R_correction_avg @ kp3d_side[i, 0]) + (R_correction_avg @ trans_side[i]) + global_t_correction
        pelvis_offsets_per_frame.append(pelvis_pos_front - pelvis_pos_side_coarse_aligned)
        
    final_pelvis_offset = np.mean(pelvis_offsets_per_frame, axis=0)
    final_t_correction = global_t_correction + final_pelvis_offset
    
    aligned_poses_side = np.copy(poses_side)
    aligned_trans_side = np.copy(trans_side)
    for i in range(num_frames):
        R_side = Rotation.from_rotvec(poses_side[i, 0]).as_matrix()
        # (修改) 使用平均旋转
        new_R_side = R_correction_avg @ R_side
        aligned_poses_side[i, 0] = Rotation.from_matrix(new_R_side).as_rotvec()
        
        # (修改) 使用平均旋转
        aligned_trans_side[i] = (R_correction_avg @ trans_side[i]) + final_t_correction
        
    aligned_side_data_dict = dict(side_data)
    aligned_side_data_dict['poses'] = aligned_poses_side
    aligned_side_data_dict['trans'] = aligned_trans_side
    
    # (修改) 使用平均旋转
    R_corr_expanded = np.expand_dims(R_correction_avg, axis=0)
    kp3d_side_aligned = (R_corr_expanded @ kp3d_side.transpose(0, 2, 1)).transpose(0, 2, 1) + aligned_trans_side[:, np.newaxis, :]
    aligned_side_data_dict['smpl_kp3d_aligned'] = kp3d_side_aligned
    
    print("Spatial alignment pass complete (with rotation stabilization).")
    return aligned_side_data_dict

# --- 时间对齐函数 (核心功能，保持不变) ---
def find_temporal_offset_on_aligned_data(kp3d_front, kp3d_side_aligned, search_range=5):
    """在空间对齐的数据上寻找最佳时间偏移"""
    print(f"\n--- Stage 4: Temporal Alignment Search ---")
    print(f"Searching for best offset in range [{-search_range}, {search_range}]...")
    best_offset = 0
    min_error = float('inf')
    for offset in range(-search_range, search_range + 1):
        if offset >= 0:
            segment_front = kp3d_front[offset:]
            segment_side = kp3d_side_aligned[:-offset if offset != 0 else None]
        else:
            segment_front = kp3d_front[:offset]
            segment_side = kp3d_side_aligned[-offset:]
        min_len = min(len(segment_front), len(segment_side))
        if min_len == 0:
            continue
        segment_front = segment_front[:min_len]
        segment_side = segment_side[:min_len]
        error = np.mean(np.linalg.norm(segment_front[:, :22, :] - segment_side[:, :22, :], axis=2))
        #print(f"  Testing offset {offset: >2}: MPJPE = {error:.4f}")
        if error < min_error:
            min_error = error
            best_offset = offset
    print(f"Temporal alignment found! Best offset: {best_offset} frames (min MPJPE: {min_error:.4f})")
    return best_offset

# --- 应用时间偏移和填充的辅助函数 (核心功能，保持不变) ---
def create_synced_data_with_padding(data_dict, offset, target_len):
    """根据偏移量创建同步后的数据，并进行边缘填充"""
    print(f"\nCreating new time-synced data with offset {offset} and target length {target_len}...")
    synced_dict = {}
    if 'poses' not in data_dict or data_dict['poses'] is None: 
        # 如果没有'poses'键，可能不是时序数据，直接返回副本
        return dict(data_dict) 
        
    original_len = len(data_dict['poses'])

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.shape and value.shape[0] == original_len:
            new_array = np.zeros((target_len, *value.shape[1:]), dtype=value.dtype)
            for i in range(target_len):
                src_idx = i - offset
                src_idx = max(0, min(src_idx, original_len - 1))
                new_array[i] = value[src_idx]
            synced_dict[key] = new_array
        else:
            synced_dict[key] = value
    return synced_dict

# --- (更新) Debug可视化函数，支持交互式和更清晰的相机表示 ---
def debug_visualization_interactive(front_data, side_data, K, title_suffix, frame_index=0):
    """
    创建并显示一个交互式3D可视化图像，用于调试对齐结果。
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for debug visualization. Run without --debug or install matplotlib.")
    print(f"Generating interactive debug visualization for frame {frame_index}...")

    # 提取指定帧的数据
    kp3d_front = front_data['smpl_kp3d'][frame_index]
    kp3d_side = side_data['smpl_kp3d_aligned'][frame_index]
    P_front = front_data['P_canonical_cam']
    P_side = side_data['P_virtual_cam']

    # 1. 骨架根关节对齐到原点
    # 找到所有关键点的最大/最小范围，用于设置坐标轴限制
    all_kp = np.vstack([kp3d_front, kp3d_side])
    
    # 选择统一原点（例如 front pelvis）
    origin = kp3d_front[0]  # 若 kp3d 是局部，那用 (kp3d_front[0] + trans_front[frame]) 更合理


    # 2. 从投影矩阵 P 中提取相机中心和方向
    def get_camera_extrinsics(P_matrix):
        # P = K @ [R|t]
        # inv(K) @ P = [R|t]
        # cam_center = -R.T @ t
        # cam_direction = R.T @ [0,0,1] (局部Z轴在世界坐标系中的方向)

        P = np.array(P_matrix)
        K_mat = P[:, :3] # 假设P矩阵的左上角3x3是K
        
        # 提取K矩阵 (如果P是K@Extrinsics)
        # 注意：这里我们假设 P = K @ Extrinsics。
        # 如果 K_mat 不是纯粹的上三角矩阵，可能需要更复杂的分解。
        # 实际上 K 是一个固定的矩阵，我们直接使用传入的 K。
        
        # K_inv @ P = [R|t]
        K_inv = np.linalg.inv(K)
        extrinsics_world_to_cam = K_inv @ P_matrix # 这是一个 3x4 矩阵
        R_cam_to_world = extrinsics_world_to_cam[:, :3] # 这是从世界到相机的旋转
        t_cam_to_world = extrinsics_world_to_cam[:, 3] # 这是从世界到相机的平移

        # 相机在世界坐标系中的位置 (C = -R_world_to_cam_T @ t_world_to_cam)
        # 这里R_cam_to_world是世界坐标到相机坐标的变换，所以它的逆是相机到世界的变换
        R_world_to_cam_inv = R_cam_to_world.T # 这是从相机坐标到世界坐标的旋转
        camera_center_world = -R_world_to_cam_inv @ t_cam_to_world

        # 相机在世界坐标系中的Z轴方向 (通常是相机朝向)
        camera_direction_world = R_world_to_cam_inv[:, 2] # 相机局部Z轴在世界中的方向

        return camera_center_world, camera_direction_world

    center_front, dir_front = get_camera_extrinsics(P_front)
    center_side, dir_side = get_camera_extrinsics(P_side)
    kp3d_front_centered = kp3d_front - origin
    kp3d_side_centered  = kp3d_side  - origin

    center_front_c = center_front - origin
    center_side_c  = center_side  - origin

    # 3. 创建3D绘图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制骨架函数
    def plot_skeleton(ax, joints, connections, color, label):
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=50, label=label + ' Joints')
        for start_j, end_j in connections:
            if start_j < len(joints) and end_j < len(joints): # 确保索引有效
                ax.plot(*zip(joints[start_j], joints[end_j]), color=color, linewidth=2)
        # 标记根关节
        ax.scatter(joints[0, 0], joints[0, 1], joints[0, 2], color=color, s=200, marker='o', edgecolors='black', linewidth=1, label=label + ' Root')

    # 绘制骨架
    plot_skeleton(ax, kp3d_front_centered, SMPL_BODY_CONNECTIONS, 'blue', 'Front View')
    plot_skeleton(ax, kp3d_side_centered, SMPL_BODY_CONNECTIONS, 'red', 'Side View Aligned')

    # 绘制相机 (位置用点，朝向用更长的箭头)
    camera_length = 0.5 # 箭头的长度
    ax.scatter(*center_front_c, color='darkblue', marker='X', s=300, label='Front Camera Position')
    ax.quiver(*center_front, *dir_front, color='darkblue', length=camera_length, linewidth=3, arrow_length_ratio=0.1, label='Front Camera Direction')
    
    ax.scatter(*center_side_c, color='darkred', marker='X', s=300, label='Side Camera Position')
    ax.quiver(*center_side, *dir_side, color='darkred', length=camera_length, linewidth=3, arrow_length_ratio=0.1, label='Side Camera Direction')

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Interactive Alignment Debug - {title_suffix}')
    ax.legend()

    # 统一坐标轴比例，确保物体不会变形
    # 计算所有骨架和相机的最大范围
    all_points = np.vstack([kp3d_front_centered, kp3d_side_centered, center_front, center_side])
    max_range = np.max(all_points.max(axis=0) - all_points.min(axis=0)) / 2.0
    mid_x = (all_points[:, 0].min() + all_points[:, 0].max()) / 2.0
    mid_y = (all_points[:, 1].min() + all_points[:, 1].max()) / 2.0
    mid_z = (all_points[:, 2].min() + all_points[:, 2].max()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 确保3D显示比例一致
    ax.set_box_aspect([1,1,1]) # 设置X, Y, Z轴的比例

    # 调整初始视角
    ax.view_init(elev=20, azim=-60)
    
    plt.tight_layout()
    plt.show() # 显示交互式窗口
    plt.close(fig) # 关闭当前图表，准备下一个
# ==== 在文件顶部附近放一个统一的列表 ====
TIME_SERIES_KEYS = ['poses', 'smpl_kp3d', 'trans', 'cam_trans']  # 你还用到别的就加进来

def slice_by_mask_inplace(data_dict: dict, mask: np.ndarray):
    """对 data_dict 里所有首维= T 的时序数组同时做掩码裁剪。"""
    T = None
    # 先确定基准 T（以 poses 优先；没有就找第一个时序键）
    for k in TIME_SERIES_KEYS:
        if k in data_dict and isinstance(data_dict[k], np.ndarray):
            T = data_dict[k].shape[0]
            break
    if T is None:
        return
    assert mask.ndim == 1 and mask.shape[0] == T, f"mask length {mask.shape[0]} != T={T}"
    for k, v in list(data_dict.items()):
        if isinstance(v, np.ndarray) and v.shape and v.shape[0] == T:
            data_dict[k] = v[mask].copy()

def force_sync_len_inplace(front_dict: dict, side_dict: dict):
    """
    把 front / side 两路所有时序键统一裁到同一个长度（取能找到的最小长度）。
    防止某个键长度没同步造成越界。
    """
    lens = []
    for d in (front_dict, side_dict):
        for k in TIME_SERIES_KEYS:
            if k in d and isinstance(d[k], np.ndarray) and d[k].shape:
                lens.append(d[k].shape[0])
    if not lens:
        return
    T = min(lens)
    for d in (front_dict, side_dict):
        for k in TIME_SERIES_KEYS:
            if k in d and isinstance(d[k], np.ndarray) and d[k].shape and d[k].shape[0] >= T:
                d[k] = d[k][:T].copy()
# ===== SO(3) helpers =====
def so3_geodesic_angle(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """
    Ra,Rb: [...,3,3]
    return: angle in radians [...,]
    """
    Rt = Ra.transpose(-1, -2) @ Rb
    tr = Rt[..., 0, 0] + Rt[..., 1, 1] + Rt[..., 2, 2]
    c = (tr - 1.0) * 0.5
    c = torch.clamp(c, -1.0, 1.0)
    return torch.acos(c)

def so3_log(R: torch.Tensor) -> torch.Tensor:
    """
    R: [...,3,3] -> omega: [...,3]  (axis-angle vector)
    这里直接复用你已有的 rotation_matrix_to_rodrigues，做个包装。
    """
    shape = R.shape[:-2]
    w = rotation_matrix_to_rodrigues(R.reshape(-1,3,3)).reshape(*shape, 3)
    return w

def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """
    omega: [...,3] -> R: [...,3,3]
    复用 rodrigues_to_rotation_matrix。
    """
    shape = omega.shape[:-1]
    R = rodrigues_to_rotation_matrix(omega.reshape(-1,3)).reshape(*shape, 3, 3)
    return R

def so3_slerp(Ra: torch.Tensor, Rb: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Ra,Rb: [...,3,3], tau: [...,] in [0,1]
    Return R(tau).
    """
    Rab = Ra.transpose(-1, -2) @ Rb
    xi = so3_log(Rab)                              # [...,3]
    Rtau = Ra @ so3_exp(xi * tau.unsqueeze(-1))    # broadcast tau
    return Rtau
def boolean_runs(mask_1d: np.ndarray):
    """
    输入: 1D 布尔数组（长度=T）
    输出: [(start, end), ...]  其中 end 为包含式索引（即 [start..end]）
    """
    runs = []
    Tlen = mask_1d.shape[0]
    i = 0
    while i < Tlen:
        if not mask_1d[i]:
            i += 1
            continue
        start = i
        while i + 1 < Tlen and mask_1d[i + 1]:
            i += 1
        end = i
        runs.append((start, end))
        i += 1
    return runs
def _rot_heat_from_poses(poses: np.ndarray, joint_ids: list[int], win: int = 3) -> np.ndarray:
    """
    poses: (T,55,3) Rodrigues
    返回该关节组每帧的“旋转热度”（so(3)测地步长的中位数），长度 T。
    """
    import torch, numpy as np
    T = poses.shape[0]
    if T <= 1: return np.zeros(T, dtype=np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aa = torch.as_tensor(poses[:, joint_ids, :], dtype=torch.float32, device=device)  # [T,J,3]
    R  = rodrigues_to_rotation_matrix(aa.reshape(-1,3)).reshape(T, -1, 3,3)          # [T,J,3,3]
    d  = so3_geodesic_angle(R[:-1], R[1:]).median(dim=1).values.detach().cpu().numpy()  # [T-1]
    heat = np.zeros(T, dtype=np.float32); heat[1:] = d
    if win and win>1:
        k = win//2; pad = np.pad(heat, (k,k), mode='edge'); heat = np.convolve(pad, np.ones(win)/win, 'valid')
    return heat

def _robust_unit(x: np.ndarray, k_low=0.5, k_high=2.0) -> np.ndarray:
    """median±k*MAD 归一化到[0,1]（截断）"""
    med = np.median(x); mad = np.median(np.abs(x-med)) + 1e-8
    lo = med - k_low*mad; hi = med + k_high*mad
    if hi <= lo: return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

def boolean_runs_1d(mask: np.ndarray) -> list[tuple[int,int]]:
    """True runs -> [(s,e)] 包含端点"""
    runs=[]; T=len(mask); i=0
    while i<T:
        if not mask[i]: i+=1; continue
        s=i
        while i+1<T and mask[i+1]: i+=1
        runs.append((s,i)); i+=1
    return runs
def detect_idle_v3(
    front_poses: np.ndarray,            # (T,55,3)
    front_kp3d:  np.ndarray,            # (T,N,3) 用于高度判断
    save_debug_png: str = "idle_detect_debug.png",
    # 三组关节
    joints_mouth   : list[int] = [22],
    joints_wrists  : list[int] = [20,21],
    joints_fingers : list[int] = list(range(25,55)),
    # 腕-肘高度参数（两侧都要低于各自肘部-γ·肩宽）
    idx_L_sh=16, idx_R_sh=17, idx_L_el=18, idx_R_el=19, idx_L_wr=20, idx_R_wr=21,
    height_gamma: float = 0.30,
    # 死线/阈值
    fingers_dead_eps: float = 1e-3,     # 指尖近零判据（在归一化前）
    fingers_dead_min: int   = 18,       # 连续这么多帧近零 -> 手指置 0
    norm_idle_th:   float   = 0.30,     # 任一路 < 0.30 即“低活动”
    min_idle_len:   int     = 24,       # “延迟起报”阈值：低活跃累计 ≥ 24 帧才开始记为 idle
    lookahead_when_resume: int = 8,     # 三路再次活跃时，末端向后吞 8 帧
    smooth_win_heat: int    = 3,        # 热度计算平滑窗口（用于最小可删长度判定）
    verbose: bool = True
) -> list[tuple[int,int]]:
    """
    返回 idle_runs=[(s,e)]（基于 front 判定；删除时用同样片段裁掉 side）
    关键语义：
      - 不是一低就切，而是：自上一次 high_all 之后，连续低活动帧累计 ≥ min_idle_len 才“起报”，
        将该段从 (起始低帧 + min_idle_len - 1) 作为真正的删除起点。
      - 仅当整段长度 > (lookahead_when_resume + smooth_win_heat) 才删，否则不剪。
    """
    import numpy as np, matplotlib.pyplot as plt

    T = front_poses.shape[0]
    if T == 0:
        return []

    # ---------- 1) 三组热度 + 归一化 ----------
    h_m = _rot_heat_from_poses(front_poses, joints_mouth,   win=smooth_win_heat)
    h_w = _rot_heat_from_poses(front_poses, joints_wrists,  win=smooth_win_heat)
    h_f = _rot_heat_from_poses(front_poses, joints_fingers, win=smooth_win_heat)
    n_m = _robust_unit(h_m); n_w = _robust_unit(h_w); n_f = _robust_unit(h_f)

    # ---------- 2) 两腕同时“放下” -> wrists 置 0（自动判断 Y 轴朝向，且用肩宽尺度） ----------
    Y = front_kp3d[..., 1].astype(np.float32)
    sh_vec = front_kp3d[:, idx_L_sh, :] - front_kp3d[:, idx_R_sh, :]
    sh_w = np.median(np.linalg.norm(sh_vec, axis=1)) + 1e-8

    w_minus_s = (
        np.median(Y[:, idx_L_wr] - Y[:, idx_L_sh]) +
        np.median(Y[:, idx_R_wr] - Y[:, idx_R_sh])
    ) / 2.0
    y_down_positive = (w_minus_s > 0)   # True: Y 向下为正；False: Y 向上为正
    margin = height_gamma * sh_w
    if y_down_positive:
        L_down = Y[:, idx_L_wr] > (Y[:, idx_L_el] + margin)
        R_down = Y[:, idx_R_wr] > (Y[:, idx_R_el] + margin)
    else:
        L_down = Y[:, idx_L_wr] < (Y[:, idx_L_el] - margin)
        R_down = Y[:, idx_R_wr] < (Y[:, idx_R_el] - margin)
    both_down = L_down & R_down
    n_w[both_down] = 0.0  # 两腕都明显低于各自肘 -> wrists_norm 直接视作 0

    # ---------- 3) 手指长期近零 -> 置 0 ----------
    nf0 = (h_f < fingers_dead_eps).astype(np.int32)
    long_zero = np.zeros(T, dtype=bool)
    run = 0
    for t in range(T):
        if nf0[t] == 1:
            run += 1
        else:
            if run >= fingers_dead_min:
                long_zero[t-run:t] = True
            run = 0
    if run >= fingers_dead_min:
        long_zero[T-run:T] = True
    n_f[long_zero] = 0.0

    # ---------- 4) 活跃 / 低活动 判定 ----------
    low_any  = (n_m < norm_idle_th) | (n_w < norm_idle_th) | (n_f < norm_idle_th)
    high_all = (n_m >= norm_idle_th) & (n_w >= norm_idle_th) & (n_f >= norm_idle_th)

    # ---------- 5) 延迟起报 + 最小可删长度 ----------
    # 先找到所有“低活动连续段”（raw）
    def _boolean_runs(mask: np.ndarray) -> list[tuple[int,int]]:
        runs = []
        i = 0
        while i < len(mask):
            if not mask[i]:
                i += 1; continue
            s = i
            while i+1 < len(mask) and mask[i+1]:
                i += 1
            e = i
            runs.append((s,e))
            i += 1
        return runs

    raw_runs = _boolean_runs(low_any)

    # 对每个 raw 低活动段 (s,e)，只有在“从上一段高活跃之后，连续低活动累计 ≥ min_idle_len”
    # 才确认为 idle，并且将起点推进到 s' = s + min_idle_len - 1。
    confirmed = []
    last_active = -1
    # 构造“上一次高活跃”数组，便于度量从 last_active 到当前的低活动累计
    high_idx = np.where(high_all)[0]

    for (s,e) in raw_runs:
        # 找到 s 之前最后一个高活跃帧
        prev_high = high_idx[high_idx < s]
        last_active = prev_high[-1] if prev_high.size > 0 else -1
        # 计算自上一次高活跃后的低活动累计长度
        low_len_from_last_high = (e - max(last_active+1, s) + 1)
        if low_len_from_last_high >= min_idle_len:
            s_prime = max(s, last_active + min_idle_len)  # s + min_idle_len - 1 的等价安全写法
            # 从 e 往后找“恢复完全活跃”的首帧，以便后面吃 lookahead
            t = e + 1
            add = 0
            while t < T and add < lookahead_when_resume and not high_all[t]:
                t += 1; add += 1
            e_prime = min(e + add, T-1)

            # 只有当片段长度 > (lookahead_when_resume + smooth_win_heat) 才剪
            if (e_prime - s_prime + 1) > (lookahead_when_resume + smooth_win_heat):
                confirmed.append((s_prime, e_prime))

    # 合并相邻/重叠
    if confirmed:
        merged = [confirmed[0]]
        for s,e in confirmed[1:]:
            ps,pe = merged[-1]
            if s <= pe + 1:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s,e))
        confirmed = merged

    # ---------- 6) 可视化 ----------
    if save_debug_png:
        x = np.arange(T)
        plt.figure(figsize=(16,6))
        plt.plot(x, n_m, label="mouth_norm")
        plt.plot(x, n_w, label="wrists_norm")
        plt.plot(x, n_f, label="fingers_norm")
        plt.hlines(norm_idle_th, 0, T-1, colors='gray', linestyles='--', alpha=0.5, label="idle threshold")
        for s,e in confirmed:
            plt.axvspan(s, e, color='lightgray', alpha=0.3)
        plt.title(f"Idle detection (ANY < {norm_idle_th:.2f} treated as low; start after {min_idle_len} lows)")
        plt.xlabel("Frame"); plt.ylabel("Normalized heats [0..1]")
        plt.legend(); plt.tight_layout(); plt.savefig(save_debug_png, dpi=150); plt.close()
        if verbose:
            print(f"[IdleDetect-v3] saved: {save_debug_png}")
            print(f"[IdleDetect-v3] runs={confirmed}")

    return confirmed

def _ease_inout_cos(n: int) -> np.ndarray:
    """n 个插值点的 ease-in-out，范围 (0,1)，不含端点（用于桥接中间帧）"""
    # t = 1..n, 归一化到 (0,1)
    t = (np.arange(1, n+1, dtype=np.float32)) / (n+1)
    return 0.5 - 0.5*np.cos(np.pi*t)

def _bridge_rotations(Ra: torch.Tensor, Rb: torch.Tensor, n_bridge: int) -> torch.Tensor:
    """在 Ra->Rb 之间插 n_bridge 帧，cos-ease 的 SLERP。输入输出都是 [J,3,3] 批。"""
    if n_bridge <= 0:
        return Ra.new_zeros((0, Ra.shape[0], 3,3))
    taus = torch.as_tensor(_ease_inout_cos(n_bridge), dtype=Ra.dtype, device=Ra.device)  # [n]
    Ra_b = Ra.unsqueeze(0).expand(n_bridge, -1, -1, -1)  # [n,J,3,3]
    Rb_b = Rb.unsqueeze(0).expand(n_bridge, -1, -1, -1)
    # 批 slerp：先展平 J 维
    n,J = n_bridge, Ra.shape[0]
    Rtau = so3_slerp(Ra_b.reshape(n, J, 3,3), Rb_b.reshape(n, J, 3,3), taus.view(n,1).expand(n,J)).reshape(n, J, 3,3)
    return Rtau
def drop_idle_segments_with_bridge(
    front_dict: dict, side_dict: dict, idle_runs: list[tuple[int,int]], bridge_frames: int = 4
):
    """
    删除 front/side 中 idle_runs；对所有“首维为 T”的键同步裁剪；
    段间桥接：
      - poses: SO(3) SLERP（每个关节的旋转）
      - 其它时序键（如 smpl_kp3d、trans 等）: 逐元素线性插值
    返回 (front_out, side_out)。
    """
    import numpy as np, torch

    def _merge_and_clip(runs, T):
        if not runs: return []
        rs=[]
        for s,e in sorted(runs):
            s=max(0,int(s)); e=min(T-1,int(e))
            if s<=e: rs.append((s,e))
        if not rs: return []
        merged=[rs[0]]
        for s,e in rs[1:]:
            ps,pe=merged[-1]
            if s<=pe+1: merged[-1]=(ps,max(pe,e))
            else: merged.append((s,e))
        return merged

    def _compute_keep(runs, T):
        if not runs: return [(0,T-1)]
        keep=[]; cur=0
        for s,e in runs:
            if cur<s: keep.append((cur,s-1))
            cur=e+1
        if cur<T: keep.append((cur,T-1))
        return keep

    def _is_time_series(arr, T):
        return isinstance(arr, np.ndarray) and arr.ndim>=1 and arr.shape[0]==T

    # ---------- SO(3) 工具 ----------
    def _bridge_rotations(Ra: torch.Tensor, Rb: torch.Tensor, n: int) -> torch.Tensor:
        """
        Ra,Rb: [J,3,3],  n: 过渡帧数 -> [n,J,3,3]
        """
        if n<=0: return Ra.unsqueeze(0).expand(0,*Ra.shape)
        def so3_log(R):  # [...,3,3] -> [...,3]
            return rotation_matrix_to_rodrigues(R.reshape(-1,3,3)).reshape(*R.shape[:-2],3)
        def so3_exp(w):  # [...,3] -> [...,3,3]
            return rodrigues_to_rotation_matrix(w.reshape(-1,3)).reshape(*w.shape[:-1],3,3)
        device=Ra.device
        taus=torch.linspace(1.0/(n+1.0), n/(n+1.0), n, device=device)  # (n,)
        Rab=Ra.transpose(1,2)@Rb   # [J,3,3]
        xi =so3_log(Rab)           # [J,3]
        mids=[]
        for t in taus:
            mids.append((Ra@so3_exp(xi*t)).unsqueeze(0))
        return torch.cat(mids,dim=0)  # [n,J,3,3]

    # ---------- 通用线性插值（用于非旋转时序键） ----------
    def _lerp_frames(A_last: np.ndarray, B_first: np.ndarray, n: int) -> np.ndarray:
        """
        A_last, B_first: 形状相同 (...), 产生 n 帧线性插值 -> (n, ...)
        """
        if n<=0:
            return A_last.reshape((0,)+A_last.shape)
        alphas = np.linspace(1.0/(n+1.0), n/(n+1.0), n, dtype=np.float32).reshape((n,)+ (1,)*A_last.ndim)
        return (1.0 - alphas) * A_last[None, ...] + alphas * B_first[None, ...]

    def _concat_one(din: dict) -> dict:
        T0 = int(din['poses'].shape[0])
        # 找出所有“时序键”
        time_keys=[k for k,v in din.items() if _is_time_series(v, T0)]
        # 先按 keep 切段
        parts={k:[] for k in time_keys}
        for (s,e) in keep:
            for k in time_keys:
                parts[k].append(din[k][s:e+1].copy())

        # 段间桥接：poses 用 SO(3)，其它时序键用线性插值
        if len(keep)>=2 and bridge_frames>0:
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 先处理 poses（SO(3)）
            if 'poses' in parts:
                poses_bridged=[]
                for i in range(len(keep)-1):
                    A = parts['poses'][i]            # (Ta,55,3)
                    poses_bridged.append(A)
                    Ra_aa = A[-1]                    # (55,3)
                    Rb_aa = din['poses'][ keep[i+1][0] ]  # (55,3)
                    J = Ra_aa.shape[0]
                    Ra = rodrigues_to_rotation_matrix(torch.as_tensor(Ra_aa, dtype=torch.float32, device=device)).reshape(J,3,3)
                    Rb = rodrigues_to_rotation_matrix(torch.as_tensor(Rb_aa, dtype=torch.float32, device=device)).reshape(J,3,3)
                    Rmid = _bridge_rotations(Ra, Rb, bridge_frames)   # [n,J,3,3]
                    aa_mid = rotation_matrix_to_rodrigues(Rmid.reshape(-1,3,3)).reshape(bridge_frames,J,3).cpu().numpy()
                    poses_bridged.append(aa_mid)
                poses_bridged.append(parts['poses'][-1])
                parts['poses']=poses_bridged

            # 其它 time_keys 线性桥接
            for k in time_keys:
                if k == 'poses': 
                    continue
                k_bridged=[]
                for i in range(len(keep)-1):
                    A = parts[k][i]                 # (Ta, ...rest)
                    k_bridged.append(A)
                    A_last = A[-1]                  # (...rest)
                    B_first = din[k][ keep[i+1][0] ]   # (...rest)
                    k_mid = _lerp_frames(A_last, B_first, bridge_frames)  # (n, ...rest)
                    k_bridged.append(k_mid)
                k_bridged.append(parts[k][-1])
                parts[k]=k_bridged

        # 从空字典开始装配，避免把老数组“带回去”
        dout={}
        new_T=None
        for k in din.keys():
            if k in parts:
                dout[k]=np.concatenate(parts[k],axis=0) if parts[k] else din[k][:0]
                if k=='poses': new_T=dout[k].shape[0]
            else:
                dout[k]=din[k]

        # 长度对齐到 poses
        if new_T is not None:
            for k in time_keys:
                if dout[k].shape[0]!=new_T:
                    dout[k]=dout[k][:new_T].copy()
        return dout

    # ===== 主流程 =====
    T_all = int(front_dict['poses'].shape[0])
    if not idle_runs:
        return front_dict, side_dict

    idle_runs = _merge_and_clip(idle_runs, T_all)
    keep      = _compute_keep(idle_runs, T_all)

    if len(keep)==0:
        # 全删
        def _empty_like(d):
            out={}
            for k,v in d.items():
                if isinstance(v, np.ndarray) and v.ndim>=1 and v.shape[0]==T_all:
                    out[k]=v[:0]
                else:
                    out[k]=v
            return out
        f_out=_empty_like(front_dict); s_out=_empty_like(side_dict)
        print(f"[Idle-v3] removed {T_all} frames; new T=0")
        return f_out, s_out

    f_out=_concat_one(front_dict)
    s_out=_concat_one(side_dict)
    print(f"[Idle-v3] removed {T_all - f_out['poses'].shape[0]} frames; new T={f_out['poses'].shape[0]}")
    print(f"[Idle-v3] poses old/new: {T_all} -> {f_out['poses'].shape[0]}")
    return f_out, s_out

def fix_body_rotations_spikes_and_smooth(
    front_poses_in: np.ndarray,
    side_poses_in: np.ndarray,
    joints_body: range = range(22),           # 0..21
    spike_deg_hi: float = 120.0,              # 判定“本视角尖峰”的高阈值（度）
    spike_deg_lo: float = 40.0,               # 另一视角“无尖峰”的低阈值（度）
    max_spike_len: int = 2,                   # 连续尖峰的最长长度
    wavelet_denoise: bool = True,
    wavelet_name: str = "db4",
    wavelet_level: int = 2,
    fallback_savgol: bool = True,             # 没有 pywt 时回退
    savgol_window: int = 7,                   # 奇数
    savgol_poly: int = 2,
    verbose: bool = True,
    verbose_detail: bool = True,               # <<< 新增
    joint_names: dict | None = None            # <<< 可选：给 0..21 命名（打印更友好）
):
    """
    输入/输出: 与 hand-fix 一致的 poses 形状 (T, 55, 3)；只修改 0..21 的旋转。
    流程: 转 R -> 尖峰检测(两视角互证) -> SO(3) 插值回填 -> (可选) 小波/回退平滑 -> 回写。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = front_poses_in.shape[0]
    assert front_poses_in.shape[1:] == side_poses_in.shape[1:] and front_poses_in.shape[2] == 3

    # numpy -> torch
    f_aa = torch.as_tensor(front_poses_in, dtype=torch.float32, device=device)   # [T,55,3]
    s_aa = torch.as_tensor(side_poses_in,  dtype=torch.float32, device=device)

    # 只取 body joints
    J = len(joints_body)
    idx = torch.tensor(list(joints_body), dtype=torch.long, device=device)       # [J]

    f_aa_b = f_aa[:, idx, :]             # [T,J,3]
    s_aa_b = s_aa[:, idx, :]

    # -> R
    f_R = rodrigues_to_rotation_matrix(f_aa_b.reshape(-1,3)).reshape(T, J, 3,3) # [T,J,3,3]
    s_R = rodrigues_to_rotation_matrix(s_aa_b.reshape(-1,3)).reshape(T, J, 3,3)

    # 帧间角速度（弧度）
    f_ang = so3_geodesic_angle(f_R[:-1], f_R[1:])   # [T-1, J]
    s_ang = so3_geodesic_angle(s_R[:-1], s_R[1:])   # [T-1, J]

    # 阈值（弧度）
    hi = np.deg2rad(spike_deg_hi)
    lo = np.deg2rad(spike_deg_lo)

    # 标记尖峰（本视角>hi，另一视角<=lo）且是“孤立的1-2帧”
    def mark_spikes(ang_main: torch.Tensor, ang_other: torch.Tensor) -> torch.Tensor:
        main_peak = (ang_main > hi) & (ang_other <= lo)          # [T-1,J]
        # 只保留短段（<= max_spike_len）
        marks = torch.zeros_like(main_peak, dtype=torch.bool)
        for j in range(J):
            arr = main_peak[:, j].detach().cpu().numpy().astype(np.int32)  # 0/1
            # 找连续段
            start = None
            for t in range(arr.shape[0]):
                if arr[t] == 1 and start is None:
                    start = t
                if (arr[t] == 0 and start is not None) or (t == arr.shape[0]-1 and start is not None):
                    end = t if arr[t]==0 else t+1    # [start, end)
                    if 1 <= (end - start) <= max_spike_len:
                        marks[start:end, j] = True
                    start = None
        return marks

    f_marks = mark_spikes(f_ang, s_ang)   # [T-1,J], 标记 t 表示 (t->t+1) 之间存在峰
    s_marks = mark_spikes(s_ang, f_ang)

    # 把“间隔上的标记”映射为“帧上的标记”（影响两端帧）
    def interval_to_frame_marks(marks: torch.Tensor) -> torch.Tensor:
        frame_m = torch.zeros((T, J), dtype=torch.bool, device=marks.device)
        frame_m[1:] |= marks
        frame_m[:-1] |= marks
        return frame_m

    f_bad = interval_to_frame_marks(f_marks)    # [T,J]
    s_bad = interval_to_frame_marks(s_marks)
    # === 详细日志（可选）===
    detail_log = {"front": {}, "side": {}}
    if verbose_detail:
        # joints_body 是原始的关节编号列表（长度 J）
        j_idx_list = list(joints_body)
        for j_local, j_global in enumerate(j_idx_list):
            name = (joint_names.get(j_global, f"J{j_global}") if joint_names else f"J{j_global}")
            f_runs = boolean_runs(f_bad[:, j_local].detach().cpu().numpy())
            s_runs = boolean_runs(s_bad[:, j_local].detach().cpu().numpy())
            if f_runs:
                detail_log["front"][j_global] = f_runs
                msg = ", ".join([f"{a:04d}-{b:04d}" if a != b else f"{a:04d}" for a,b in f_runs])
                print(f"[BodySmooth] front  {name:<6s} bad runs: {msg}")
            if s_runs:
                detail_log["side"][j_global] = s_runs
                msg = ", ".join([f"{a:04d}-{b:04d}" if a != b else f"{a:04d}" for a,b in s_runs])
                print(f"[BodySmooth] side   {name:<6s} bad runs: {msg}")
    # === 用 SO(3) 插值回填坏帧 ===
    def inpaint_rotations(R: torch.Tensor, bad: torch.Tensor) -> torch.Tensor:
        """
        R: [T,J,3,3], bad: [T,J] (True=需要回填)
        """
        R_out = R.clone()
        for j in range(J):
            bad_j = bad[:, j].detach().cpu().numpy().astype(bool)
            # 找连续坏段
            t = 0
            while t < T:
                if not bad_j[t]:
                    t += 1
                    continue
                # 坏段 [tL, tR)
                tL = t
                while t < T and bad_j[t]:
                    t += 1
                tR = t
                # 需要左右“好帧”各1个
                left = tL - 1
                right = tR
                if left < 0 or right >= T:
                    # 边界：无法双端插值，选择就近复制
                    if left >= 0:
                        R_out[tL:tR, j] = R_out[left, j].unsqueeze(0).expand(tR-tL, 3,3)
                    elif right < T:
                        R_out[tL:tR, j] = R_out[right, j].unsqueeze(0).expand(tR-tL, 3,3)
                    continue
                Ra = R_out[left, j].unsqueeze(0)          # [1,3,3]
                Rb = R_out[right, j].unsqueeze(0)
                n = tR - tL
                taus = torch.linspace(1/(n+1), n/(n+1), n, device=R.device)  # 均匀 tau
                # 批量 SLERP
                Ra_b = Ra.expand(n, 3,3)
                Rb_b = Rb.expand(n, 3,3)
                R_interp = so3_slerp(Ra_b, Rb_b, taus)     # [n,3,3]
                R_out[tL:tR, j] = R_interp
        return R_out

    f_R_fixed = inpaint_rotations(f_R, f_bad)
    s_R_fixed = inpaint_rotations(s_R, s_bad)

    # === 可选：小波降噪 ===
    if wavelet_denoise:
        try:
            import pywt  # type: ignore
            use_wavelet = True
        except Exception:
            use_wavelet = False

        def denoise_seq_R(R_seq: torch.Tensor) -> torch.Tensor:
            # R_seq: [T,J,3,3] -> ω: [T,J,3] （注意：log(R) 就是轴角）
            w = so3_log(R_seq.reshape(-1,3,3)).reshape(T, J, 3).detach().cpu().numpy()
            if use_wavelet:
                if verbose:
                    # 你若有阈值倍率，可一并打印（见下文“二-2 调强”）
                    print(f"[BodySmooth] Denoiser=Wavelet(name={wavelet_name}, level={wavelet_level})")
                w_d = np.empty_like(w)
                for j in range(J):
                    for k in range(3):
                        coeffs = pywt.wavedec(w[:, j, k], wavelet_name, level=wavelet_level)
                        # VisuShrink-like 阈值
                        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-8
                        thr = sigma * np.sqrt(2.0 * np.log(max(T, 2)))
                        coeffs_f = [pywt.threshold(c, thr, mode='soft') if i>0 else c
                                    for i, c in enumerate(coeffs)]
                        w_d[:, j, k] = pywt.waverec(coeffs_f, wavelet_name)[:T]
                w_new = torch.as_tensor(w_d, dtype=torch.float32, device=R_seq.device)
            elif fallback_savgol:

                # 回退：Savitzky-Golay（需要 scipy>=1.2）
                try:
                    if verbose:
                        print(f"[BodySmooth] Denoiser=Savitzky-Golay(window={savgol_window}, poly={savgol_poly})")
                    from scipy.signal import savgol_filter
                    w_d = np.empty_like(w)
                    for j in range(J):
                        for k in range(3):
                            w_d[:, j, k] = savgol_filter(w[:, j, k], savgol_window, savgol_poly, mode='nearest')
                    w_new = torch.as_tensor(w_d, dtype=torch.float32, device=R_seq.device)
                except Exception:
                    if verbose:
                        print("[BodySmooth] Denoiser=None (no pywt, no savgol fallback)  -> skipped smoothing")
                    # 最后回退：中值+均值
                    w_t = torch.as_tensor(w, dtype=torch.float32, device=R_seq.device)
                    w_new = torch.nn.functional.avg_pool1d(
                        w_t.permute(1,2,0).unsqueeze(0),  # [1,J*3,T]
                        kernel_size=5, stride=1, padding=2
                    ).squeeze(0).permute(2,0,1)  # [T,J,3]
            else:
                w_new = torch.as_tensor(w, dtype=torch.float32, device=R_seq.device)

            # exp 回 R
            R_new = so3_exp(w_new.reshape(-1,3)).reshape(T, J, 3,3)
            return R_new

        f_R_final = denoise_seq_R(f_R_fixed)
        s_R_final = denoise_seq_R(s_R_fixed)
    else:
        f_R_final, s_R_final = f_R_fixed, s_R_fixed

    # 写回轴角
    f_aa_new = rotation_matrix_to_rodrigues(f_R_final.reshape(-1,3,3)).reshape(T, J, 3)
    s_aa_new = rotation_matrix_to_rodrigues(s_R_final.reshape(-1,3,3)).reshape(T, J, 3)

    # 填回原数组（只改 0..21）
    front_out = front_poses_in.copy()
    side_out  = side_poses_in.copy()
    front_out[:, idx.cpu().numpy(), :] = f_aa_new.detach().cpu().numpy()
    side_out[:,  idx.cpu().numpy(), :] = s_aa_new.detach().cpu().numpy()

    if verbose:
        n_f = int(f_bad.any(dim=1).sum().item())
        n_s = int(s_bad.any(dim=1).sum().item())
        print(f"[BodySmooth] Marked frames (any joint): front={n_f} / side={n_s}")
    return front_out, side_out
    
# 索引是相对于整个poses张量(55,)和kp3d张量(144,)的
FINGER_DEFS  = [
    # --- Left Hand (hand_idx = 0) ---
    {
        'thumb': {'pose_slice': slice(37, 40), 'tip_kp': 66},
        'index': {'pose_slice': slice(25, 28), 'tip_kp': 67},
        'middle':{'pose_slice': slice(28, 31), 'tip_kp': 68},
        'ring':  {'pose_slice': slice(34, 37), 'tip_kp': 69},
        'pinky': {'pose_slice': slice(31, 34), 'tip_kp': 70},
    },
    # --- Right Hand (hand_idx = 1) ---
    {
        'thumb': {'pose_slice': slice(52, 55), 'tip_kp': 71},
        'index': {'pose_slice': slice(40, 43), 'tip_kp': 72},
        'middle':{'pose_slice': slice(43, 46), 'tip_kp': 73},
        'ring':  {'pose_slice': slice(49, 52), 'tip_kp': 74},
        'pinky': {'pose_slice': slice(46, 49), 'tip_kp': 75},
    }
]

def coordtrans_single(rm_local: torch.Tensor,
                      T_joint: torch.Tensor,
                      transtype: int) -> torch.Tensor:
    """Change rotation basis: 1 = SMPL-X to adjustment, 0 = inverse."""
    if rm_local.shape[-2:] != (3, 3) or T_joint.shape[-2:] != (3, 3):
        raise ValueError("Rotations and frames must end in (3, 3)")
    T_joint = T_joint.to(device=rm_local.device, dtype=rm_local.dtype)
    # Columns of T are adjustment-frame axes in SMPL-X rest coordinates.
    if transtype == 1:
        return T_joint.transpose(-1, -2) @ rm_local @ T_joint
    if transtype == 0:
        return T_joint @ rm_local @ T_joint.transpose(-1, -2)
    raise ValueError("transtype must be 1 (to adjustment) or 0 (to SMPL-X)")

FOUR_FINGER_NAMES = ['index', 'middle', 'ring', 'pinky']
HAND_NAMES = ['left', 'right']
HAND_FIX_VERSION = 'rest_geometry_x_flexion_v1'
HAND_FRAME_PATH = Path(__file__).with_name('hand_coordinate_frames.npz')


def load_hand_flexion_frames(device, dtype=torch.float32):
    with np.load(HAND_FRAME_PATH, allow_pickle=False) as data:
        matrices = data['T_target_to_smplx']
        names = data['finger_names'].tolist()
    if names != ['index', 'middle', 'pinky', 'ring', 'thumb'] or matrices.shape != (2, 15, 3, 3):
        raise ValueError(f"Unexpected hand frame layout: {HAND_FRAME_PATH}")
    if not np.isfinite(matrices).all():
        raise ValueError("Hand frames contain non-finite values")
    if not np.allclose(matrices.swapaxes(-1, -2) @ matrices, np.eye(3), atol=1e-8):
        raise ValueError("Hand frames must be orthogonal")
    if not np.allclose(np.linalg.det(matrices), 1.0, atol=1e-8):
        raise ValueError("Hand frames must be right-handed")
    indices = [names.index(name) * 3 + j for name in FOUR_FINGER_NAMES for j in range(3)]
    return torch.as_tensor(matrices[:, indices].copy(), dtype=dtype, device=device)


def hand_euler_xyz_to_matrix(angles):
    """Target Euler [flexion X, twist Y, splay Z], R = Rz @ Ry @ Rx."""
    x, y, z = angles.unbind(-1)
    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)
    return torch.stack((
        cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy, cy * sx, cy * cx,
    ), dim=-1).reshape(*angles.shape[:-1], 3, 3)


def hand_matrix_to_euler_xyz(matrix, eps=1e-6):
    """Keep flexion X outside the middle Euler axis to allow bends past 90 deg."""
    sin_y = (-matrix[..., 2, 0]).clamp(-1, 1)
    cos_y = torch.sqrt(torch.clamp(1 - sin_y.square(), min=0))
    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    y = torch.atan2(sin_y, cos_y)
    z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    locked = cos_y <= eps
    x = torch.where(locked, torch.zeros_like(x), x)
    z = torch.where(locked, torch.atan2(-matrix[..., 0, 1], matrix[..., 1, 1]), z)
    return torch.stack((x, y, z), dim=-1)
# =============================================================================
# --- (核心重写) 手指姿态纠错函数 ---
# =============================================================================

def _apply_percentage_fix(poses_np_flat: np.ndarray, t: int, h_idx: int, f_idx: int, excess_percentage: float, 
                          upper_bounds: torch.Tensor, T_h: torch.Tensor, device: torch.device,
                          judgment_weights = (0.7, 0.3), correction_weights = (0.7, 0.7, 0.5)):
    """
    将一个表示“额外伸展度”的百分比转换成旋转量，并应用到目标手指上。
    """
    if not np.isfinite(excess_percentage) or excess_percentage < 0:
        raise ValueError("excess_percentage must be finite and nonnegative")
    pose_slice = FINGER_DEFS[h_idx][FOUR_FINGER_NAMES[f_idx]]['pose_slice']
    if len(correction_weights) != 3 or len(judgment_weights) != 2:
        raise ValueError("Expected three correction weights and two judgment weights")
    weights = torch.as_tensor(correction_weights, device=device, dtype=T_h.dtype)
    bounds = upper_bounds[h_idx, f_idx].to(dtype=T_h.dtype)
    judgment = torch.as_tensor(judgment_weights, device=device, dtype=T_h.dtype)
    if torch.any(weights < 0) or torch.any(bounds <= 0) or torch.any(judgment < 0):
        raise ValueError("Invalid correction weights or angular bounds")
    # Invert the same normalized bend score used by the detector.
    denominator = torch.sum(judgment * weights[:2] / bounds)
    if denominator.item() <= 0:
        raise ValueError("Correction weights must contribute to the bend score")
    scale = excess_percentage / denominator.item()
    for j, weight in enumerate(weights):
        joint_idx = pose_slice.start + j
        offset = joint_idx * 3
        aa = torch.as_tensor(poses_np_flat[t, offset:offset + 3], device=device, dtype=T_h.dtype).unsqueeze(0)
        frame = T_h[h_idx, f_idx * 3 + j]
        adjusted = coordtrans_single(rodrigues_to_rotation_matrix(aa), frame, 1)
        euler = hand_matrix_to_euler_xyz(adjusted)
        original_x = euler[0, 0].item()
        # Straighten each joint in its own frame; never cross zero or bend an already straight joint.
        new_magnitude = max(abs(original_x) - scale * weight.item(), 0.0)
        if new_magnitude == abs(original_x):
            continue
        euler[0, 0] = np.copysign(new_magnitude, original_x)
        fixed = coordtrans_single(hand_euler_xyz_to_matrix(euler), frame, 0)
        poses_np_flat[t, offset:offset + 3] = rotation_matrix_to_rodrigues(fixed)[0].cpu().numpy()


def correct_hand_gestures_across_views(
    front_poses_in: np.ndarray,
    side_poses_in: np.ndarray,
    diagnostic_prints: bool = False,
    verbose: bool = True
):
    """
    (REVISED) 两阶段手指修正逻辑:
    1. 基于归一化百分比的跨视角修复。
    2. 基于记录事件的时间扩散修复。
    """
    print("\n--- Running Two-Pass Finger Correction (Normalization & Temporal Diffusion) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    -> Using {str(device).upper()} for tensor operations.")

    # --- 1. 设置 ---
    T = front_poses_in.shape[0]
    original_shape = front_poses_in.shape
    is_3d_input = len(original_shape) == 3
    
    front_poses_flat = front_poses_in.copy().reshape(T, -1)
    side_poses_flat = side_poses_in.copy().reshape(T, -1)
    if front_poses_in.shape != side_poses_in.shape or front_poses_flat.shape[1] != 165:
        raise ValueError("Front and side must have identical (T,55,3) or (T,165) pose shapes")
    if not np.isfinite(front_poses_flat).all() or not np.isfinite(side_poses_flat).all():
        raise ValueError("Hand correction requires finite poses")

    upper_bounds_j1 = torch.tensor([1.75, 1.8, 1.8, 1.7], device=device)
    upper_bounds_j2 = torch.tensor([1.75, 1.75, 1.75, 1.75], device=device)
    upper_bounds = torch.stack([
        torch.stack([upper_bounds_j1, upper_bounds_j2], dim=1),
        torch.stack([upper_bounds_j1, upper_bounds_j2], dim=1)
    ], dim=0)

    judgment_weights = torch.tensor([0.7, 0.3], device=device)
    
    T_matrices = load_hand_flexion_frames(device)

    # --- 2. 预计算 ---
    def _calculate_straightness(poses_flat):
        all_x_rots = torch.zeros(T, 2, 4, 2, device=device)
        for h in range(2):
            for f, name in enumerate(FOUR_FINGER_NAMES):
                for j in range(2): # First two joints
                    joint_idx = FINGER_DEFS[h][name]['pose_slice'].start + j
                    aa = torch.tensor(poses_flat[:, joint_idx*3 : joint_idx*3+3], device=device, dtype=torch.float32)
                    rm_local = rodrigues_to_rotation_matrix(aa)
                    T_joint = T_matrices[h, f*3+j]
                    rm_adjusted = coordtrans_single(rm_local, T_joint, transtype=1)
                    all_x_rots[:, h, f, j] = hand_matrix_to_euler_xyz(rm_adjusted)[:, 0]
        
        bend_perc_per_joint = torch.abs(all_x_rots) / upper_bounds
        weighted_bend_perc = bend_perc_per_joint * judgment_weights.view(1, 1, 1, 2)
        bend_perc = torch.sum(weighted_bend_perc, dim=3)
        
        return (1.0 - bend_perc.clamp(0.0, 1.0)), all_x_rots
    
    straightness_front, _ = _calculate_straightness(front_poses_flat)
    straightness_side, _ = _calculate_straightness(side_poses_flat)

    # --- 3. Pass 1: 跨视角修复 ---
    print("    -> Pass 1: Cross-view correction...")
    events = []
    total_corrections_pass1 = 0
    for t in range(T):
        repaired_in_frame = {'left': False, 'right': False}
        for h_idx in range(2):
            hand_key = 'left' if h_idx == 0 else 'right'
            
            s_front = straightness_front[t, h_idx]
            means_f = torch.zeros(4, device=device)
            vars_f = torch.zeros(4, device=device)
            for i in range(4):
                others = s_front[torch.arange(4, device=device) != i]
                means_f[i] = others.mean()
                vars_f[i] = others.var()
            
            excess_f = s_front - means_f
            is_outlier_f = (s_front > means_f * 1.15) & (vars_f < 0.2)
            
            if is_outlier_f.any():
                winner_idx_f = torch.argmax(excess_f)
                if is_outlier_f[winner_idx_f] and not repaired_in_frame[hand_key]:
                    if straightness_side[t, h_idx, winner_idx_f] < s_front[winner_idx_f] * 0.65:
                        excess_perc = excess_f[winner_idx_f].item()*1.2
                        _apply_percentage_fix(side_poses_flat, t, h_idx, winner_idx_f.item(), excess_perc, upper_bounds, T_matrices, device,
                                              judgment_weights=judgment_weights)

                        ev = {'t': t, 'h': h_idx, 'f': winner_idx_f.item(), 'excess': excess_perc, 
                            'repaired_view': 'side', 'mean_others': means_f[winner_idx_f].item(), 'var_others': vars_f[winner_idx_f].item()}
                        events.append(ev)

                        msg = (f"[Pass1] t={t:04d} hand={HAND_NAMES[h_idx]:>5s} "
                            f"finger={FOUR_FINGER_NAMES[winner_idx_f]:>6s}  -> applied to: side   "
                            f"excess={excess_perc:+.3f}")
                        #print(msg)
                        # if 'log_lines' in locals():
                        #     log_lines.append(msg)

                        total_corrections_pass1 += 1
                        repaired_in_frame[hand_key] = True

            if not repaired_in_frame[hand_key]:
                s_side = straightness_side[t, h_idx]
                means_s = torch.zeros(4, device=device)
                vars_s = torch.zeros(4, device=device)
                for i in range(4):
                    others = s_side[torch.arange(4, device=device) != i]
                    means_s[i] = others.mean()
                    vars_s[i] = others.var()
                
                excess_s = s_side - means_s
                is_outlier_s = (s_side > means_s * 1.15) & (vars_s < 0.2)

                if is_outlier_s.any():
                    winner_idx_s = torch.argmax(excess_s)
                    if is_outlier_s[winner_idx_s]:
                        if straightness_front[t, h_idx, winner_idx_s] < s_side[winner_idx_s] * 0.65:
                            excess_perc = excess_s[winner_idx_s].item()*1.2
                            _apply_percentage_fix(front_poses_flat, t, h_idx, winner_idx_s.item(), excess_perc, upper_bounds, T_matrices, device,
                                                  judgment_weights=judgment_weights)

                            ev = {'t': t, 'h': h_idx, 'f': winner_idx_s.item(), 'excess': excess_perc, 
                                'repaired_view': 'front', 'mean_others': means_s[winner_idx_s].item(), 'var_others': vars_s[winner_idx_s].item()}
                            events.append(ev)

                            msg = (f"[Pass1] t={t:04d} hand={HAND_NAMES[h_idx]:>5s} "
                                f"finger={FOUR_FINGER_NAMES[winner_idx_s]:>6s}  -> applied to: front  "
                                f"excess={excess_perc:+.3f}")
                            #print(msg)
                            # if 'log_lines' in locals():
                            #     log_lines.append(msg)

                            total_corrections_pass1 += 1
                            repaired_in_frame[hand_key] = True
    print(f"    -> Pass 1 Complete. Corrections: {total_corrections_pass1}")

    # --- 4. Pass 2: 时间扩散修复 ---
    print("    -> Pass 2: Temporal diffusion...")
    straightness_front_p2, _ = _calculate_straightness(front_poses_flat)
    straightness_side_p2, _ = _calculate_straightness(side_poses_flat)
    total_corrections_pass2 = 0

    for event in events:
        t, h, f, excess = event['t'], event['h'], event['f'], event['excess']
        
        source_straightness = straightness_front_p2 if event['repaired_view'] == 'side' else straightness_side_p2
        
        for dt in [-1, 1]:
            t_adj = t + dt
            if not (0 <= t_adj < T): continue
            
            poses_to_fix = side_poses_flat if event['repaired_view'] == 'side' else front_poses_flat
            straightness_to_fix = straightness_side_p2 if event['repaired_view'] == 'side' else straightness_front_p2

            if straightness_to_fix[t_adj, h, f] < source_straightness[t, h, f] * 0.65:
                s_adj = straightness_to_fix[t_adj, h]
                others_adj = s_adj[torch.arange(4, device=device) != f]
                mean_adj_others = others_adj.mean()
                var_adj_others = others_adj.var()
                
                # (BUG FIX) Correctly compare scalar values
                if abs(mean_adj_others.item() - event['mean_others']) < 0.15 and abs(var_adj_others.item() - event['var_others']) < 0.2:
                    _apply_percentage_fix(poses_to_fix, t_adj, h, f, excess, upper_bounds, T_matrices, device,
                                          judgment_weights=judgment_weights)

                    msg = (f"[Pass2] t={t_adj:04d} hand={HAND_NAMES[h]:>5s} "
                        f"finger={FOUR_FINGER_NAMES[f]:>6s}  -> propagated from t={t:04d} (base={event['repaired_view']})  "
                        f"excess={excess:+.3f}")
                    # print(msg)
                    # if 'log_lines' in locals():
                    #     log_lines.append(msg)

                    total_corrections_pass2 += 1

    print(f"    -> Pass 2 Complete. Corrections: {total_corrections_pass2}")
    
    # --- 5. 返回结果 ---
    print(f"--- Finger Correction Complete. Total corrections: {total_corrections_pass1 + total_corrections_pass2} ---")
    
    if is_3d_input:
        return front_poses_flat.reshape(original_shape), side_poses_flat.reshape(original_shape)
    else:
        return front_poses_flat, side_poses_flat

# --- (修改) 流程控制函数 ---
# --- (MODIFIED) Process Control Function ---
def process_pair(
    front_path: Path,
    side_path: Path,
    output_front_path: Path,
    output_side_path: Path,
    debug_enabled: bool,
    enable_hand_fix: bool = True,
    hand_fix_diag: bool = False,
    disable_idle_trim: bool = False,
):
    print(f"Processing pair:\n  Front: {front_path.name}\n  Side:  {side_path.name}")
    
    front_data_in = dict(np.load(front_path))
    side_data_in = dict(np.load(side_path))

    front_output_data = {k: v.copy() for k, v in front_data_in.items()}
    side_output_data = {k: v.copy() for k, v in side_data_in.items()}

    # --- 简单对齐长度：谁短补谁（尾帧重复），只需在 Pass 1 之前做一次 ---
    TIME_KEYS = ('poses', 'smpl_kp3d', 'trans')  # 你只想补 poses + kp3d；trans 建议一并补，避免后续用到报错

    def _len_safe(d, k):
        v = d.get(k, None)
        return int(v.shape[0]) if (isinstance(v, np.ndarray) and v.ndim >= 1) else 0

    # 取双方“可用长度”的最大值作为统一目标长度
    target_len = max(
        _len_safe(front_output_data, 'poses'),
        _len_safe(front_output_data, 'smpl_kp3d'),
        _len_safe(side_output_data,  'poses'),
        _len_safe(side_output_data,  'smpl_kp3d')
    )

    def _pad_or_trunc_inplace(d: dict, keys: tuple[str, ...], T: int):
        for k in keys:
            if k not in d: 
                continue
            arr = d[k]
            if not (isinstance(arr, np.ndarray) and arr.ndim >= 1):
                continue
            cur = arr.shape[0]
            if cur == T:
                continue
            if cur == 0:
                # 空就直接用零填（形状对齐）
                d[k] = np.zeros((T, *arr.shape[1:]), dtype=arr.dtype)
            elif cur < T:
                # 尾帧重复补齐
                out = np.empty((T, *arr.shape[1:]), dtype=arr.dtype)
                out[:cur] = arr
                out[cur:] = arr[cur-1]   # 重复最后一帧
                d[k] = out
            else:
                # 比目标更长就截断到 T
                d[k] = arr[:T].copy()

    _pad_or_trunc_inplace(front_output_data, TIME_KEYS, target_len)
    _pad_or_trunc_inplace(side_output_data,  TIME_KEYS, target_len)
    print("\n--- Pass 1: Performing preliminary spatial alignment ---")
    front_data_in=front_output_data
    side_data_in=side_output_data
    preliminary_aligned_side = align_views_with_pelvis_correction(front_data_in, side_data_in)

    best_offset = find_temporal_offset_on_aligned_data(
        front_data_in['smpl_kp3d'], 
        preliminary_aligned_side['smpl_kp3d_aligned'],
        search_range=8
    )

    target_len = len(front_output_data['poses'])
    side_output_data = create_synced_data_with_padding(side_output_data, best_offset, target_len)
    if disable_idle_trim:
        idle_runs = []
        print("[Idle-v3] disabled by --disable_idle_trim.")
    else:
        idle_runs = detect_idle_v3(
            front_output_data['poses'],
            front_output_data['smpl_kp3d'],
            save_debug_png=None, #"idle_detect_debug.png"
            height_gamma=0.25,           # 更严格可 0.32~0.35
            norm_idle_th=0.30,
            min_idle_len=24,
            lookahead_when_resume=8,
            fingers_dead_eps=1e-3,
            fingers_dead_min=18,
            verbose=False
        )

    if idle_runs:
        front_output_data, side_output_data = drop_idle_segments_with_bridge(
            front_output_data, side_output_data, idle_runs, bridge_frames=4
        )
        print(f"[Idle-v3] removed {sum(e-s+1 for s,e in idle_runs)} frames; "
            f"new T={front_output_data['poses'].shape[0]}")
    else:
        print("[Idle-v3] no idle removed.")
    front_body_fixed, side_body_fixed = fix_body_rotations_spikes_and_smooth(
        front_output_data['poses'],            # 形状 (T,55,3)
        side_output_data['poses'],
        joints_body = range(1, 22),
        spike_deg_hi=80.0,    # 可以按你的素材调，比如 100~160
        spike_deg_lo=30.0,     # “另一视角无峰”的判据
        max_spike_len=2,
        wavelet_denoise=True,  # 关掉就设 False
        wavelet_name="coif3",
        wavelet_level=2,
        fallback_savgol=True,
        savgol_window=7,
        savgol_poly=2,
        verbose=False,
        verbose_detail=False,
        joint_names=None  # 或者传 {0:"pelvis",1:"left_hip",...}
    )
    front_output_data['poses'] = front_body_fixed
    side_output_data['poses']  = side_body_fixed
    if enable_hand_fix:
        # (MODIFIED) 调用新的修正函数
        corrected_front_poses, corrected_side_poses = correct_hand_gestures_across_views(
            front_output_data['poses'],
            side_output_data['poses'],
            diagnostic_prints=hand_fix_diag
        )
        front_output_data['poses'] = corrected_front_poses
        side_output_data['poses'] = corrected_side_poses
        front_output_data['hand_fix_applied'] = True
        side_output_data['hand_fix_applied'] = True
        front_output_data['hand_fix_version'] = HAND_FIX_VERSION
        side_output_data['hand_fix_version'] = HAND_FIX_VERSION
        print("Hand-fix: ENABLED (new gesture correction logic applied).")
    else:
        front_output_data['hand_fix_applied'] = False
        side_output_data['hand_fix_applied'] = False
        print("Hand-fix: DISABLED (skipping gesture correction).")

    print("\n--- Pass 2: Performing final spatial alignment ---")
    final_aligned_side_data = align_views_with_pelvis_correction(front_output_data, side_output_data)

    print("\n--- Calculating Final Camera Parameters ---")
    img_height, img_width = 720, 1280
    focal_length = 5000.
    K = np.array([[focal_length, 0, img_width / 2], [0, focal_length, img_height / 2], [0, 0, 1]])
    
    front_cam_trans_avg = np.mean(front_data_in['cam_trans'], axis=0)
    canonical_distance = np.linalg.norm(front_cam_trans_avg)
    R_front_cam = np.identity(3)
    t_front_cam = np.array([0, 0, canonical_distance])
    P_front_canonical = K @ np.hstack([R_front_cam, t_front_cam.reshape(3, 1)])

    R_corr_final = Rotation.from_rotvec(final_aligned_side_data['poses'][0, 0]).as_matrix() @ \
                   Rotation.from_rotvec(side_output_data['poses'][0, 0]).as_matrix().T
    t_corr_final = final_aligned_side_data['trans'][0] - (R_corr_final @ side_output_data['trans'][0])
    R_side_virtual_cam = R_front_cam @ R_corr_final.T
    t_side_virtual_cam = t_front_cam - R_side_virtual_cam @ t_corr_final
    P_side_virtual = K @ np.hstack([R_side_virtual_cam, t_side_virtual_cam.reshape(3, 1)])
    
    front_output_data['P_canonical_cam'] = P_front_canonical
    final_aligned_side_data['P_virtual_cam'] = P_side_virtual

    print("Zeroing out final side translation vector for better consistency.")
    final_aligned_side_data['trans'] = np.zeros_like(final_aligned_side_data['trans'])

    if debug_enabled:
        title_suffix = front_path.stem.replace('_front_aioswilor', '')
        debug_visualization_interactive(front_output_data, final_aligned_side_data, K, title_suffix, frame_index=0)
            
    tag = "corrected" if enable_hand_fix else "uncorrected"
    print(f"\nSaving final (front, {tag}) data to: {output_front_path}")
    np.savez(output_front_path, **front_output_data)
    
    print(f"Saving final (side-aligned, {tag}) data to: {output_side_path}")
    np.savez(output_side_path, **final_aligned_side_data)
    
    print("\nPair processing complete.")


# --- 主函数 (保持不变) ---
# --- Main Function (Unchanged) ---
def main():
    parser = argparse.ArgumentParser(
        description="Batch align and correct hand gestures for side-view NPZ files relative to front-view files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--front_dir', type=str, required=True, help="Directory containing front view NPZ files.")
    parser.add_argument('--side_dir', type=str, required=True, help="Directory containing side view NPZ files.")
    parser.add_argument('--output_front_dir', type=str, required=True, help="Directory to save the processed front view NPZ files.")
    parser.add_argument('--output_side_dir', type=str, required=True, help="Directory to save the aligned and processed side view NPZ files.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for 3D visualization.")
    parser.add_argument('--no_hand_fix', action='store_true', help="Disable the finger gesture correction step.")
    parser.add_argument('--hand_fix_diag', action='store_true', help="Enable diagnostic prints for finger correction to tune threshold.")
    parser.add_argument('--disable_idle_trim', action='store_true',
                        help="Disable idle segment detection/removal for datasets where frame ranges are already curated.")
    # 新增：覆盖已存在输出的开关（默认不覆盖 -> 断点续跑）
    parser.add_argument('--overwrite', action='store_true',
                        help="Reprocess even if output files already exist (disable resume).")

    args = parser.parse_args()

    front_dir = Path(args.front_dir)
    side_dir = Path(args.side_dir)
    output_front_dir = Path(args.output_front_dir)
    output_side_dir = Path(args.output_side_dir)
    
    output_front_dir.mkdir(parents=True, exist_ok=True)
    output_side_dir.mkdir(parents=True, exist_ok=True)
    
    front_files = sorted(list(front_dir.glob('*_front_aioswilor.npz')),reverse=True)
    if not front_files:
        print(f"Error: No files matching '*_front_aioswilor.npz' found in {front_dir}")
        return

    pairs_to_process = []
    for front_path in front_files:
        expected_side_name = front_path.name.replace('_front_aioswilor.npz', '_side_aioswilor.npz')
        side_path = side_dir / expected_side_name
        if side_path.is_file():
            pairs_to_process.append((front_path, side_path))
        else:
            print(f"Warning: No matching side file found for '{front_path.name}'.")

    if not pairs_to_process:
        print("Error: No matching front/side file pairs were found.")
        return

    print(f"\nFound {len(pairs_to_process)} pairs to process.")
    for i, (front_path, side_path) in enumerate(pairs_to_process):
        print(f"\n{'='*60}\nProcessing Pair {i+1} / {len(pairs_to_process)}\n{'='*60}")
        
        output_front_path = output_front_dir / front_path.name
        output_side_path = output_side_dir / side_path.name
        # === 断点续跑逻辑：若两个输出都已存在且文件可读，就跳过 ===
        if not args.overwrite:
            both_exist = output_front_path.exists() and output_side_path.exists()
            if both_exist:
                #skipped += 1
                print(f"[Resume] Skip Pair {i+1}/{len(pairs_to_process)}: outputs exist -> {output_front_path.name}, {output_side_path.name}")
                continue


        process_pair(
            front_path, 
            side_path, 
            output_front_path, 
            output_side_path, 
            args.debug,
            enable_hand_fix=not args.no_hand_fix,
            hand_fix_diag=args.hand_fix_diag,
            disable_idle_trim=args.disable_idle_trim,
        )

    print(f"\n{'='*60}\nBatch processing complete.\n{'='*60}")
    print(f"Processed front view files saved to: {output_front_dir}")
    print(f"Aligned side view files saved to: {output_side_dir}")

if __name__ == "__main__":
    main()

