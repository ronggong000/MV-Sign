import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
# smplx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'smplx')
# sys.path.append(smplx_path)
# import smplx
from tqdm import tqdm

#loss joint
LOSS_JOINT_INDICES = [
    18, 19, 20, 21,  # Elbows and Wrists
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,  # Left Hand
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,  # Right Hand
    66, 67, 68, 69, 70,  # Left Fingertips
    71, 72, 73, 74, 75,  # Right Fingertips
]
# --- Constants for Bio-Limit Hinge Loss ---

# Coordinate transformation matrices from SMPL-X Euler space to KNOWN Euler space
euler_coordtrans_RIGHT = np.array([
       [   0,  -6,  10,# right_index1
           0,   1,  -5,# right_index2
           0, -10,   0,# right_index3
           0,  10,   2,# right_middle1
           0,  10,  -2,
           0,  10,  10,
          30,  35,  10,# right_pinky1
          20,  42,   8,
          15,  45,   8,
           0,  12,   5,# right_ring1
           0,  25,  -5,
           0,  25,  15,
          30, -75, -35,# right_thumb1
          20, -45, -30,
          20, -45, -30]]).reshape(15, 3) / 180 * np.pi

euler_coordtrans_LEFT = np.array([
       [    0,   6, -10,
            0,  -1,   5,
            0,  10,   0,
            0, -10,  -2,
            0, -10,   2,
            0, -10, -10,
            30, -35, -10,
            20, -42,  -8,
            15, -45,  -8,
            0, -12,  -5,
            0, -25,   5,
            0, -25, -15,
            30,  75,  35,
            20,  45,  30,
            20,  45,  30]
            ]).reshape(15, 3) / 180 * np.pi

# Bounds of Motion (BOF) for right hand in KNOWN Euler space [rad]
BOF_RIGHT = np.array([
       [    0, -40, -30,
            0,   0,   0,
            0,   0, -45,
            0, -20, -20,
            0,   0,   0,
            0,   0, -20,
            0, -10, -30,
            0,   0,   0,
            0,   0, -40,
            0, -20, -20,
            0,   0,   0,
            0,   0, -10,
            0, -90, -45,
            0,-100,   0,
            0, -90,   0],
       [    0,   5,  90,
            0,   0, 110, 
            0,   0,  90, 
            0,  15, 100, 
            0,   0, 110,
            0,   0,  85, 
            0,  45, 100, 
            0,   0, 100, 
            0,   0,  90, 
            0,  20,  90,
            0,   0, 110, 
            0,   0,  80, 
            0,  20,  45, 
            0,   0,   0, 
            0,  45,   0]
]).reshape(2, 15, 3) / 180 * np.pi

# Bounds of Motion (BOF) for left hand in KNOWN Euler space [rad]
BOF_LEFT = np.array([
        [  0,  -5, -90, 0,   0,-110, 0,   0, -90, 0, -15,-100, 0,   0,-110,
           0,   0, -85, 0, -45,-100, 0,   0,-100, 0,   0, -90, 0, -20, -90,
           0,   0,-110, 0,   0, -80, 0, -20, -45, 0,   0,   0, 0, -45,   0],
        [  0,  40,  30, 0,   0,   0, 0,   0,  45, 0,  20,  20, 0,   0,   0,
           0,   0,  20, 0,  10,  30, 0,   0,   0, 0,   0,  40, 0,  20,  20,
           0,   0,   0, 0,   0,  10, 0,  90,  45, 0, 100,   0, 0,  90,   0],
]).reshape(2, 15, 3) / 180 * np.pi
MANO_TO_SMPLX_REMAPPING = [
    # Index (SMPL-X joints 0,1,2 get MANO joints 3,4,5)
    3, 4, 5,
    # Middle (SMPL-X joints 3,4,5 get MANO joints 6,7,8)
    6, 7, 8,
    # Pinky (SMPL-X joints 6,7,8 get MANO joints 12,13,14)
    12, 13, 14,
    # Ring (SMPL-X joints 9,10,11 get MANO joints 9,10,11)
    9, 10, 11,
    # Thumb (SMPL-X joints 12,13,14 get MANO joints 0,1,2)
    0, 1, 2
]
mano_order_map = {'thumb':[0,1,2],'index': [3, 4, 5], 'middle': [6, 7, 8], 'ring': [9, 10, 11], 'pinky': [12, 13, 14]}
# --- Core Conversion Functions (Torch-enabled for autograd) ---


def batch_euler2mat_zxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = ymat @ xmat @ zmat
    return rotMat

def batch_mat2euler_zxy(R, eps=1e-6):
    """
    Inverse of batch_euler2mat_zxy.
    Args:
        R: [N, 3, 3] rotation matrices with R = Ry @ Rx @ Rz (intrinsic Z-X-Y).
    Returns:
        angles: [N, 3] where [:,0]=alpha(x), [:,1]=beta(y), [:,2]=gamma(z)
    """
    assert R.shape[-2:] == (3, 3)
    r00, r01, r02 = R[:,0,0], R[:,0,1], R[:,0,2]
    r10, r11, r12 = R[:,1,0], R[:,1,1], R[:,1,2]
    r20, r21, r22 = R[:,2,0], R[:,2,1], R[:,2,2]

    # alpha from -sin(alpha) = r12
    sin_alpha = -r12
    sin_alpha = sin_alpha.clamp(-1.0, 1.0)
    cos_alpha = torch.sqrt(torch.clamp(1.0 - sin_alpha*sin_alpha, min=0.0))

    # general branch (no gimbal lock)
    alpha = torch.atan2(sin_alpha, cos_alpha)
    beta  = torch.atan2(r02, r22)          # shares cos_alpha factor, cancels out
    gamma = torch.atan2(r10, r11)          # shares cos_alpha factor, cancels out

    # gimbal lock: cos_alpha ~ 0  => alpha ≈ ±pi/2, only (beta±gamma) observable
    lock = cos_alpha <= eps
    if lock.any():
        # determine sign of alpha from r12 = -sin(alpha)
        alpha_lock = torch.where(r12[lock] < 0,  0.5*torch.pi, -0.5*torch.pi)
        # set gamma = 0 by convention, solve beta from top-left 2x2 block
        beta_lock = torch.where(
            r12[lock] < 0,  # alpha ≈ +pi/2  -> use beta = atan2(r01, r00)
            torch.atan2(r01[lock], r00[lock]),
            torch.atan2(-r01[lock], r00[lock])  # alpha ≈ -pi/2
        )
        # write back
        alpha = alpha.clone(); beta = beta.clone(); gamma = gamma.clone()
        alpha[lock] = alpha_lock
        beta[lock]  = beta_lock
        gamma[lock] = 0.0

    return torch.stack([alpha, beta, gamma], dim=-1)


def rodrigues_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert Rodrigues vectors (axis-angle) to rotation matrices.
    Args:
        axis_angle: [N, 3], Rodrigues vectors.
    Returns:
        R: [N, 3, 3], rotation matrices.
    """
    N = axis_angle.shape[0]
    angle = torch.norm(axis_angle, p=2, dim=1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    K = torch.zeros((N, 3, 3), device=axis_angle.device)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    
    I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(N, -1, -1)
    R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * torch.bmm(K, K)
    
    # Handle small angles
    is_small_angle = (angle.squeeze() < 1e-8)
    R[is_small_angle] = torch.eye(3, device=axis_angle.device)
    
    return R

def rotation_matrix_to_rodrigues(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to Rodrigues vectors (axis-angle).
    Args:
        R: [N, 3, 3]
    Returns:
        axis_angle: [N, 3]
    """
    N = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    # For general case
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = torch.stack([rx, ry, rz], dim=1)
    denom = 2.0 * torch.sin(theta).unsqueeze(1) + 1e-8
    axis = axis / denom

    # Small angle: fall back to first-order approx around identity
    small = theta < 1e-8
    if small.any():
        axis[small] = torch.stack([
            (R[small, 2, 1] - R[small, 1, 2]) / 2.0,
            (R[small, 0, 2] - R[small, 2, 0]) / 2.0,
            (R[small, 1, 0] - R[small, 0, 1]) / 2.0,
        ], dim=1)

    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
    axis_angle = axis * theta.unsqueeze(1)
    return axis_angle
def coordtrans(rotmat, local2global, transtype):

    batchsize = rotmat.shape[0]
    local2global = local2global.clone()
    if transtype == 1:
        rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', rotmat[:,1:], local2global[:batchsize])
        rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', local2global[:batchsize].transpose(2,3), rotmat[:,1:])
    else:
        rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', rotmat[:,1:], local2global[:batchsize].transpose(2,3))
        rotmat[:,1:] = torch.einsum('...ij,...jk->...ik', local2global[:batchsize], rotmat[:,1:])

    return rotmat
def compute_anglebound(pred_pose, angle_limits, batch_size, interjoint, device):
    # update joint range of motion given different joint angles using the function anatomy knowledge

    angle_limits_bound = angle_limits.expand(2, batch_size, 48).clone()

    if interjoint == 0:
        return angle_limits_bound
    # eliminate prediction that are outside the original bound
    pred_pose_bound = pred_pose.clone().detach()
    pred_pose_bound = torch.where(pred_pose_bound>angle_limits_bound[0],pred_pose_bound,angle_limits_bound[0])
    pred_pose_bound = torch.where(pred_pose_bound<angle_limits_bound[1],pred_pose_bound,angle_limits_bound[1])

    # DIP on PIP
    DIP_index = [11,20,38,29] # [4,7,13,10]
    PIP_index = [8,17,35,26] # [3,6,12,9]
    angle_limits_bound[0,:,PIP_index] = torch.where(angle_limits_bound[0,:,DIP_index]>0, torch.zeros([batch_size,4]).to(device), angle_limits[0,:,PIP_index])

    # MCP joint fingers
    MCP_index_beta = [4,13,31,22] # [2,5,11,8]
    MCP_index_gamma = [5,14,32,23]
    angle_limits_bound[0,:,MCP_index_beta] = angle_limits_bound[0,:,MCP_index_beta] * (1 - (pred_pose_bound[:,MCP_index_gamma].clamp(max=70/180*np.pi)/(70/180*np.pi)))
    angle_limits_bound[1,:,MCP_index_beta] = angle_limits_bound[1,:,MCP_index_beta] * (1 - (pred_pose_bound[:,MCP_index_gamma].clamp(max=70/180*np.pi)/(70/180*np.pi)))
    # MCP joint thumb
    MCP_thumb_beta = 40 # 14
    MCP_thumb_gamma = 41
    angle_limits_bound[0,:,MCP_thumb_gamma] = angle_limits_bound[0,:,MCP_thumb_gamma] * (1 + (pred_pose_bound[:,MCP_thumb_beta].clamp(min=-70/180*np.pi)/(70/180*np.pi)))
    angle_limits_bound[1,:,MCP_thumb_gamma] = angle_limits_bound[1,:,MCP_thumb_gamma] * (1 + (pred_pose_bound[:,MCP_thumb_beta].clamp(min=-70/180*np.pi)/(70/180*np.pi)))

    # eliminate new bound that are outside the original bound
    angle_limits_bound[0] = torch.where(angle_limits_bound[0]>angle_limits[0],angle_limits_bound[0],angle_limits[0])
    angle_limits_bound[1] = torch.where(angle_limits_bound[1]<angle_limits[1],angle_limits_bound[1],angle_limits[1])

    return angle_limits_bound
def calculate_bio_loss(pred_euler, static_bof_tensor, interjoint, device):
    """
    这是我们最终在主循环中调用的函数。
    它接收45维数据(15x3)，通过“包装”来调用48维的核心函数。
    """
    num_frames = pred_euler.shape[0]

    # --- 1. 数据适配：将45维输入填充为48维 ---
    
    # 填充姿态数据：在关节维度(dim=1)的最前面加上一个虚拟手腕
    dummy_wrist_pose = torch.zeros(num_frames, 1, 3, device=device)
    pred_euler_16j = torch.cat([dummy_wrist_pose, pred_euler], dim=1) # Shape: [N, 16, 3]
    pred_euler_48d = pred_euler_16j.reshape(num_frames, 48)           # Shape: [N, 48]
    
    # 填充边界数据
    dummy_wrist_bounds = torch.zeros(2, 1, 3, device=device)
    static_bof_16j = torch.cat([dummy_wrist_bounds, static_bof_tensor], dim=1) # Shape: [2, 16, 3]
    static_bof_48d = static_bof_16j.reshape(2, 48)                            # Shape: [2, 48]

    # --- 2. 调用48维核心函数进行计算 ---
    # angle_limits_48d = torch.from_numpy(static_bof_48d.cpu().numpy()).to(device).float().unsqueeze(1)
    # dynamic_bounds_48d = compute_anglebound(
    #     pred_euler_48d, angle_limits_48d, num_frames, interjoint, device
    # ) # 返回 [2, N, 48]
    bounds_48d = static_bof_48d.unsqueeze(1).expand(2, num_frames, 48)  # [2, N, 48]
    # --- 3. 计算损失 (在45维上) ---
    # 我们只关心手指关节的损失，所以要剔除虚拟手腕
    # lower_bounds = dynamic_bounds_48d[0].reshape(num_frames, 16, 3)[:, 1:, :] # 剔除手腕
    # upper_bounds = dynamic_bounds_48d[1].reshape(num_frames, 16, 3)[:, 1:, :] # 剔除手腕
    lower_bounds = bounds_48d[0].reshape(num_frames, 16, 3)[:, 1:, :]
    upper_bounds = bounds_48d[1].reshape(num_frames, 16, 3)[:, 1:, :]
    # over_upper = torch.clamp(pred_euler - upper_bounds, min=0)
    # under_lower = torch.clamp(lower_bounds - pred_euler, min=0)
    eps = 1e-4
    over_upper = torch.clamp(pred_euler - (upper_bounds + eps), min=0)
    under_lower = torch.clamp((lower_bounds - eps) - pred_euler, min=0)

    jointangle_loss_vec = torch.sum(over_upper**2 + under_lower**2, dim=[1, 2])
    
    return torch.mean(jointangle_loss_vec)
# =========================================================================================
# Main Fitting Logic
# =========================================================================================

def project_points(points_3d, P):
    """
    使用投影矩阵将3D点投影到2D。
    """
    num_frames, num_points, _ = points_3d.shape
    points_3d_hom = torch.cat([points_3d, torch.ones(num_frames, num_points, 1, device=points_3d.device)], dim=-1)
    points_2d_hom = torch.bmm(P, points_3d_hom.transpose(1, 2)).transpose(1, 2)
    points_2d = points_2d_hom[..., :2] / (points_2d_hom[..., 2:3] + 1e-8)
    return points_2d

def calculate_sliding_window_loss(pred_2d, gt_2d, joint_indices, window_size=3):
    """
    (新增) 计算带滑动窗口平滑的2D关键点损失。
    """
    # 1. (新) 首先从预测和真值中选择我们关心的关节点
    # 这样可以避免不匹配的张量大小，并且计算效率更高
    pred_selected = pred_2d[:, joint_indices]
    gt_selected = gt_2d[:, joint_indices]
    
    # 2. (新) 在这些选定的关节点上计算L2误差
    error_on_selected_joints = torch.sum((pred_selected - gt_selected)**2, dim=-1) # Shape: [N, num_selected_joints]
    
    # 3. 计算每帧的平均损失
    per_frame_loss = torch.mean(error_on_selected_joints, dim=1) # Shape: [N]
    
    # 4. 应用滑动窗口平均
    if window_size <= 1:
        return torch.mean(per_frame_loss)
        
    padding = (window_size - 1) // 2
    # --- (核心修正) ---
    # F.pad需要至少2D的输入，所以我们临时给per_frame_loss增加维度
    # 将一维张量 [N] 临时升维到三维 [1, 1, N] (Batch, Channel, Length)
    temp_loss = per_frame_loss.view(1, 1, -1)
    
    # 在三维张量上进行填充
    padded_temp_loss = F.pad(temp_loss, (padding, padding), mode='replicate')
    
    # 完成填充后，降维回一维
    padded_loss = padded_temp_loss.squeeze()
    # --- (修正结束) ---
    
    # 使用unfold创建滑动窗口
    windows = padded_loss.unfold(dimension=0, size=window_size, step=1)
    
    # 计算每个窗口的平均值
    smoothed_loss = torch.mean(windows, dim=1)
    
    # 返回整个序列平滑后的损失的平均值
    return torch.mean(smoothed_loss)

def fit_with_2d_loss(
    # --- Input Data ---
    initial_poses,
    #initial_betas,
    initial_expression,
    # --- Loss Targets (现在是2D) ---
    target_2d_front,
    target_2d_side,
    P_canonical_front_cam,
    P_virtual_side_cam,
    # --- Model & Optimizer Params ---
    smplx_model_path,
    num_iterations=200,
    lr=0.01,
    # --- Loss Weights ---
    w_2d_front=1.0,
    w_2d_side=1.0, 
    w_bio=5.0,
    # --- Bio Loss Params ---
    interjoint=1.0,
    device=torch.device("cpu")
):
    """
    (重构 V2) 使用纯2D损失和滑动窗口优化SMPL-X姿态，并冻结无关的身体部分。
    """
    num_frames = initial_poses.shape[0]

    # --- 1. 设置SMPL-X模型和优化器 ---
    
    # (核心修正) --- START ---
    
    # 步骤 A: 定义需要优化的 pose 参数的索引 (在54个可动参数中)
    # SMPL-X body_pose 中与手臂相关的索引 (Clavicles, Shoulders, Elbows, Wrists)
    # 索引相对于 poses_to_optimize[0:21]
    ARM_POSE_INDICES = [
        12, 13, # Clavicles (SMPL-X pose index 13, 14) -> poses_to_optimize index 12, 13
        15, 16, # Shoulders (SMPL-X pose index 16, 17) -> poses_to_optimize index 15, 16
        17, 18, # Elbows    (SMPL-X pose index 18, 19) -> poses_to_optimize index 17, 18
        19, 20, # Wrists    (SMPL-X pose index 20, 21) -> poses_to_optimize index 19, 20
    ]
    # 手部姿态的索引
    LHAND_POSE_INDICES = list(range(24, 39)) # poses_to_optimize index
    RHAND_POSE_INDICES = list(range(39, 54)) # poses_to_optimize index

    # 合并所有需要优化的索引
    OPTIMIZE_INDICES = sorted(list(set(ARM_POSE_INDICES + LHAND_POSE_INDICES + RHAND_POSE_INDICES)))
    print(f"Optimizer will only update {len(OPTIMIZE_INDICES)} pose parameters for arms and hands.")
    
    # 步骤 B: 创建一个完整的、可优化的姿态参数张量
    poses_to_optimize = torch.nn.Parameter(initial_poses[:, 1:, :].clone())
    
    # 步骤 C: 创建梯度掩码 (Gradient Mask)
    # 创建一个全零的掩码，然后只在需要优化的位置设置为1
    grad_mask = torch.zeros_like(poses_to_optimize)
    grad_mask[:, OPTIMIZE_INDICES, :] = 1.0
    
    # 步骤 D: 注册一个“钩子”(Hook)
    # 这个钩子会在每次计算完梯度后，自动将梯度乘以我们的掩码
    # 这样，所有被冻结部分的梯度都会自动变成0
    poses_to_optimize.register_hook(lambda grad: grad * grad_mask)

    # (核心修正) --- END ---
    fixed_global_orient = initial_poses[:, 0:1, :].clone()
    

    expression_tensor = initial_expression.clone()
    
    model = smplx.create(
        model_path=smplx_model_path, 
        model_type='smplx', 
        gender='neutral',
        use_pca=False, 
        flat_hand_mean=True, 
        batch_size=num_frames
    ).to(device)
    
    optimizer = torch.optim.Adam([poses_to_optimize], lr=lr)

    # --- 2. 准备常量 ---
    bof_left_tensor = torch.tensor(BOF_LEFT, dtype=torch.float32, device=device)
    bof_right_tensor = torch.tensor(BOF_RIGHT, dtype=torch.float32, device=device)
    #T_left = batch_euler2mat_zxy(torch.tensor(euler_coordtrans_LEFT, dtype=torch.float32, device=device)).unsqueeze(0).expand(num_frames, -1, -1, -1)
    #T_right = batch_euler2mat_zxy(torch.tensor(euler_coordtrans_RIGHT, dtype=torch.float32, device=device)).unsqueeze(0).expand(num_frames, -1, -1, -1)
    
    P_front_tensor = P_canonical_front_cam.unsqueeze(0).expand(num_frames, -1, -1)
    P_side_tensor = P_virtual_side_cam.unsqueeze(0).expand(num_frames, -1, -1)

    # --- 3. 优化循环 ---
    print(f"Starting 2D-2D optimization for {num_frames} frames...")
    for i in tqdm(range(num_iterations), desc="Fitting"):
        optimizer.zero_grad()
        
        # --- SMPL-X 前向传播 ---

        output = model(
            global_orient=fixed_global_orient,
            body_pose=poses_to_optimize[:, 0:21], 
            jaw_pose=poses_to_optimize[:, 21:22],
            left_hand_pose=poses_to_optimize[:, 24:39], 
            right_hand_pose=poses_to_optimize[:, 39:54],
            expression=expression_tensor, 
            return_verts=False
        )
        model_joints_3d = output.joints
        
        # --- (修改) 2D 投影 ---
        pred_2d_front = project_points(model_joints_3d, P_front_tensor)
        pred_2d_side = project_points(model_joints_3d, P_side_tensor)

        # --- (修改) Loss A: 2D Loss vs. Front View (滑动窗口) ---
        loss_2d_front = calculate_sliding_window_loss(
            pred_2d_front, target_2d_front, LOSS_JOINT_INDICES, window_size=3
        )

        # --- (修改) Loss B: 2D Loss vs. Side View (滑动窗口) ---
        loss_2d_side = calculate_sliding_window_loss(
            pred_2d_side, target_2d_side, LOSS_JOINT_INDICES, window_size=3
        )

        # # --- Loss C: Bio-Limit Hinge Loss (保持不变) ---
        # lhand_poses_rod = poses_to_optimize[:, 24:39]
        # R_lhand_local = rodrigues_to_rotation_matrix(lhand_poses_rod.reshape(-1, 3)).reshape(num_frames, 15, 3, 3)
        # R_lhand_global = coordtrans(R_lhand_local, T_left, transtype=1)
        # euler_lhand_global = batch_mat2euler_zxy(R_lhand_global.reshape(-1, 3, 3)).reshape(num_frames, 15, 3)
        # rhand_poses_rod = poses_to_optimize[:, 39:54]
        # R_rhand_local = rodrigues_to_rotation_matrix(rhand_poses_rod.reshape(-1, 3)).reshape(num_frames, 15, 3, 3)
        # R_rhand_global = coordtrans(R_rhand_local, T_right, transtype=1)
        # euler_rhand_global = batch_mat2euler_zxy(R_rhand_global.reshape(-1, 3, 3)).reshape(num_frames, 15, 3)
        # loss_bio_left = calculate_bio_loss(euler_lhand_global, bof_left_tensor, interjoint, device)
        # loss_bio_right = calculate_bio_loss(euler_rhand_global, bof_right_tensor, interjoint, device)
        # loss_bio = loss_bio_left + loss_bio_right
        
        # --- 总损失和优化步骤 ---
        total_loss = (w_2d_front * loss_2d_front) + (w_2d_side * loss_2d_side)# + (w_bio * loss_bio)
        #total_loss = w_bio * loss_bio
        total_loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            tqdm.write(f"Iter {i+1}/{num_iterations} | "
                       f"Total: {total_loss.item():.6f} | "
                       f"2D Front: {(w_2d_front * loss_2d_front).item():.6f} | "
                       f"2D Side: {(w_2d_side * loss_2d_side).item():.6f} | "
                       #f"Bio: {(w_bio * loss_bio).item():.6f}"
                       )

    print("Optimization finished.")
    final_poses = torch.cat([initial_poses[:, 0:1, :].clone(), poses_to_optimize.detach()], dim=1)
    return final_poses


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading ALIGNED front view data from: {args.view_a_npz}")
    front_data = dict(np.load(args.view_a_npz, allow_pickle=True))

    print(f"Loading ALIGNED side view data from: {args.view_b_npz}")
    side_data = dict(np.load(args.view_b_npz, allow_pickle=True))

    # --- (新增顺序调整) 先准备初始数据供 SMPL-X 生成 target 用 ---  # <<< 修改
    initial_poses_front = torch.tensor(front_data['poses'], dtype=torch.float32, device=device)          # <<< 修改
    initial_expression = torch.tensor(front_data['smplx_expr'], dtype=torch.float32, device=device) # <<< 修改
    initial_poses_side = torch.tensor(side_data['poses'], dtype=torch.float32, device=device)          # <<< 修改

    # --- (修改) 动态生成 2D Ground Truth：不再读取 smpl_kp3d / smpl_kp3d_aligned --- 
    print("Generating 2D ground truth targets from SMPL-X forward (no smpl_kp3d).")                # <<< 修改
    if 'P_canonical_cam' not in front_data:
        raise KeyError("Front view NPZ is missing 'P_canonical_cam'.")                             # <<< 修改
    if 'P_virtual_cam' not in side_data:
        raise KeyError("Side view NPZ is missing 'P_virtual_cam'.")                                # <<< 修改

    # 相机矩阵
    P_front_np = front_data['P_canonical_cam']
    P_side_np = side_data['P_virtual_cam']

    num_frames = initial_poses_front.shape[0]

    # --- 用 SMPL-X 根据 initial_poses/betas/expression 生成 3D 关节作为 target ---
    # 与拟合阶段保持一致的拆分：global + body(21) + jaw + (可选眼睛) + hands
    with torch.no_grad():                                                                           # <<< 修改
        model_target = smplx.create(                                                                # <<< 修改
            model_path=args.smplx_path,
            model_type='smplx',
            gender='neutral',
            use_pca=False,
            flat_hand_mean=True,
            batch_size=num_frames
        ).to(device)

        # 拆分 pose（full pose 包含 global 在第 0 维）
        global_orient = initial_poses_front[:, 0:1, :]                      # [B,1,3]
        body_pose     = initial_poses_front[:, 1:22, :]                     # [B,21,3]
        jaw_pose      = initial_poses_front[:, 22:23, :]                    # [B,1,3]
        left_hand     = initial_poses_front[:, 25:40, :]                    # [B,15,3]
        right_hand    = initial_poses_front[:, 40:55, :]                    # [B,15,3]

        out_target = model_target(                                    # <<< 修改
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            expression=initial_expression,                            # 使用表情参数
            return_verts=False
        )
        target_3d_front = out_target.joints                                # [B, J, 3]
        global_orient = initial_poses_side[:, 0:1, :]                      # [B,1,3]
        body_pose     = initial_poses_side[:, 1:22, :]                     # [B,21,3]
        jaw_pose      = initial_poses_side[:, 22:23, :]                    # [B,1,3]
        left_hand     = initial_poses_side[:, 25:40, :]                    # [B,15,3]
        right_hand    = initial_poses_side[:, 40:55, :]                    # [B,15,3]
        out_target = model_target(                                    # <<< 修改
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            expression=initial_expression,                            # 使用表情参数
            return_verts=False
        )
        target_3d_side = out_target.joints                                # [B, J, 3]
    # --- 相机投影，得到 2D target（前/侧视角使用相同 3D，仅相机不同） ---
    P_front = torch.tensor(P_front_np, dtype=torch.float32, device=device).unsqueeze(0).expand(num_frames, -1, -1)
    P_side  = torch.tensor(P_side_np, dtype=torch.float32, device=device).unsqueeze(0).expand(num_frames, -1, -1)

    with torch.no_grad():
        target_2d_front = project_points(target_3d_front, P_front)          # <<< 修改
        target_2d_side  = project_points(target_3d_side, P_side)           # <<< 修改

    # 运行拟合流程（保持不变）
    fitted_poses = fit_with_2d_loss(
        initial_poses=initial_poses_front,
        #initial_betas=initial_betas,
        initial_expression=initial_expression,
        target_2d_front=target_2d_front,
        target_2d_side=target_2d_side,
        P_canonical_front_cam=torch.tensor(P_front_np, dtype=torch.float32, device=device),
        P_virtual_side_cam=torch.tensor(P_side_np, dtype=torch.float32, device=device),
        smplx_model_path=args.smplx_path,
        num_iterations=args.num_iterations,
        lr=args.lr,
        w_2d_front=args.w_2d_front,
        w_2d_side=args.w_2d_side,
        w_bio=args.w_bio,
        device=device
    )

    # 保存（保持不变）
    output_data = front_data.copy()
    output_data['poses'] = fitted_poses.cpu().numpy()
    base, ext = os.path.splitext(args.view_a_npz)
    output_path = f"{base}_fused.npz"
    print(f"\nSaving fused result to {output_path}")
    np.savez(output_path, **output_data)
    print("Done.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse two aligned views using a dual 2D loss with temporal smoothing.")
    parser.add_argument("view_a_npz", type=str, help="Path to the ALIGNED front view NPZ file.")
    parser.add_argument("view_b_npz", type=str, help="Path to the ALIGNED side view NPZ file.")
    parser.add_argument("--smplx_path", type=str, default='../smplx', required=False, help="Path to the SMPL-X model directory.")
    
    parser.add_argument("--num_iterations", type=int, default=200, help="Number of optimization iterations.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    
    # (修改) 更新损失权重的名称和默认值
    parser.add_argument("--w_2d_front", type=float, default=0.1, help="Weight for the 2D front view loss.")
    parser.add_argument("--w_2d_side", type=float, default=0.1, help="Weight for the 2D side view loss.")
    parser.add_argument("--w_bio", type=float, default=0.0, help="Weight for the bio-limit hinge loss.")
    
    args = parser.parse_args()
    main(args)