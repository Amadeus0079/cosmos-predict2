import torch
import numpy as np
import open3d as o3d
import os
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from collections import OrderedDict
from typing import Tuple, Union, Optional
import torch.nn.functional as F


def pose2pytorch3d(camera_pose):
    """
    将相机位姿转换为 PyTorch3D 格式的 4x4 外参矩阵
    输入: camera_pose - [B,4,4](torch.Tensor),相机→世界变换
    输出: pytorch3d_mat - [B,4,4](torch.Tensor),世界→相机变换（适配 PyTorch3D 坐标系）
    """
    device = camera_pose.device
    
    # 1. 定义轴方向修正矩阵（Open3D 相机系 → PyTorch3D 相机系）
    # X轴翻转,Y轴翻转
    correction = torch.Tensor([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).to(device)

    world_to_cam_open3d = torch.inverse(camera_pose)

    pytorch3d_mat = correction @ world_to_cam_open3d  # 矩阵乘法
    pytorch3d_mat[:, :3, :3] = pytorch3d_mat[:, :3, :3].transpose(1, 2)

    return pytorch3d_mat


def compose_camera_extrinsic(lookat: np.ndarray, distance: float, azimuth: float, elevation: float) -> np.ndarray:
    '''
    Calculates the camera extrinsic matrix (T_world_camera, camera pose in world coordinates)
    based on lookat point, distance, azimuth, and elevation, respecting specific coordinate systems.

    World coordinate system:
        Z-axis: Vertical (up)
        Y-axis: Horizontal left
        X-axis: Horizontal forward

    Camera coordinate system:
        Z-axis: Forward
        Y-axis: Downward
        X-axis: Rightward

    Input Parameters:
        azimuth (float): Rotation around world's vertical Z-axis, in degrees.
                         0 deg looks +X (forward), 90 deg looks +Y (left) (counter-clockwise from +X).
        elevation (float): Rotation around camera's local X-axis, in degrees.
                           Negative values move camera up, positive down (e.g., -45 deg means camera looks up 45 deg).

    Args:
        lookat (np.ndarray): (x, y, z) - Point camera is looking at in world coordinates.
        distance (float): Distance from camera to the lookat point.
        azimuth (float): Azimuth angle in degrees.
        elevation (float): Elevation angle in degrees.

    Returns:
        np.ndarray: 4x4 camera pose (T_world_camera) which transforms points from camera frame to world frame.
    '''
    lookat = np.asarray(lookat, dtype=np.float32)

    # World coordinate system up vector (Z-axis is vertical)
    world_up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Convert angles to radians
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

    # Adjust elevation for "negative value moves camera up" convention
    # If elevation is -45 (up 45), effective_elevation_rad becomes +45 deg (pi/4)
    effective_elevation_rad = -elevation_rad

    # 1. Calculate camera position in world coordinates (T_world_camera's translation)
    # Based on spherical coordinates relative to 'lookat' point
    # azimuth=0 -> +X, elevation=0 -> horizontal
    cam_pos_x = distance * np.cos(effective_elevation_rad) * np.cos(azimuth_rad)
    cam_pos_y = distance * np.cos(effective_elevation_rad) * np.sin(azimuth_rad)
    cam_pos_z = distance * np.sin(effective_elevation_rad)
    camera_pos_world = lookat + np.array([cam_pos_x, cam_pos_y, cam_pos_z], dtype=np.float32)

    # 2. Build T_world_camera (camera pose in world)
    # Calculate the camera's axes in world coordinates
    # Camera Z-axis points from camera_pos_world towards lookat (Forward)
    camera_z_world = (lookat - camera_pos_world)
    camera_z_world /= np.linalg.norm(camera_z_world)

    # Camera X-axis (Rightward) in world coordinates
    # For robust cross product, especially when camera_z_world is aligned with world_up_vector
    # camera_x_world = np.cross(camera_z_world, world_up_vector) would give a vector to the *left*
    # if camera_z_world is in +X and world_up_vector is +Z (cross(X,Z) = -Y, which is right in new coord)
    # Let's test with new definition: cross([1,0,0], [0,0,1]) = [0, -1, 0] (World -Y, which is Right)
    camera_x_world = np.cross(camera_z_world, world_up_vector)
    # Handle gimbal lock case: camera_z_world is aligned with world_up_vector
    if np.linalg.norm(camera_x_world) < 1e-6:
        # Camera is looking straight up or straight down.
        # Arbitrarily pick world Y-axis as camera X-axis
        if camera_z_world[2] > 0: # Looking up (+Z)
             camera_x_world = np.array([0.0, 1.0, 0.0], dtype=np.float32) # World +Y (Left)
        else: # Looking down (-Z)
             camera_x_world = np.array([0.0, -1.0, 0.0], dtype=np.float32) # World -Y (Right)
    camera_x_world /= np.linalg.norm(camera_x_world)

    # Camera Y-axis (Downward) in world coordinates
    # In a right-handed system, cross(X, Z) = -Y
    # Here, we want Camera Y-axis to be Downward, which is opposite of the Up vector derived from cross(X,Z)
    camera_y_world = np.cross(camera_x_world, camera_z_world)
    # This cross product gives a vector that is UPWARD relative to the camera's XZ plane.
    # Since camera Y-axis is defined as DOWNWARD, we need to negate this vector.
    camera_y_world = -camera_y_world
    camera_y_world /= np.linalg.norm(camera_y_world)

    # Construct rotation matrix for T_world_camera
    # Columns are Camera X, Y, Z axes in World coordinates
    rot_matrix_world_camera = np.eye(3, dtype=np.float32)
    rot_matrix_world_camera[:, 0] = camera_x_world
    rot_matrix_world_camera[:, 1] = camera_y_world
    rot_matrix_world_camera[:, 2] = camera_z_world

    # Build T_world_camera homogeneous matrix
    t_world_camera = np.eye(4, dtype=np.float32)
    t_world_camera[:3, :3] = rot_matrix_world_camera
    t_world_camera[:3, 3] = camera_pos_world

    return t_world_camera


def create_camera_intrinsic(h: int, w: int, fovy: float) -> np.ndarray:
    '''
    Generates a 3x3 camera intrinsic matrix based on image dimensions and vertical field of view.

    Assumptions:
    - Principal point (cx, cy) is at the center of the image (w/2, h/2).
    - Pixels are square, meaning fx = fy (or fx = fy * aspect_ratio to maintain physical focal length).
      Here, we will calculate fy directly from fovy and h, then set fx = fy.

    Args:
        h (int): Image height in pixels.
        w (int): Image width in pixels.
        fovy (float): Vertical Field of View in degrees.

    Returns:
        np.ndarray: A 3x3 camera intrinsic matrix.
    '''
    if not isinstance(h, int) or h <= 0:
        raise ValueError("Image height (h) must be a positive integer.")
    if not isinstance(w, int) or w <= 0:
        raise ValueError("Image width (w) must be a positive integer.")
    if not isinstance(fovy, (float, int)) or fovy <= 0 or fovy >= 180:
        raise ValueError("Vertical FOV (fovy) must be a positive float/int less than 180 degrees.")

    # Convert fovy from degrees to radians
    fovy_rad = np.deg2rad(fovy)

    # Calculate focal length fy
    # tan(fovy/2) = (h/2) / fy
    # fy = (h / 2) / tan(fovy / 2)
    # Ensure fovy_rad / 2 is not too close to 0 or pi for tan
    if np.abs(np.tan(fovy_rad / 2)) < 1e-6:
        raise ValueError("fovy is too small, leading to division by zero or very large focal length.")
    
    fy = (h / 2.0) / np.tan(fovy_rad / 2.0)

    # Assuming square pixels, so fx = fy
    fx = fy

    # Principal point (center of the image)
    cx = w / 2.0
    cy = h / 2.0

    # Construct the intrinsic matrix
    intrinsic_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return intrinsic_matrix


def sample_cameras(
    lookat: np.ndarray,
    d_range: Tuple[float, float] = (0.5, 1.5),
    a_range: Tuple[float, float] = (-180.0, 180.0),
    e_range: Tuple[float, float] = (-90.0, 90.0),
    num_cameras: int = 1,
    random_seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Randomly samples camera poses (T_world_camera) within specified ranges using compose_camera_extrinsic.

    Args:
        lookat (np.ndarray): (x, y, z) - Point camera is looking at in world coordinates.
        d_range (Tuple[float, float]): (min_distance, max_distance) for camera distance.
        a_range (Tuple[float, float]): (min_azimuth, max_azimuth) for camera azimuth in degrees.
        e_range (Tuple[float, float]): (min_elevation, max_elevation) for camera elevation in degrees.
        num_cameras (int): The number of camera poses to sample.
        random_seed (int, optional): Seed for the random number generator for reproducibility. Defaults to None.

    Returns:
        np.ndarray: (num_cameras, 3) array of camera states (distance, azimuth, elevation). (range from -1 to 1)
        np.ndarray: (num_cameras, 4, 4) array of camera poses (T_world_camera).
    '''
    if random_seed is not None:
        np.random.seed(random_seed)

    sampled_states = []
    sampled_poses = []
    for _ in range(num_cameras):
        # 随机采样距离
        norm_distance = np.random.random()
        distance = (d_range[1] - d_range[0]) * norm_distance + d_range[0]
        # 随机采样方位角
        norm_azimuth = np.random.random()
        azimuth = (a_range[1] - a_range[0]) * norm_azimuth + a_range[0]
        # 随机采样俯仰角
        norm_elevation = np.random.random()
        elevation = (e_range[1] - e_range[0]) * norm_elevation + e_range[0]

        # 调用您提供的函数来计算相机位姿
        camera_state = np.array((norm_distance, norm_azimuth, norm_elevation))
        camera_pose = compose_camera_extrinsic(lookat, distance, azimuth, elevation)
        sampled_states.append(camera_state)
        sampled_poses.append(camera_pose)

    return np.array(sampled_states, dtype=np.float32), np.array(sampled_poses, dtype=np.float32)


def compose_camera_extrinsic_batched(
    lookat: torch.Tensor, 
    distance: torch.Tensor, 
    azimuth: torch.Tensor, 
    elevation: torch.Tensor
) -> torch.Tensor:
    '''
    Batched version of compose_camera_extrinsic using PyTorch.

    Args:
        lookat (torch.Tensor): (B, 3) or (3,) - Point camera is looking at.
        distance (torch.Tensor): (B,) or scalar - Distance from camera to lookat.
        azimuth (torch.Tensor): (B,) or scalar - Azimuth in degrees.
        elevation (torch.Tensor): (B,) or scalar - Elevation in degrees.

    Returns:
        torch.Tensor: (B, 4, 4) Camera extrinsic matrices (T_world_camera).
    '''
    # 1. 广播和形状处理
    # 确保所有输入至少是 1D 的，并且广播到相同的 batch size
    if lookat.ndim == 1:
        lookat = lookat.unsqueeze(0) # (1, 3)
    
    B = lookat.shape[0]
    device = lookat.device
    dtype = lookat.dtype

    # 辅助函数：处理标量输入广播到 (B,)
    def ensure_tensor(val, target_B, device, dtype):
        if not torch.is_tensor(val):
            val = torch.tensor(val, device=device, dtype=dtype)
        if val.ndim == 0:
            val = val.view(1).expand(target_B)
        elif val.ndim == 1 and val.shape[0] == 1:
            val = val.expand(target_B)
        return val

    distance = ensure_tensor(distance, B, device, dtype)
    azimuth = ensure_tensor(azimuth, B, device, dtype)
    elevation = ensure_tensor(elevation, B, device, dtype)
    
    # 确保 lookat 也是广播后的形状 (如果 lookat 只有 1 个但其他有 B 个)
    if lookat.shape[0] == 1 and B > 1:
        lookat = lookat.expand(B, 3)
    
    # World coordinate system up vector (Z-axis is vertical) -> (B, 3)
    world_up_vector = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand(B, 3)

    # Convert angles to radians
    azimuth_rad = torch.deg2rad(azimuth)
    elevation_rad = torch.deg2rad(elevation)

    # Adjust elevation
    effective_elevation_rad = -elevation_rad

    # 2. Calculate camera position in world coordinates
    cam_pos_x = distance * torch.cos(effective_elevation_rad) * torch.cos(azimuth_rad)
    cam_pos_y = distance * torch.cos(effective_elevation_rad) * torch.sin(azimuth_rad)
    cam_pos_z = distance * torch.sin(effective_elevation_rad)
    
    # Stack to form (B, 3) relative vector
    cam_pos_relative = torch.stack([cam_pos_x, cam_pos_y, cam_pos_z], dim=1)
    camera_pos_world = lookat + cam_pos_relative

    # 3. Build T_world_camera axes
    # Camera Z-axis (Forward): from camera to lookat
    camera_z_world = lookat - camera_pos_world
    camera_z_world = F.normalize(camera_z_world, p=2, dim=1)

    # Camera X-axis (Rightward)
    # cross product: (B, 3) x (B, 3) -> (B, 3)
    camera_x_world = torch.cross(camera_z_world, world_up_vector, dim=1)
    
    # Handle gimbal lock (vectorized)
    # Check if cross product length is close to 0
    x_norm = torch.linalg.norm(camera_x_world, dim=1, keepdim=True)
    mask_singularity = x_norm < 1e-6 # (B, 1)

    if mask_singularity.any():
        # Fallback vectors
        # If looking up (Z > 0), Right is +Y [0, 1, 0]
        # If looking down (Z < 0), Right is -Y [0, -1, 0]
        # We construct a tensor of shape (B, 3) for fallback
        fallback_right = torch.zeros_like(camera_x_world)
        # 这里的逻辑对应原代码: if z > 0: [0, 1, 0] else: [0, -1, 0]
        # 也就是 y 分量为 sign(z)
        sign_z = torch.sign(camera_z_world[:, 2:3])
        # sign(0) is 0, but we assume perfect vertical looks usually imply direction. 
        # Fix sign: if 0, treat as 1 (rare case for float)
        sign_z[sign_z == 0] = 1.0 
        
        fallback_right[:, 1] = sign_z.squeeze()
        
        # Apply fallback where mask is True
        camera_x_world = torch.where(mask_singularity, fallback_right, camera_x_world)

    camera_x_world = F.normalize(camera_x_world, p=2, dim=1)

    # Camera Y-axis (Downward)
    # Original logic: cross(X, Z) gives Up, so negate to get Down
    camera_y_world = torch.cross(camera_x_world, camera_z_world, dim=1)
    camera_y_world = -camera_y_world
    camera_y_world = F.normalize(camera_y_world, p=2, dim=1)

    # 4. Construct Matrix (B, 4, 4)
    # Rotation part (B, 3, 3)
    # Columns are X, Y, Z axes
    R = torch.stack([camera_x_world, camera_y_world, camera_z_world], dim=2)

    # Combine into 4x4
    # Create (B, 4, 4) identity
    T_world_camera = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    
    # Assign Rotation
    T_world_camera[:, :3, :3] = R
    
    # Assign Translation (Position)
    T_world_camera[:, :3, 3] = camera_pos_world

    return T_world_camera


def create_camera_intrinsic_torch(
    h: Union[int, float, torch.Tensor],
    w: Union[int, float, torch.Tensor],
    fovy: Union[float, torch.Tensor],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    '''
    Generates a 3x3 camera intrinsic matrix (or a batch of matrices)
    using PyTorch based on image dimensions and vertical field of view.

    Assumptions:
    - Principal point (cx, cy) is at the center of the image (w/2, h/2).
    - Pixels are square (fx = fy).

    Args:
        h (Union[int, float, torch.Tensor]): Image height in pixels. Can be scalar or (B,) tensor.
        w (Union[int, float, torch.Tensor]): Image width in pixels. Can be scalar or (B,) tensor.
        fovy (Union[float, torch.Tensor]): Vertical Field of View in degrees. Can be scalar or (B,) tensor.
        device (torch.device): Device to create tensors on. Defaults to "cpu".
        dtype (torch.dtype): Data type for the intrinsic matrix. Defaults to torch.float32.

    Returns:
        torch.Tensor: A (3, 3) intrinsic matrix or (B, 3, 3) batch of matrices.
    '''
    # 统一将所有输入转换为 PyTorch 张量，并移至指定设备和数据类型
    def _to_tensor(val):
        if not torch.is_tensor(val):
            return torch.tensor(val, device=device, dtype=dtype)
        return val.to(device=device, dtype=dtype)

    h_t = _to_tensor(h)
    w_t = _to_tensor(w)
    fovy_t = _to_tensor(fovy)

    # 确保所有输入在批处理时形状一致 (广播到最大的批次维度)
    # 例如，如果 h 是标量，w 是 (B,)，fovy 是 (B,)，则 h 会被广播到 (B,)
    h_t, w_t, fovy_t = torch.broadcast_tensors(h_t, w_t, fovy_t)

    # 确定输出的批次大小
    B = h_t.shape[0] if h_t.ndim > 0 else 1

    # 输入有效性检查 (现在是批处理的)
    if (h_t <= 0).any() or (w_t <= 0).any():
        raise ValueError("Image height (h) and width (w) must be positive.")
    if (fovy_t <= 0).any() or (fovy_t >= 180).any():
        raise ValueError("Vertical FOV (fovy) must be positive and less than 180 degrees.")

    # Convert fovy from degrees to radians
    fovy_rad = torch.deg2rad(fovy_t)

    # Calculate focal length fy
    # tan(fovy/2) = (h/2) / fy => fy = (h / 2) / tan(fovy / 2)
    tan_half_fovy = torch.tan(fovy_rad / 2.0)

    # 检查 fovy 是否过小导致 tan 接近零
    if (torch.abs(tan_half_fovy) < 1e-6).any():
        raise ValueError("fovy is too small, leading to division by zero or very large focal length.")
    
    fy = (h_t / 2.0) / tan_half_fovy

    # Assuming square pixels, so fx = fy
    fx = fy

    # Principal point (center of the image)
    cx = w_t / 2.0
    cy = h_t / 2.0

    # Construct the intrinsic matrix (或批次矩阵)
    # 初始化为零矩阵，形状为 (B, 3, 3) 或 (3, 3)
    intrinsic_matrix = torch.zeros((B, 3, 3) if B > 1 else (3, 3), device=device, dtype=dtype)

    # 赋值主对角线和主点
    if B > 1:
        intrinsic_matrix[:, 0, 0] = fx
        intrinsic_matrix[:, 1, 1] = fy
        intrinsic_matrix[:, 0, 2] = cx
        intrinsic_matrix[:, 1, 2] = cy
        intrinsic_matrix[:, 2, 2] = 1.0
    else:
        intrinsic_matrix[0, 0] = fx.item() if fx.ndim==0 else fx
        intrinsic_matrix[1, 1] = fy.item() if fy.ndim==0 else fy
        intrinsic_matrix[0, 2] = cx.item() if cx.ndim==0 else cx
        intrinsic_matrix[1, 2] = cy.item() if cy.ndim==0 else cy
        intrinsic_matrix[2, 2] = 1.0

    return intrinsic_matrix


def sample_cameras_batched(
    lookat: torch.Tensor,
    d_range: Tuple[float, float] = (0.5, 1.5),
    a_range: Tuple[float, float] = (-180.0, 180.0),
    e_range: Tuple[float, float] = (-90.0, 90.0),
    num_cameras: int = 1,  # 这里 N 代表每个 lookat 采样的数量
    random_seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Batched version of sample_cameras using PyTorch.
    Generates `num_cameras` samples for EACH lookat point.

    Args:
        lookat (torch.Tensor): (B, 3) or (3,). The centers to look at.
        d_range (Tuple[float, float]): Min and max distance.
        a_range (Tuple[float, float]): Min and max azimuth (degrees).
        e_range (Tuple[float, float]): Min and max elevation (degrees).
        num_cameras (int): N, the number of cameras to sample per lookat point.
        random_seed (int): Random seed.
        device (torch.device): Device override.

    Returns:
        states: (B, N, 3) tensor containing [distance, azimuth, elevation] for each camera.
        poses:  (B, N, 4, 4) tensor containing T_world_camera matrices.
    '''
    if device is None:
        device = lookat.device
    
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # 1. 规范化输入 lookat 为 (B, 3)
    if lookat.ndim == 1:
        lookat = lookat.unsqueeze(0) # (1, 3)
    
    B = lookat.shape[0]
    N = num_cameras

    # 2. 采样参数: 形状 (B, N)
    # 我们需要为 B 个 lookat 中的每一个生成 N 个参数
    # def sample_uniform(r, shape):
    #     return r[0] + torch.rand(shape, device=device) * (r[1] - r[0])
    
    def range_change(r, x):
        return r[0] + x * (r[1] - r[0])

    # 采样出的形状是 (B, N)
    norm_dist = torch.rand((B, N), device=device)
    dist = range_change(d_range, norm_dist)
    norm_azim = torch.rand((B, N), device=device)
    azim = range_change(a_range, norm_azim)
    norm_elev = torch.rand((B, N), device=device)
    elev = range_change(e_range, norm_elev)
    

    # 3. 准备数据以调用 compose_camera_extrinsic_batched
    # compose 函数接受扁平化的批次输入，所以我们需要将数据展平为 (B*N, ...)
    
    # 将 lookat 从 (B, 3) 扩展并重塑为 (B*N, 3)
    # 逻辑: lookat[0] 重复 N 次, lookat[1] 重复 N 次...
    lookat_expanded = lookat.unsqueeze(1).expand(-1, N, -1).reshape(B * N, 3)
    
    # 将参数展平为 (B*N,)
    dist_flat = dist.flatten()
    azim_flat = azim.flatten()
    elev_flat = elev.flatten()

    # 4. 计算位姿
    # 输出形状为 (B*N, 4, 4)
    poses_flat = compose_camera_extrinsic_batched(lookat_expanded, dist_flat, azim_flat, elev_flat)

    # 5. 重塑输出以匹配 (B, N, ...)
    # 状态: (B, N, 3)
    # states = torch.stack([dist, azim, elev], dim=-1)
    states = torch.stack([norm_dist, norm_azim, norm_elev], dim=-1)
    
    # 位姿: (B*N, 4, 4) -> (B, N, 4, 4)
    poses = poses_flat.view(B, N, 4, 4)

    return states, poses


class CameraRenderer:
    
    def __init__(self, repo_id: str = None, cache_size: int = 10):
        self.repo_id = repo_id
        self.scene = None
        self.camera_keys = None
        
        self._cache = OrderedDict()
        self._cache_size = cache_size  # 显存中最多存多少个场景的点云
    
    def set_camera_keys(self, camera_list):
        self.camera_keys = camera_list
        return
    
    def get_pointcloud(self, rgb, depth, intrinsics, extrinsics, depth_scale=1, 
                           max_depth=10, save_path=None):
        """
        构建单样本点云（世界坐标系），生成Open3D点云并可选保存，返回点云坐标数组
        inputs:
            rgb: np.ndarray [H, W, 3]  # RGB图像（通道最后）
            depth: np.ndarray [H, W, 1]  # 深度图（通道最后）
            intrinsics: np.ndarray [3, 3]  # 相机内参
            extrinsics: np.ndarray [4, 4]  # 相机外参（相机到世界）
            depth_scale: float  # 深度缩放因子
            max_depth: float  # 最大有效深度值
            save_path: str | None  # 点云保存路径（如"output.ply"），None则不保存
        return:
            pointcloud_np: np.ndarray [N, 3]  # 点云坐标（世界坐标系）
        """
        # 获取图像尺寸
        H, W, C = rgb.shape
        assert C == 3, f"RGB图像通道数应为3，当前为{C}"
        assert depth.shape == (H, W, 1), f"深度图形状应为({H},{W},1)，当前为{depth.shape}"
        
        # 生成像素坐标网格 (H, W)
        u = np.linspace(0, W-1, W)
        v = np.linspace(0, H-1, H)
        u, v = np.meshgrid(u, v, indexing='xy')  # (H, W)
        
        # 展平坐标 (H*W,)
        u_flat = u.flatten()
        v_flat = v.flatten()
        
        # 深度值处理 (H*W,)：移除最后一维并展平
        z_flat = depth.reshape(-1) / depth_scale  # 等价于depth.squeeze(-1).reshape(-1)
        
        # 过滤有效深度点
        valid_mask = (z_flat > 0) & (z_flat < max_depth)
        # valid_mask = (z_flat > 0)
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        z_valid = z_flat[valid_mask]
        
        # 提取内参
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        # 转换为相机坐标系3D点 (N, 3)
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        points_cam = np.stack([x, y, z_valid], axis=1)
        
        # 转换到世界坐标系：齐次坐标变换
        points_hom = np.hstack([
            points_cam,
            np.ones((points_cam.shape[0], 1), dtype=np.float64)
        ])
        world_points = (extrinsics @ points_hom.T).T[:, :3]  # (N, 3)

        # 设置点云颜色
        rgb_flat = rgb.reshape(-1, 3)  # (H*W, 3)
        rgb_valid = rgb_flat[valid_mask]
        # 归一化颜色到0-1（兼容uint8和float格式）
        if rgb_valid.dtype == np.uint8:
            rgb_valid = rgb_valid / 255.0
        else:
            rgb_valid = np.clip(rgb_valid, 0.0, 1.0)
        
        # ---------------------- 构造并保存Open3D点云 ----------------------
        # 初始化Open3D点云对象
        # 保存点云（如果指定路径）
        if save_path is not None:
            pcd = o3d.geometry.PointCloud()
            # 设置点云坐标（需为float64格式）
            pcd.points = o3d.utility.Vector3dVector(world_points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(rgb_valid.astype(np.float64))

            # 自动创建保存目录
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 保存为PLY格式（支持颜色）
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"点云已保存至: {save_path}")
        
        # 返回世界坐标系下的点云坐标数组
        return world_points, rgb_valid

    def reconstruct_scene(self, images, depths, extrinsics, intrinsics, save_path=None):
        '''
        inputs:
            images: dict = {
                'cam_name': np.ndarray [h, w, c]
                ...
            }
            depths: dict = {
                'cam_name': np.ndarray [h, w, 1]
                ...
            }
            extrinsics: dict = {
                'cam_name': np.ndarray [4, 4]
                ...
            }
            intrinsics: dict = {
                'cam_name': np.ndarray [3, 3]
                ...
            }
        '''
        all_pos = []
        all_color = []
        
        for key in self.camera_keys:
            points_pos, points_color = self.get_pointcloud(images[key], depths[key], intrinsics[key], extrinsics[key])
            all_pos.append(points_pos)
            all_color.append(points_color)
            
        all_pos = np.concatenate(all_pos, axis=0)
        all_color = np.concatenate(all_color, axis=0)
        
        self.pc_points = all_pos
        self.pc_colors = all_color
        
        if save_path is not None:
            self.save_pointcloud_to_disk(all_pos, all_color, save_path)
        
        return all_pos, all_color
    
    def index2path(self, episode_index: int, frame_index: int):
        pcd_path = os.path.join(self.repo_id, f"pointclouds/episode_{episode_index:06d}/frame_{frame_index:06d}.ply")
        return pcd_path
    
    def save_pointcloud_to_disk(self, points, colors, path):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        o3d.io.write_point_cloud(path, pcd)
        return
    
    def load_scene_from_disk(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Point cloud file not found: {path}")
        
        pcd = o3d.io.read_point_cloud(path)
        self.pc_points = np.asarray(pcd.points, dtype=np.float32)
        self.pc_colors = np.asarray(pcd.colors, dtype=np.float32)
        return self.pc_points, self.pc_colors
    
    def load_scene_cached(self, episode_index, frame_index, base_dir):
        """
        训练时使用此函数。
        如果在缓存中，直接读取。
        否则从disk加载并存入缓存。
        """
        scene_id = f"episode_{episode_index:06d}.frame_{frame_index:06d}"
        
        # 1. 命中缓存
        if scene_id in self._cache:
            self._cache.move_to_end(scene_id)
            self.pc_points, self.pc_colors = self._cache[scene_id]
            return

        # 2. 未命中，加载
        ply_path = self.index2path(episode_index, frame_index)
        points, colors = self.load_scene_from_disk(ply_path)
        
        # 3. 存入缓存
        # 存入显存 Tensor 以加速后续渲染 (可选，取决于显存大小)
        # 这里先存 numpy，render_scene 里会转 tensor
        self._cache[scene_id] = (points, colors)
        self.pc_points = points
        self.pc_colors = colors

        # 4. 淘汰旧缓存
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
    
    def render_scene(self, extrinsics, intrinsics, point_size=1.0, 
                    background_color=(1, 1, 1), return_depth=False):
        '''
        Use the camera params to render the image in the cached pointcloud
        
        inputs:
            extrinsics: np.ndarray (4, 4)
            intrinsics: np.ndarray (3, 3)
        '''
        points_list = [torch.tensor(self.pc_points).float().cuda()]
        colors_list = [torch.tensor(self.pc_colors).float().cuda()]
        pointclouds = Pointclouds(
            points=points_list,  # [torch.Tensor(N, 3)]
            features=colors_list  # [torch.Tensor(N, 3)]
        )
        
        extrinsics = torch.tensor(extrinsics).float().cuda()
        intrinsics = torch.tensor(intrinsics).float().cuda()

        # 提取相机参数（批量）
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        height = int(cy * 2)
        width = int(cx * 2)

        # 计算视野角（批量）
        fov_y = 2 * torch.atan2(cy, fy) * 180 / torch.pi

        extrinsics = extrinsics.unsqueeze(0)
        extrinsics = pose2pytorch3d(extrinsics)  # convert to pytorch3d format
        
        # 提取旋转矩阵和平移向量（批量）
        R = extrinsics[:, :3, :3]  # (B, 3, 3)
        T = extrinsics[:, :3, 3]   # (B, 3)

        # 创建批量相机
        cameras = FoVPerspectiveCameras(
            R=R,  # 直接使用批量旋转矩阵
            T=T,  # 直接使用批量平移向量
            fov=fov_y,
            znear=0.01
        ).cuda()

        # 光栅化设置（批量共享）
        raster_settings = PointsRasterizationSettings(
            image_size=(height, width),
            radius=point_size / max(width, height),  # 归一化点大小
            points_per_pixel=10,
            bin_size=0
        )

        # 构建渲染器
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=background_color)
        )

        # 批量渲染（一次调用处理所有B个点云）
        images = renderer(pointclouds)  # (B, H, W, 3)
        render_images = images.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        render_images = render_images[0].detach().cpu().numpy()

        # 批量提取深度（如果需要）
        render_depths = None
        if return_depth:
            fragments = rasterizer(pointclouds)  # 批量获取光栅化结果
            render_depths = fragments.zbuf[..., 0].unsqueeze(3)  # (B, H, W, 1)
            render_depths = render_depths[0].detach().cpu().numpy()
            return render_images, render_depths
        
        return render_images


class BatchCameraRenderer:
    def __init__(self, repo_id: str = None, device: str = "cuda", cache_size: int = 50):
        self.repo_id = repo_id
        self.device = torch.device(device)
        self.cache_size = cache_size
        
        # 缓存：key -> (points_tensor, colors_tensor)
        self._cache = OrderedDict()
        
        # 默认内参
        self.default_intrinsics = None

    def set_intrinsics(self, intrinsics: torch.Tensor):
        """
        设置默认内参
        intrinsics: torch.Tensor (3, 3) or (B, 3, 3)
        """
        self.default_intrinsics = intrinsics.to(self.device)

    def index2path(self, episode_index: int, frame_index: int):
        # 保持与原逻辑一致的路径生成
        pcd_path = os.path.join(self.repo_id, f"pointclouds/episode_{episode_index:06d}/frame_{frame_index:06d}.ply")
        return pcd_path

    def _load_single_pcd(self, episode_idx, frame_idx):
        """
        加载单个点云，优先查显存缓存，未命中则读盘并放入缓存
        """
        key = f"{int(episode_idx)}_{int(frame_idx)}"
        
        # 1. 命中缓存
        if key in self._cache:
            self._cache.move_to_end(key) # 标记为最近使用
            return self._cache[key]
        
        # 2. 读取磁盘
        ply_path = self.index2path(int(episode_idx), int(frame_idx))
        if not os.path.exists(ply_path):
            # 如果文件不存在，返回空点云防止崩溃，或者抛出异常
            # 这里返回一个空的 Tensor
            return torch.zeros((0, 3), device=self.device), torch.zeros((0, 3), device=self.device)
            
        pcd = o3d.io.read_point_cloud(ply_path)
        points = torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)
        colors = torch.from_numpy(np.asarray(pcd.colors)).float().to(self.device)
        
        # 3. 更新缓存
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False) # 移除最旧的
        
        self._cache[key] = (points, colors)
        return points, colors

    def get_batch_pointclouds(self, episode_indices, frame_indices):
        """
        根据索引批量获取 PyTorch3D Pointclouds 对象
        """
        points_list = []
        features_list = []
        
        # 这是一个IO密集或CPU密集循环，但由于有缓存，通常很快
        for ep_idx, fr_idx in zip(episode_indices, frame_indices):
            pts, clrs = self._load_single_pcd(ep_idx, fr_idx)
            points_list.append(pts)
            features_list.append(clrs)
            
        # 构建 Pointclouds 对象 (PyTorch3D 会处理不同点数的 padding)
        return Pointclouds(points=points_list, features=features_list)

    def render_batch(self, extrinsics, episode_indices, frame_indices, 
                     intrinsics=None, point_size=1.0, 
                     background_color=(1, 1, 1), return_depth=False):
        """
        批量渲染函数
        Inputs:
            extrinsics: (B, 4, 4) 相机到世界的变换矩阵
            episode_indices: (B,)
            frame_indices: (B,)
            intrinsics: Optional (B, 3, 3) or (3, 3). 如果为None则使用set_intrinsics设置的值
            point_size: float
            return_depth: bool
        Returns:
            rgb: (B, H, W, 3) uint8
            depth: (B, H, W, 1) float32 (Optional)
        """
        B = extrinsics.shape[0]
        extrinsics = extrinsics.to(self.device).float()
        
        # 1. 处理内参
        if intrinsics is None:
            if self.default_intrinsics is None:
                raise ValueError("Intrinsics not provided and default intrinsics not set.")
            K = self.default_intrinsics
        else:
            K = intrinsics.to(self.device).float()
            
        # 确保 K 是 (B, 3, 3)
        if K.ndim == 2:
            K = K.unsqueeze(0).expand(B, -1, -1)
        
        # 假设 Batch 中所有图片的尺寸是一样的，取第一个 K 计算尺寸
        # 如果 batch 中图片尺寸不同，PyTorch3D 需要更复杂的设置，通常假设一致
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        
        # 这种计算方式假设 cx, cy 位于图像中心
        W = int(cx[0].item() * 2)
        H = int(cy[0].item() * 2)
        
        # 2. 准备点云数据 (利用 PyTorch3D 的并行结构)
        pointclouds = self.get_batch_pointclouds(episode_indices, frame_indices)
        
        # 3. 准备相机 (利用 PyTorch3D 的并行相机)
        # 计算 FOV (PyTorch3D 使用 FOV 来定义透视投影)
        # fov_y = 2 * arctan(h / (2 * fy))
        # 注意：这里假设 fy 和 cy 对应的高度关系一致
        fov_y = 2 * torch.atan2(cy, fy) * 180 / torch.pi
        
        # 转换外参坐标系
        extrinsics = pose2pytorch3d(extrinsics)  # convert to pytorch3d format
        
        # 提取旋转矩阵和平移向量（批量）
        R = extrinsics[:, :3, :3]  # (B, 3, 3)
        T = extrinsics[:, :3, 3]   # (B, 3)
        
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=R, 
            T=T, 
            fov=fov_y,
            znear=0.01
        )
        
        # 4. 设置光栅化器
        # radius 是相对于图像较长边的归一化大小
        raster_settings = PointsRasterizationSettings(
            image_size=(H, W),
            radius=point_size / max(H, W),
            points_per_pixel=10,
            bin_size=0 # 0 表示自适应或不分箱，对于复杂点云通常设为 0 较快
        )
        
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=background_color)
        )

        # 批量渲染（一次调用处理所有B个点云）
        images = renderer(pointclouds)  # (B, H, W, 3)
        render_images = images.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

        # 批量提取深度（如果需要）
        render_depths = None
        if return_depth:
            fragments = rasterizer(pointclouds)  # 批量获取光栅化结果
            render_depths = fragments.zbuf[..., 0].unsqueeze(3)  # (B, H, W, 1)
            return render_images, render_depths
        
        return render_images

    
if __name__ == "__main__":
    print(create_camera_intrinsic_torch(224, 224, fovy=75, device="cuda"))
    # # 假设环境配置
    # B = 4
    # H, W = 256, 256
    # repo_id = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/openpi/datasets/robocasa/PnPCounterToSink"
    
    # renderer = BatchCameraRenderer(repo_id=repo_id, cache_size=20)
    
    # # 模拟数据
    # episode_indices = torch.tensor([2, 28, 41, 54])
    # frame_indices = torch.tensor([6, 53, 75, 279]) 
    
    # # 模拟内参
    # fx = fy = 128.0
    # cx = cy = 128.0
    # intrinsics = torch.tensor([
    #     [fx, 0, cx],
    #     [0, fy, cy],
    #     [0, 0, 1]
    # ])
    # renderer.set_intrinsics(intrinsics) # 设置默认内参
    
    # # 模拟外参 (B, 4, 4) - 单位矩阵，即相机在原点
    # extrinsics = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    
    # # 渲染
    # # 注意：你需要确保 repo_id 路径下有真实的 .ply 文件才能运行成功
    # rgb, depth = renderer.render_batch(
    #     extrinsics, 
    #     episode_indices, 
    #     frame_indices, 
    #     return_depth=True,
    #     point_size=4
    # )
    # print(f"RGB Shape: {rgb.shape}")     # Should be (4, 256, 256, 3)
    # print(f"Depth Shape: {depth.shape}") # Should be (4, 256, 256, 1)
    
    # rgb_np = rgb[0].detach().cpu().numpy()
    # from PIL import Image
    # img = Image.fromarray(rgb_np)
    # img.save("outputs/tt.png")
