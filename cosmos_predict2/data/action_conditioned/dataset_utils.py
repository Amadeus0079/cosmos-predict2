# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_util.py
"""

import math
import os

import numpy as np
import torch
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
from pytorch3d.io import save_ply
from pytorch3d.structures import Pointclouds
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation


def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return rotm


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rotm


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rotm


def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def rotvec2rotm(rotvec):
    """Rotation vector to rotation matrix."""
    rotm = Rotation.from_rotvec(rotvec).as_matrix()
    
    return rotm


def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert isRotm(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # (-pi , pi]
    while x > np.pi:
        x -= 2 * np.pi
    while x <= -np.pi:
        x += 2 * np.pi
    while y > np.pi:
        y -= 2 * np.pi
    while y <= -np.pi:
        y += 2 * np.pi
    while z > np.pi:
        z -= 2 * np.pi
    while z <= -np.pi:
        z += 2 * np.pi
    return np.array([x, y, z])


def rotm2rotvec(R):
    rotvec = Rotation.from_matrix(R).as_rotvec()
    
    return rotvec

class Resize_Preprocess:
    def __init__(self, size):
        """
        Initialize the preprocessing class with the target size.
        Args:
        size (tuple): The target height and width as a tuple (height, width).
        """
        self.size = size

    def __call__(self, video_frames):
        """
        Apply the transformation to each frame in the video.
        Args:
        video_frames (torch.Tensor): A tensor representing a batch of video frames.
        Returns:
        torch.Tensor: The transformed video frames.
        """
        # Resize each frame in the video
        resized_frames = torch.stack([F.resize(frame, self.size, antialias=True) for frame in video_frames])
        return resized_frames


class Preprocess:
    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        clip = Preprocess.resize_scale(clip, self.size[0], self.size[1], interpolation_mode="bilinear")
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"

    @staticmethod
    def resize_scale(clip, target_height, target_width, interpolation_mode):
        target_ratio = target_height / target_width
        H = clip.size(-2)
        W = clip.size(-1)
        clip_ratio = H / W
        if clip_ratio > target_ratio:
            scale_ = target_width / W
        else:
            scale_ = target_height / H
        return torch.nn.functional.interpolate(clip, scale_factor=scale_, mode=interpolation_mode, align_corners=False)


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def save_pointcloud_to_ply(pointcloud, output_path, ascii=True):
    """
    将PyTorch3D的Pointclouds对象保存为PLY文件
    
    参数:
        pointcloud: PyTorch3D的Pointclouds对象
        output_path: 输出PLY文件路径（如"output.ply"）
        ascii: 是否以ASCII格式保存，默认为True（便于查看）
    """
    # 提取点坐标和颜色
    # 注意：Pointclouds可能包含多个点云，这里取第一个
    points = pointcloud.points_list()[0]  # 形状为(N, 3)
    colors = pointcloud.features_list()[0]  # 形状为(N, 3)，假设是RGB颜色
    
    # 确保颜色在[0, 255]范围（如果原来在[0,1]范围需要转换）
    if colors.max() <= 1.0:
        colors = (colors * 255).byte()
    
    # 保存为PLY文件
    save_ply(
        output_path,
        verts=points,
        ascii=ascii
    )
    print(f"点云已保存到: {output_path}")


def save_depth_tensor(
    depth_tensor, 
    save_dir="depth_maps",  # 保存文件夹路径
    base_name="depth",      # 图片基础文件名（后续会加通道号）
    colormap="viridis",     # 深度图配色（推荐viridis/plasma）
    normalize=True,         # 是否将深度值归一化到[0,1]
    dpi=300                 # 保存图片的分辨率
):
    """
    保存[1, N, H, W]格式的深度图张量（不显示，仅保存）
    
    参数说明：
    - depth_tensor: 输入深度图张量，形状必须为[1, N, H, W]
    - save_dir: 图片保存的文件夹（不存在会自动创建）
    - base_name: 图片基础名，最终保存名会是“base_name_通道号.png”
    - colormap: Matplotlib配色方案（深度图推荐用感知均匀的配色）
    - normalize: 是否归一化（若深度值有物理意义，可设为False直接保存原始值）
    - dpi: 保存图片的分辨率（dpi越高，图片越清晰）
    """
    # 1. 检查输入张量形状是否正确
    assert depth_tensor.ndim == 4 and depth_tensor.shape[0] == 1, \
        f"输入张量必须是[1, N, H, W]格式，当前形状是{depth_tensor.shape}"
    
    # 2. 创建保存文件夹（不存在则自动创建）
    os.makedirs(save_dir, exist_ok=True)  # exist_ok=True避免文件夹已存在时报错
    
    # 3. 去除batch维度（从[1, N, H, W]变为[N, H, W]）
    depth_map = depth_tensor.squeeze(0)  # 挤压第0维（batch维）
    N, H, W = depth_map.shape  # N=通道数，H=高度，W=宽度
    
    # 4. 循环保存每个通道的深度图
    for channel_idx in range(N):
        # 4.1 提取当前通道的深度图（从[N, H, W]变为[H, W]）
        current_depth = depth_map[channel_idx]
        
        # 4.2 张量转numpy数组（若在GPU上，需先移到CPU；detach()脱离计算图）
        current_depth_np = current_depth.cpu().detach().numpy()
        
        # 4.3 （可选）归一化深度值到[0,1]（避免因深度范围过大导致显示模糊）
        if normalize:
            # 计算当前通道的最大/最小值（避免跨通道归一化导致信息丢失）
            depth_min = current_depth_np.min()
            depth_max = current_depth_np.max()
            # 加1e-8避免分母为0（当所有深度值相同时）
            current_depth_np = (current_depth_np - depth_min) / (depth_max - depth_min + 1e-8)
        
        # 4.4 创建画布并绘制深度图（不显示，仅用于保存）
        plt.figure(figsize=(W/100, H/100), dpi=dpi)  # 画布尺寸匹配图片分辨率
        ax = plt.gca()  # 获取当前坐标轴
        
        # 绘制深度图（extent确保图片无拉伸）
        im = ax.imshow(current_depth_np, cmap=colormap, extent=[0, W, 0, H])
        
        # （可选）添加颜色条（标注深度值对应关系，增强可读性）
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)  # shrink调整颜色条大小
        cbar.set_label("Depth Value", fontsize=8)  # 颜色条标签
        
        # （可选）添加标题（标注通道号）
        ax.set_title(f"Depth Map - Channel {channel_idx}", fontsize=10)
        
        # 关闭坐标轴（避免多余边框，专注深度图本身）
        ax.axis("off")
        
        # 4.5 保存图片（bbox_inches="tight"去除多余空白）
        save_path = os.path.join(save_dir, f"{base_name}_{channel_idx}.png")
        plt.savefig(
            save_path,
            bbox_inches="tight",  # 裁剪多余空白（重要，避免图片周围有黑边）
            pad_inches=0.1,       # 保留少量边距（可选）
            dpi=dpi
        )
        
        # 4.6 关闭当前画布（释放内存，避免多通道时内存溢出）
        plt.close()
        print(f"已保存深度图：{save_path}")
        

def pytorch3d_pointcloud_to_open3d(pytorch3d_pc, device='cpu'):
    """
    将PyTorch3D的Pointclouds对象转换为Open3D的PointCloud对象
    
    参数:
    - pytorch3d_pc: PyTorch3D的Pointclouds实例
    - device: 设备（'cpu'或'cuda'）
    
    返回:
    - o3d_pc: Open3D的PointCloud对象
    """
    # 提取点坐标 (形状: [batch_size, num_points, 3])
    points = pytorch3d_pc.points_padded()  # 获取填充后的点坐标
    points = points[0].to(device)  # 取第一个batch，并转移到指定设备
    points_np = points.cpu().detach().numpy()  # 转换为numpy数组
    
    # 创建Open3D点云对象
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(points_np)
    
    # 如果有点云颜色信息，添加颜色
    colors = pytorch3d_pc.features_padded()[0].to(device)  # 获取第一个batch的颜色
    colors_np = colors.cpu().detach().numpy()  # 转换为numpy数组
    # 确保颜色在[0, 1]范围内
    if colors_np.max() > 1.0:
        colors_np = colors_np / 255.0
    o3d_pc.colors = o3d.utility.Vector3dVector(colors_np)
    
    return o3d_pc


def merge_pointclouds(pcl_list):
    # pcl_list 是 Pointclouds 对象的列表
    all_points = []
    all_feats = []
    for pcl in pcl_list:
        pts = pcl.points_list()[0]      # 取出 (N,3) 点坐标
        feats = pcl.features_list()[0]  # 取出 (N,C) 特征
        all_points.append(pts)
        all_feats.append(feats)

    merged_points = torch.cat(all_points, dim=0)
    merged_feats = torch.cat(all_feats, dim=0)
    return Pointclouds(points=[merged_points], features=[merged_feats])


def pointclouds_world2cam(pointclouds, extrinsics):
    points = pointclouds.points_list()[0]
    feats = pointclouds.features_list()[0]
    device = points.device
    points_homogeneous = torch.hstack((points, torch.ones((points.shape[0], 1)).to(device)))  # (N, 4)
    cam_points = (torch.inverse(extrinsics) @ points_homogeneous.T).T[:, :3]
    return Pointclouds(points=[cam_points], features=[feats])


def pointclouds_world2cam_batch(pointclouds, extrinsics):
    '''
    pointclouds: Pointclouds
    extrinsics: torch.Tensor [B, 4, 4]
    '''
    points = pointclouds.points_list()
    feats = pointclouds.features_list()
    device = points.device
    points_homogeneous = torch.stack((points, torch.ones((points.shape[0], points.shape[1], 1)).to(device)), dim=2)  # (B, N, 4)
    cam_points = (torch.inverse(extrinsics) @ points_homogeneous.transpose(0, 2, 1)).transpose(0, 1, 2)[..., :3]
    return Pointclouds(points=[cam_points], features=[feats])


def campose_to_pytorch3d_mat(camera_pose):
    """
    将相机位姿转换为 PyTorch3D 格式的 4x4 外参矩阵
    输入: camera_pose - [4,4](torch.Tensor),相机→世界变换
    输出: pytorch3d_mat - [4,4](torch.Tensor),世界→相机变换（适配 PyTorch3D 坐标系）
    """
    # 1. 定义轴方向修正矩阵（Open3D 相机系 → PyTorch3D 相机系）
    # X轴翻转,Y轴翻转
    correction = torch.Tensor([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).cuda()

    world_to_cam_open3d = torch.inverse(camera_pose)

    pytorch3d_mat = correction @ world_to_cam_open3d  # 矩阵乘法
    pytorch3d_mat[:3, :3] = pytorch3d_mat[:3, :3].T

    return pytorch3d_mat


def campose_to_pytorch3d_mat_batch(camera_pose):
    """
    将相机位姿转换为 PyTorch3D 格式的 4x4 外参矩阵
    输入: camera_pose - [B,4,4](torch.Tensor),相机→世界变换
    输出: pytorch3d_mat - [B,4,4](torch.Tensor),世界→相机变换（适配 PyTorch3D 坐标系）
    """
    # 1. 定义轴方向修正矩阵（Open3D 相机系 → PyTorch3D 相机系）
    # X轴翻转,Y轴翻转
    correction = torch.Tensor([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).cuda()

    world_to_cam_open3d = torch.inverse(camera_pose)

    pytorch3d_mat = correction @ world_to_cam_open3d  # 矩阵乘法
    pytorch3d_mat[:, :3, :3] = pytorch3d_mat[:, :3, :3].transpose(1, 2)

    return pytorch3d_mat


def tensor_to_video_opencv(
    tensor: torch.Tensor,
    output_path: str = "output.mp4",
    fps: int = 6,  # 视频帧率（根据需求调整，如10、25）
    is_normalized: bool = True  # 若Tensor已归一化（[-1,1]或[0,1]），设为True
):
    """
    将(C, T, H, W)的PyTorch Tensor保存为视频
    Args:
        tensor: 输入Tensor，形状为(C, T, H, W)，C=3（RGB）或C=1（灰度）
        output_path: 输出视频路径（需带后缀，如.mp4、.avi）
        fps: 视频帧率
        is_normalized: Tensor是否处于归一化范围（True=[-1,1]或[0,1]，False=[0,255]）
    """
    # -------------------------- 1. 基础格式检查 --------------------------
    C, T, H, W = tensor.shape
    assert C in [1, 3], f"通道数必须为1（灰度）或3（RGB），当前为{C}"

    # -------------------------- 2. 调整Tensor格式 --------------------------
    # 1. 迁移到CPU（OpenCV不支持GPU Tensor）
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 2. 维度转换：(C, T, H, W) → (T, H, W, C)（视频库要求的顺序）
    tensor = tensor.permute(1, 2, 3, 0)  # 交换维度：帧数(T)放第一，通道(C)放最后

    # 3. 数据范围缩放：[-1,1]或[0,1] → [0,255]
    if is_normalized:
        # 若输入是[-1,1]（如GAN输出），先转为[0,1]，再缩放为[0,255]
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2  # [-1,1] → [0,1]
        tensor = tensor * 255  # [0,1] → [0,255]
        # 4. 截断溢出值（避免像素值超出0-255，如计算误差导致的-0.1或255.5）
        tensor = torch.clamp(tensor, 0, 255)
        # 5. 转换为uint8类型（视频像素标准类型）
        tensor = tensor.to(torch.uint8)

    # 6. 灰度图处理（OpenCV默认保存3通道视频，单通道需扩展为3通道）
    if C == 1:
        tensor = tensor.repeat(1, 1, 3)  # (T, H, W, 1) → (T, H, W, 3)

    # -------------------------- 3. 初始化视频写入器 --------------------------
    # 获取视频编码格式（根据输出后缀自动选择，MP4推荐使用mp4v编码）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 对应.mp4格式（兼容性好）
    # 若输出为.avi格式，可改为 fourcc = cv2.VideoWriter_fourcc(*"XVID")
    
    # 初始化VideoWriter（参数：路径、编码、帧率、分辨率(H,W)）
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (W, H)  # 注意：OpenCV要求分辨率为 (宽, 高)，即(W, H)
    )

    # -------------------------- 4. 逐帧写入视频 --------------------------
    for t in range(T):
        frame = tensor[t].numpy()  # Tensor → numpy数组（OpenCV支持numpy格式）
        # OpenCV默认使用BGR通道顺序，而我们的Tensor是RGB，需转换通道
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    # -------------------------- 5. 释放资源 --------------------------
    video_writer.release()
    cv2.destroyAllWindows()  # 关闭OpenCV的临时窗口（若有）
    print(f"视频已保存到：{output_path}")
        