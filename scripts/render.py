import cv2
import open3d as o3d
import numpy as np
import os

import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras
)
import matplotlib.pyplot as plt
import h5py
from PIL import Image
# from getmimicgen import get_world_pcd, get_pointcloud, matrix_c2w, transform_pointcloud

# os.environ["DISPLAY"] = ":10.0"

def points_o2p(points):
    """
    将Open3D点云转换为PyTorch3D格式的点云。
    
    参数:
        points: np.ndarray, shape=[N, 3]
    
    返回:
        坐标系转换后的ndarray，shape=[N, 3]
    """
    transform_matrix = np.array([[-1, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 0]])  # Open3D to PyTorch3D
    transformed_points = points @ transform_matrix
    return transformed_points

def cam_o2p(extrinsic_matrix):
    """
    将Open3D相机外参矩阵转换为PyTorch3D格式。
    
    参数:
        extrinsic_matrix: np.ndarray, shape=[4, 4]
    
    返回:
        转换后的ndarray，shape=[4, 4]
    """
    transform_matrix = np.array([[-1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]])  # Open3D to PyTorch3D
    transformed_matrix = transform_matrix @ extrinsic_matrix
    return transformed_matrix


def render_pcd(pcd, extrinsic_matrix, intrinsic_matrix, width=256, height=256, point_size=10):
    # 创建一个相机视角
    camera = o3d.camera.PinholeCameraParameters()
    camera.extrinsic = extrinsic_matrix  # 外参矩阵

    # fx = 0.5 * width / np.tan(120 * np.pi / 360)
    # fy = 0.5 * height / np.tan(120 * np.pi / 360)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx,
                                                         fy=fy, cx=intrinsic_matrix[0, 2],
                                                         cy=intrinsic_matrix[1, 2])  # 内参矩阵
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)  # 设置窗口不可见

    # 添加点云到可视化窗口
    vis.add_geometry(pcd)

    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True)

    # 设置点云的大小
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    # 更新渲染器和事件
    vis.poll_events()
    vis.update_renderer()

    # 保存当前渲染结果为图片
    img = vis.capture_screen_float_buffer()  # 保存为PNG文件

    img = np.asarray(img)
    img = (255 * img).astype(np.uint8)

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("outputs/images/render.png", img)

    # 关闭可视化窗口
    vis.destroy_window()
    return img


def pytorch3d_render(pcd, extrinsic_matrix, intrinsic_matrix, width=256, height=256, 
                     point_size=1.0, background_color=(0, 0, 0), return_depth=False,
                     visualize=False, title="Result", device=None, znear=0.01, points_per_pixel=10):
    """
    使用PyTorch3D渲染点云并可视化结果
    
    参数:
        pcd (np.ndarray, torch.Tensor, or o3d.geometry.PointCloud): 点云数据
        extrinsic_matrix (np.ndarray or torch.Tensor): 外参矩阵，形状为[4, 4]
        intrinsic_matrix (np.ndarray or torch.Tensor): 内参矩阵，形状为[3, 3]
        width (int): 渲染图像宽度，默认为256
        height (int): 渲染图像高度，默认为256
        point_size (float): 点的大小，默认为1.0
        background_color (tuple): 背景颜色，默认为黑色(0, 0, 0)
        return_depth (bool): 是否返回深度图，默认为False
        visualize (bool): 是否可视化结果，默认为True
        title (str): 可视化窗口标题，默认为"点云渲染结果"
    
    返回:
        dict: 包含渲染结果的字典，键包括'image'(RGB图像)和'depth'(深度图，可选)
    """
    # 确保输入数据是PyTorch张量并在GPU上
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 处理不同类型的点云输入
    if isinstance(pcd, o3d.geometry.PointCloud):
        # 从Open3D点云对象中提取点和颜色
        points = np.asarray(pcd.points)

        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            # 如果没有颜色，使用默认灰色
            colors = np.ones_like(points) * 0.5
        
        # 转换为PyTorch张量
        points = torch.tensor(points, dtype=torch.float32, device=device)
        colors = torch.tensor(colors, dtype=torch.float32, device=device)
        
        # 组合点和颜色
        pcd_tensor = torch.cat([points, colors], dim=1).unsqueeze(0)  # [1, N, 6]
    elif isinstance(pcd, np.ndarray):
        pcd_tensor = torch.tensor(pcd, dtype=torch.float32, device=device).unsqueeze(0)
    elif isinstance(pcd, torch.Tensor):
        pcd_tensor = pcd.to(device).unsqueeze(0) if pcd.ndim == 2 else pcd.to(device)
    else:
        raise ValueError("不支持的点云数据类型。请提供numpy数组、PyTorch张量或Open3D点云对象。")
    
    # 处理外参矩阵
    if isinstance(extrinsic_matrix, np.ndarray):
        extrinsic_matrix = torch.tensor(extrinsic_matrix, dtype=torch.float32, device=device)
    
    # 处理内参矩阵
    if isinstance(intrinsic_matrix, np.ndarray):
        intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=torch.float32, device=device)
    
    # 从内参矩阵提取焦距和主点
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # 计算视野角（field of view）
    fov_y = 2 * torch.atan2(
        torch.tensor(height / 2.0, dtype=torch.float32, device=device),
        fy
    ) * 180 / np.pi
    fov_x = 2 * torch.atan2(
        torch.tensor(width / 2.0, dtype=torch.float32, device=device),
        fx
    ) * 180 / np.pi
    
    # 从外参矩阵提取旋转矩阵和平移向量
    R = extrinsic_matrix[:3, :3]  # 旋转矩阵
    T = extrinsic_matrix[:3, 3]   # 平移向量
    
    # 创建相机
    cameras = FoVPerspectiveCameras(
        device=device,
        R=R.unsqueeze(0),
        T=T.unsqueeze(0),
        fov=fov_y.unsqueeze(0),
        znear=znear
    )
    
    # cameras = PerspectiveCameras(
    #     device=device,
    #     R=R.unsqueeze(0),
    #     T=T.unsqueeze(0),
    #     focal_length=((fx, fy),),
    #     principal_point=((cx, cy),),
    #     image_size=((height, width),)
    # )

    # 处理点云颜色
    if pcd_tensor.shape[2] == 3:
        # 创建默认颜色（灰色）
        colors = torch.ones_like(pcd_tensor) * 0.5
        points = pcd_tensor
    else:
        # 分离坐标和颜色
        points = pcd_tensor[:, :, :3]
        colors = pcd_tensor[:, :, 3:6]
    
    # 创建点云对象
    pointclouds = Pointclouds(points=points, features=colors)
    
    # 设置光栅化参数
    raster_settings = PointsRasterizationSettings(
        image_size=(height, width),
        radius=point_size / max(width, height),  # 归一化点大小
        points_per_pixel=points_per_pixel
    )
    
    # 创建渲染器
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=background_color)
    )
    
    # 渲染点云
    images = renderer(pointclouds)
    
    # 获取深度信息（如果需要）
    depth = None
    if return_depth:
        fragments = rasterizer(pointclouds)
        depth = fragments.zbuf[0, ..., 0].cpu().numpy()
    
    # 准备返回结果
    result = {
        'image': images[0, ..., :3].cpu().numpy()  # RGB图像
    }
    if return_depth:
        result['depth'] = depth
    
    # 可视化结果
    if visualize:
        plt.figure(figsize=(12, 5))
        
        # 显示渲染图像
        plt.subplot(1, 2 if return_depth else 1, 1)
        plt.imshow(result['image'])
        plt.title(title)
        plt.axis('off')
        
        # 显示深度图（如果有）
        if return_depth:
            plt.subplot(1, 2, 2)
            plt.imshow(result['depth'], cmap='jet')
            plt.title("Depth")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return result