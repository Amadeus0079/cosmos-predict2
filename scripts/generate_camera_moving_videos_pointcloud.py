#!/usr/bin/env python3
"""
生成静态场景的参考视频（基于点云渲染）
使用第一帧的所有view构建点云，然后使用原视频中的相机参数逐帧渲染
【已添加模块计时功能】
"""

import h5py
import numpy as np
import cv2
import os
import json
import argparse
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import torch
import time  # 引入time模块
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

dummy_tensor = torch.eye(3, device='cuda' if torch.cuda.is_available() else 'cpu')
_ = torch.inverse(dummy_tensor)

# --- 新增计时工具类 ---
class SectionTimer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

def find_hdf5_file(task_name, base_dir):
    """Find HDF5 file for a given task."""
    task_dir = os.path.join(base_dir, task_name)
    for root, dirs, files in os.walk(task_dir):
        for file in files:
            if file == "demo_im256_ep100_cam2+4.hdf5":
                return os.path.join(root, file)
    return None


def get_demo_key_from_episode_id(hdf5_path, episode_id):
    """
    从HDF5文件中获取对应episode_id的demo_key
    episode_id就是排序后demo列表的索引
    """
    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            return None

        data_group = f['data']
        episodes = sorted([key for key in data_group.keys() if key.startswith('demo_')])

        if episode_id < 0 or episode_id >= len(episodes):
            return None

        return episodes[episode_id]


def rgbd_to_pointcloud_batch(rgb_batch, depth_batch, intrinsics_batch, extrinsics_batch, depth_scale=1, device='cuda'):
    """
    Batch construct pointclouds in world coordinates
    """
    rgb_batch = rgb_batch.to(device)
    depth_batch = depth_batch.to(device)
    intrinsics_batch = intrinsics_batch.to(device)
    extrinsics_batch = extrinsics_batch.to(device)

    rgb_batch = rgb_batch / 255.0
    B, C, H, W = rgb_batch.shape

    # Generate pixel coordinate grid
    u = torch.linspace(0, W-1, W, device=device)
    v = torch.linspace(0, H-1, H, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')  # (H, W)
    u = u.flatten()  # (H*W,)
    v = v.flatten()  # (H*W,)
    u_batch = u.unsqueeze(0).repeat(B, 1)
    v_batch = v.unsqueeze(0).repeat(B, 1)

    # Depth processing
    z_batch = depth_batch.squeeze(1).reshape(B, -1) / depth_scale  # (B, H*W)
    valid_mask = (z_batch > 0) & (z_batch < 10)  # (B, H*W)

    # Extract intrinsics
    fx_batch = intrinsics_batch[:, 0, 0]  # (B,)
    fy_batch = intrinsics_batch[:, 1, 1]  # (B,)
    cx_batch = intrinsics_batch[:, 0, 2]  # (B,)
    cy_batch = intrinsics_batch[:, 1, 2]  # (B,)

    # Convert to camera coordinates
    x = (u_batch - cx_batch.unsqueeze(1)) * z_batch / fx_batch.unsqueeze(1)  # (B, H*W)
    y = (v_batch - cy_batch.unsqueeze(1)) * z_batch / fy_batch.unsqueeze(1)  # (B, H*W)
    points_cam = torch.stack([x, y, z_batch], dim=2)  # (B, H*W, 3)

    # Filter invalid points
    valid_points_cam = []
    valid_colors = []
    for b in range(B):
        mask = valid_mask[b]
        valid_points = points_cam[b, mask]  # (N_b, 3)
        valid_points_cam.append(valid_points)
        rgb = rgb_batch[b].permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
        valid_colors.append(rgb[mask])  # (N_b, 3)

    # Transform to world coordinates
    world_points_list = []
    for b in range(B):
        points_hom = torch.hstack([
            valid_points_cam[b],
            torch.ones(valid_points_cam[b].shape[0], 1, device=device)
        ])  # (N_b, 4)
        world_points = (extrinsics_batch[b] @ points_hom.T).T[:, :3]  # (N_b, 3)
        world_points_list.append(world_points)

    # Create Pointclouds object with batch dimension
    pointclouds = Pointclouds(
        points=world_points_list,
        features=valid_colors
    )
    return pointclouds


def merge_pointclouds_batch(pointclouds_list):
    """
    Merge multiple Pointclouds objects into a single one.
    """
    B = len(pointclouds_list[0])
    merged_points = []
    merged_features = []

    for b in range(B):
        points_b = []
        features_b = []
        for pc in pointclouds_list:
            points_b.append(pc.points_list()[b])
            features_b.append(pc.features_list()[b])
        merged_points.append(torch.cat(points_b, dim=0))
        merged_features.append(torch.cat(features_b, dim=0))

    return Pointclouds(points=merged_points, features=merged_features)


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
    pytorch3d_mat[:, :3, :3] = pytorch3d_mat[:, :3, :3].transpose(1, 2).clone()

    return pytorch3d_mat


def render_pointcloud_batch(pointclouds, extrinsic_matrixs, intrinsic_matrixs,
                           point_size=1.0, height=256, width=256,
                           background_color=(1, 1, 1), return_depth=True, device='cuda'):
    """
    Batch render pointclouds
    """
    B = extrinsic_matrixs.shape[0]

    fx = intrinsic_matrixs[:, 0, 0]
    fy = intrinsic_matrixs[:, 1, 1]
    cx = intrinsic_matrixs[:, 0, 2]
    cy = intrinsic_matrixs[:, 1, 2]

    fov_y = 2 * torch.atan2(torch.tensor(height / 2.0, device=device), fy) * 180 / torch.pi

    R = extrinsic_matrixs[:, :3, :3]
    T = extrinsic_matrixs[:, :3, 3]

    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        fov=fov_y,
        znear=0.01
    )

    raster_settings = PointsRasterizationSettings(
        image_size=(height, width),
        radius=point_size / max(width, height),
        points_per_pixel=5,
        bin_size=0
    )

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=background_color)
    )

    images = renderer(pointclouds)  # (B, H, W, 3)
    render_images = images.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

    render_depths = None
    if return_depth:
        fragments = rasterizer(pointclouds)
        render_depths = fragments.zbuf[..., 0].unsqueeze(-1)  # (B, H, W, 1)

    return render_images, render_depths


def load_video_frames(video_path, frame_ids):
    """Load specific frames from video."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    vr.seek(0)
    frame_data = vr.get_batch(frame_ids).asnumpy()
    return frame_data


def load_depth_frames(depth_path, frame_ids):
    """Load specific depth frames."""
    all_depths = np.load(depth_path)
    depths = all_depths[frame_ids]
    return depths


def process_episode_ref_videos_pointcloud(
    task_name, episode_id, split, hdf5_path, annotation_path,
    output_dir, camera_width=256, camera_height=256,
    batch_size=16, point_size=4.0, device='cuda'
):
    """
    为单个episode生成静态场景的参考视频（基于点云渲染）
    包含耗时统计
    """
    episode_start_time = time.time()
    timings = {} # 存储各阶段耗时

    try:
        # Read annotation
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # Get all available cameras
        gt_cams = ['robot0_agentview_center', 'robot0_eye_in_hand']
        randomview_cameras = [cam_name for cam_name in annotation['videos'].keys()
                             if cam_name.startswith('robot0_randomview_')]
        all_cams = gt_cams + randomview_cameras

        # Get number of frames from any camera
        first_cam = list(annotation['videos'].keys())[0]
        num_frames = len(annotation['extrinsic_matrix'][first_cam])

        # Output directory
        task_output_dir = os.path.join(output_dir, task_name)
        video_output_dir = os.path.join(output_dir, task_name, "videos", split, str(episode_id))
        os.makedirs(video_output_dir, exist_ok=True)

        generated_count = 0
        skipped_count = 0
        
        # 初始化累计时间
        total_load_time = 0
        total_pc_build_time = 0
        total_render_time = 0
        total_save_time = 0

        # Process each randomview camera
        for cam_name in randomview_cameras:
            # Check if ref video already exists
            ref_video_path = os.path.join(video_output_dir, f"{cam_name}_ref.mp4")
            ref_depth_path = os.path.join(video_output_dir, f"{cam_name}_ref.npy")

            if os.path.exists(ref_video_path) and os.path.exists(ref_depth_path):
                skipped_count += 1
                continue

            # --- Stage 1: Data Loading ---
            with SectionTimer("Load Data") as t:
                pointclouds_list = []
                for view_cam in all_cams:
                    if view_cam not in annotation['videos']:
                        continue

                    # Load first frame data
                    video_path = annotation['videos'][view_cam]['video_path']
                    video_path = os.path.join(task_output_dir, video_path)
                    depth_path = annotation['videos'][view_cam]['depth_path']
                    depth_path = os.path.join(task_output_dir, depth_path)

                    # Load frame 0
                    frame = load_video_frames(video_path, [0])[0]  # (H, W, C)
                    depth = load_depth_frames(depth_path, [0])[0]  # (H, W, 1)

                    # Convert to torch tensors
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)
                    depth_tensor = torch.from_numpy(depth).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 1, H, W)

                    # Get camera parameters for frame 0
                    extrinsic = torch.tensor(annotation['extrinsic_matrix'][view_cam][0]).unsqueeze(0).to(device)  # (1, 4, 4)
                    intrinsic = torch.tensor(annotation['intrinsic_matrix'][view_cam][0]).unsqueeze(0).to(device)  # (1, 3, 3)

                    # --- Stage 2: PC Construction (Part A - per view) ---
                    # 注意：这里PC构建和Load交织在一起，简化起见计入Load或者拆分
                    # 为了更精确，我们在内部再拆分一下，虽然会增加一点点代码复杂度
                    # 重新调整计时策略：rgbd_to_pointcloud_batch 是纯计算，load是IO
                    
                    pc = rgbd_to_pointcloud_batch(
                        frame_tensor, depth_tensor, intrinsic, extrinsic, device=device
                    )
                    pointclouds_list.append(pc)
            
            total_load_time += t.duration

            # --- Stage 2: PC Merge ---
            with SectionTimer("Merge PC") as t:
                merged_pointcloud = merge_pointclouds_batch(pointclouds_list)
            total_pc_build_time += t.duration

            # Now render from target camera viewpoints across all frames
            extrinsic_matrices = torch.tensor(annotation['extrinsic_matrix'][cam_name]).to(device)  # (T, 4, 4)
            intrinsic_matrices = torch.tensor(annotation['intrinsic_matrix'][cam_name]).to(device)  # (T, 3, 3)
            extrinsic_matrices_render = campose_to_pytorch3d_mat_batch(extrinsic_matrices)

            # --- Stage 3: Rendering ---
            with SectionTimer("Rendering") as t:
                all_render_images = []
                all_render_depths = []

                for batch_start in range(0, num_frames, batch_size):
                    batch_end = min(batch_start + batch_size, num_frames)
                    batch_extrinsics = extrinsic_matrices_render[batch_start:batch_end]
                    batch_intrinsics = intrinsic_matrices[batch_start:batch_end]

                    batch_size_actual = batch_end - batch_start
                    batch_pointcloud = Pointclouds(
                        points=[merged_pointcloud.points_list()[0]] * batch_size_actual,
                        features=[merged_pointcloud.features_list()[0]] * batch_size_actual
                    )

                    render_images, render_depths = render_pointcloud_batch(
                        batch_pointcloud,
                        batch_extrinsics,
                        batch_intrinsics,
                        point_size=point_size,
                        height=camera_height,
                        width=camera_width,
                        background_color=(1, 1, 1),
                        return_depth=True,
                        device=device
                    )

                    all_render_images.append(render_images.cpu())
                    all_render_depths.append(render_depths.cpu())

                all_render_images = torch.cat(all_render_images, dim=0)
                all_render_depths = torch.cat(all_render_depths, dim=0)
            
            total_render_time += t.duration

            # --- Stage 4: Saving ---
            with SectionTimer("Saving") as t:
                # Save video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_fps = 20.0
                out = cv2.VideoWriter(ref_video_path, fourcc, output_fps, (camera_width, camera_height))

                for frame_idx in range(num_frames):
                    frame = all_render_images[frame_idx].numpy()
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)

                out.release()

                # Save depth
                depth_array = all_render_depths.numpy()
                np.save(ref_depth_path, depth_array)
            
            total_save_time += t.duration

            generated_count += 1

        total_elapsed = time.time() - episode_start_time
        
        if generated_count > 0:
            # 只有在真正生成了视频时才返回详细的计时信息
            msg = (f"[Ep {episode_id}] Gen:{generated_count} Skip:{skipped_count} | "
                   f"Total:{total_elapsed:.1f}s | "
                   f"Load+PC:{total_load_time+total_pc_build_time:.2f}s | "
                   f"Render:{total_render_time:.2f}s | "
                   f"Save:{total_save_time:.2f}s")
            return msg
        elif skipped_count > 0:
            return f"[Ep {episode_id}] 跳过 (已存在)"
        return None

    except Exception as e:
        import traceback
        return f"错误处理 episode {episode_id}: {str(e)}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="生成静态场景的参考视频（基于点云渲染）")
    parser.add_argument("--dataset-dir", type=str,
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+4",
                       help="数据集根目录")
    parser.add_argument("--robocasa-base", type=str,
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp",
                       help="RoboCasa数据集基础目录")
    parser.add_argument("--task-name", type=str, default="PnPCounterToSink",
                       help="指定任务名称（如果为None则处理所有任务）")
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32,
                       help="渲染的batch大小")
    parser.add_argument("--point-size", type=float, default=4.0,
                       help="点云渲染的点大小")
    parser.add_argument("--max-workers", type=int, default=1,
                       help="并行处理的最大线程数")
    parser.add_argument("--device", type=str, default='cuda',
                       help="渲染设备 (cuda/cpu)")

    args = parser.parse_args()

    # Find all tasks
    if args.task_name:
        tasks = [args.task_name]
    else:
        dataset_path = Path(args.dataset_dir)
        tasks = [d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"开始处理 {len(tasks)} 个任务")

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"处理任务: {task}")
        print(f"{'='*60}")

        # Find annotation files for this task
        task_annotation_dir = os.path.join(args.dataset_dir, task, "annotation")
        if not os.path.exists(task_annotation_dir):
            print(f"警告: 未找到任务 {task} 的annotation目录，跳过")
            continue

        episodes_to_process = []
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(task_annotation_dir, split)
            if not os.path.exists(split_dir):
                continue

            for json_file in os.listdir(split_dir):
                if json_file.endswith(".json"):
                    episode_id = int(json_file.replace(".json", ""))
                    annotation_path = os.path.join(split_dir, json_file)
                    episodes_to_process.append((task, episode_id, split, annotation_path))

        print(f"找到 {len(episodes_to_process)} 个episodes需要处理")

        # Process episodes
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    process_episode_ref_videos_pointcloud,
                    task, episode_id, split, None, annotation_path,
                    args.dataset_dir, args.camera_width, args.camera_height,
                    args.batch_size, args.point_size, args.device
                ) for task, episode_id, split, annotation_path in episodes_to_process
            ]

            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=len(episodes_to_process),
                             desc=f"处理{task}"):
                result = future.result()
                if result:
                    # 使用 tqdm.write 来避免打断进度条
                    tqdm.write(result)

    print("\n所有任务处理完成！")


if __name__ == "__main__":
    main()