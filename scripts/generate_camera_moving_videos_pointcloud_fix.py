#!/usr/bin/env python3
"""
生成静态场景的参考视频（基于点云渲染）
【优化版】：修复了 lazy wrapper 错误，优化了 Pointclouds 构建速度
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
import torch
import time
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

# --- Timer 类保持不变 ---
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

# --- 辅助函数保持不变 (rgbd_to_pointcloud_batch, merge_pointclouds_batch, load_...) ---
# (为了节省篇幅，这里省略未修改的函数，请保留你原有的 rgbd_to_pointcloud_batch, load_video_frames 等函数)

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

# 必须保留原始 import 中定义的函数，此处仅展示修改过的关键函数

def rgbd_to_pointcloud_batch(rgb_batch, depth_batch, intrinsics_batch, extrinsics_batch, depth_scale=1, device='cuda'):
    # ... (保持原代码不变) ...
    rgb_batch = rgb_batch.to(device)
    depth_batch = depth_batch.to(device)
    intrinsics_batch = intrinsics_batch.to(device)
    extrinsics_batch = extrinsics_batch.to(device)
    rgb_batch = rgb_batch / 255.0
    B, C, H, W = rgb_batch.shape
    u = torch.linspace(0, W-1, W, device=device)
    v = torch.linspace(0, H-1, H, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    u = u.flatten()
    v = v.flatten()
    u_batch = u.unsqueeze(0).repeat(B, 1)
    v_batch = v.unsqueeze(0).repeat(B, 1)
    z_batch = depth_batch.squeeze(1).reshape(B, -1) / depth_scale
    valid_mask = (z_batch > 0) & (z_batch < 10)
    fx_batch = intrinsics_batch[:, 0, 0]
    fy_batch = intrinsics_batch[:, 1, 1]
    cx_batch = intrinsics_batch[:, 0, 2]
    cy_batch = intrinsics_batch[:, 1, 2]
    x = (u_batch - cx_batch.unsqueeze(1)) * z_batch / fx_batch.unsqueeze(1)
    y = (v_batch - cy_batch.unsqueeze(1)) * z_batch / fy_batch.unsqueeze(1)
    points_cam = torch.stack([x, y, z_batch], dim=2)
    valid_points_cam = []
    valid_colors = []
    for b in range(B):
        mask = valid_mask[b]
        valid_points_cam.append(points_cam[b, mask])
        rgb = rgb_batch[b].permute(1, 2, 0).reshape(-1, 3)
        valid_colors.append(rgb[mask])
    world_points_list = []
    for b in range(B):
        points_hom = torch.hstack([valid_points_cam[b], torch.ones(valid_points_cam[b].shape[0], 1, device=device)])
        world_points = (extrinsics_batch[b] @ points_hom.T).T[:, :3]
        world_points_list.append(world_points)
    pointclouds = Pointclouds(points=world_points_list, features=valid_colors)
    return pointclouds

def merge_pointclouds_batch(pointclouds_list):
    # ... (保持原代码不变) ...
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

# ==========================================
# 【关键修复 1】：使用 CPU 避免 lazy wrapper 错误
# ==========================================
def campose_to_pytorch3d_mat_batch(camera_pose):
    correction = torch.tensor([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=camera_pose.dtype, device=camera_pose.device)

    # 移至 CPU 计算 inverse，绝对安全
    pose_cpu = camera_pose.detach().cpu()
    world_to_cam_cpu = torch.inverse(pose_cpu)
    world_to_cam_open3d = world_to_cam_cpu.to(camera_pose.device)

    pytorch3d_mat = correction @ world_to_cam_open3d
    pytorch3d_mat[:, :3, :3] = pytorch3d_mat[:, :3, :3].transpose(1, 2).clone()
    return pytorch3d_mat

# ==========================================
# 【关键优化 2】：避免在循环中重复创建 Pointclouds
# ==========================================
def render_pointcloud_batch_optimized(renderer, pointcloud_batch, extrinsic_matrixs):
    """
    使用预构建的 renderer 和已扩展好的 pointcloud_batch
    """
    # 更新相机的外参 (R, T)
    # 注意：FoVPerspectiveCameras 是不可变的，通常需要重新创建 cameras 或者 update
    # 但最高效的方式是每次只创建 Cameras 对象，因为它是轻量级的，
    # 而 Renderer 和 Rasterizer 可以复用（如果 settings 不变）
    
    # 获取 renderer 中的某些设置（这里为了简单，我们还是在外部创建 cameras，但复用 renderer）
    # 实际上 PyTorch3D 的设计是 renderer(pointclouds, cameras=...)
    pass # 逻辑合并到主循环中展示

def process_episode_ref_videos_pointcloud(
    task_name, episode_id, split, hdf5_path, annotation_path,
    output_dir, camera_width=256, camera_height=256,
    batch_size=16, point_size=4.0, device='cuda'
):
    episode_start_time = time.time()
    timings = {}
    
    try:
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        gt_cams = ['robot0_agentview_center', 'robot0_eye_in_hand']
        randomview_cameras = [cam_name for cam_name in annotation['videos'].keys()
                             if cam_name.startswith('robot0_randomview_')]
        all_cams = gt_cams + randomview_cameras
        
        if not randomview_cameras:
            return None

        first_cam = list(annotation['videos'].keys())[0]
        num_frames = len(annotation['extrinsic_matrix'][first_cam])
        
        task_output_dir = os.path.join(output_dir, task_name)
        video_output_dir = os.path.join(output_dir, task_name, "videos", split, str(episode_id))
        os.makedirs(video_output_dir, exist_ok=True)

        generated_count = 0
        skipped_count = 0
        total_load_time = 0; total_pc_build_time = 0; total_render_time = 0; total_save_time = 0

        # --- 初始化渲染器设置 (只需做一次) ---
        # 假设所有相机内参在同一视频中变化不大，或者我们取第一帧的内参来初始化 FOV
        # 严谨起见，FOV 会随焦距变化，所以我们在循环内更新 Camera
        raster_settings = PointsRasterizationSettings(
            image_size=(camera_height, camera_width),
            radius=point_size / max(camera_width, camera_height),
            points_per_pixel=10,
            bin_size=0 # 0 for naive rasterization (faster for simple scenes)
        )
        rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings) # Cameras bind later
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(1, 1, 1))
        )

        for cam_name in randomview_cameras:
            ref_video_path = os.path.join(video_output_dir, f"{cam_name}_ref.mp4")
            ref_depth_path = os.path.join(video_output_dir, f"{cam_name}_ref.npy")

            if os.path.exists(ref_video_path) and os.path.exists(ref_depth_path):
                skipped_count += 1
                continue

            # --- Stage 1 & 2: Load Data and Build PC ---
            with SectionTimer("Load & Build PC") as t:
                pointclouds_list = []
                for view_cam in all_cams:
                    if view_cam not in annotation['videos']: continue
                    
                    # 构建完整路径
                    video_path = os.path.join(task_output_dir, annotation['videos'][view_cam]['video_path'])
                    depth_path = os.path.join(task_output_dir, annotation['videos'][view_cam]['depth_path'])

                    frame = load_video_frames(video_path, [0])[0]
                    depth = load_depth_frames(depth_path, [0])[0]

                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)
                    depth_tensor = torch.from_numpy(depth).permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    extrinsic = torch.tensor(annotation['extrinsic_matrix'][view_cam][0]).unsqueeze(0).to(device)
                    intrinsic = torch.tensor(annotation['intrinsic_matrix'][view_cam][0]).unsqueeze(0).to(device)

                    pc = rgbd_to_pointcloud_batch(frame_tensor, depth_tensor, intrinsic, extrinsic, device=device)
                    pointclouds_list.append(pc)
                
                merged_pointcloud = merge_pointclouds_batch(pointclouds_list)
            total_load_time += t.duration

            # Prepare rendering data
            extrinsic_matrices = torch.tensor(annotation['extrinsic_matrix'][cam_name]).to(device)
            intrinsic_matrices = torch.tensor(annotation['intrinsic_matrix'][cam_name]).to(device)
            # 使用修复后的 CPU 版本函数
            extrinsic_matrices_render = campose_to_pytorch3d_mat_batch(extrinsic_matrices)

            # --- Stage 3: Rendering (Optimized) ---
            with SectionTimer("Rendering") as t:
                all_render_images = []
                all_render_depths = []
                
                # 预先扩展点云，避免在循环中重复 copy
                # 如果 batch_size 很大，这可能会占内存，但 300k 点 * 32 还在可控范围
                # 使用 extend 扩展成一个 batch 的点云结构
                batch_pc_template = merged_pointcloud.extend(batch_size) 

                for batch_start in range(0, num_frames, batch_size):
                    batch_end = min(batch_start + batch_size, num_frames)
                    current_batch_size = batch_end - batch_start
                    
                    batch_extrinsics = extrinsic_matrices_render[batch_start:batch_end]
                    batch_intrinsics = intrinsic_matrices[batch_start:batch_end]

                    # 更新 Cameras
                    fx = batch_intrinsics[:, 0, 0]
                    fy = batch_intrinsics[:, 1, 1]
                    # 计算 FOV
                    fov_y = 2 * torch.atan2(torch.tensor(camera_height / 2.0, device=device), fy) * 180 / torch.pi
                    
                    cameras = FoVPerspectiveCameras(
                        device=device,
                        R=batch_extrinsics[:, :3, :3],
                        T=batch_extrinsics[:, :3, 3],
                        fov=fov_y,
                        znear=0.01
                    )
                    
                    # 只有当 batch size 不足（最后一组）时，才切片，否则使用预构建的 template
                    if current_batch_size == batch_size:
                        current_pc = batch_pc_template
                    else:
                        current_pc = merged_pointcloud.extend(current_batch_size)

                    # 渲染
                    # update cameras in rasterizer is not persistent, pass to renderer call if possible
                    # PyTorch3D Renderer forward accepts pointclouds, and uses rasterizer.cameras if set
                    # However, typical usage is creating rasterizer with cameras. 
                    # Here we can temporarily override cameras in the rasterizer call? 
                    # No, PointsRenderer(rasterizer=...) encapsulates the call.
                    # Best way: create a new rasterizer fragment or use functional call?
                    # The standard way is to create a NEW renderer or rasterizer with new cameras is cheap.
                    # ONLY creating the Pointclouds object is expensive.
                    
                    # 重新绑定 cameras 到 rasterizer (这是轻量级的)
                    renderer.rasterizer.cameras = cameras
                    
                    images = renderer(current_pc)
                    render_images = images.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
                    
                    # 获取深度
                    fragments = renderer.rasterizer(current_pc)
                    render_depths = fragments.zbuf[..., 0].unsqueeze(-1)

                    # 依然会有 cpu 同步，但在 batch 较大时可以接受
                    all_render_images.append(render_images.cpu())
                    all_render_depths.append(render_depths.cpu())

                all_render_images = torch.cat(all_render_images, dim=0)
                all_render_depths = torch.cat(all_render_depths, dim=0)
            
            total_render_time += t.duration

            # --- Stage 4: Saving ---
            with SectionTimer("Saving") as t:
                # Save video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(ref_video_path, fourcc, 20.0, (camera_width, camera_height))
                # 转换为 numpy 一次性操作
                video_data = all_render_images.numpy()
                for frame in video_data:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)
                out.release()
                np.save(ref_depth_path, all_render_depths.numpy())
            total_save_time += t.duration

            generated_count += 1

        total_elapsed = time.time() - episode_start_time
        if generated_count > 0:
            return (f"[Ep {episode_id}] Gen:{generated_count} | Total:{total_elapsed:.1f}s | "
                    f"Render:{total_render_time:.2f}s | Save:{total_save_time:.2f}s")
        return None

    except Exception as e:
        import traceback
        return f"Err Ep {episode_id}: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    # ... 其他参数 ...
    parser.add_argument("--dataset-dir", type=str, default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+4")
    parser.add_argument("--task-name", type=str, default="PnPCounterToSink")
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--point-size", type=float, default=4.0)
    
    # 【关键建议】：默认 max-workers 改为 1 或 2，不要用 8
    parser.add_argument("--max-workers", type=int, default=1, help="GPU任务建议设为1") 
    parser.add_argument("--device", type=str, default='cuda')
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