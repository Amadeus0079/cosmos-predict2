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
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import json
import os
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import imageio
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision import utils as vutils
from tqdm import tqdm
import open3d as o3d
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from cosmos_predict2.utils.render_wrapper import BatchCameraRenderer
from cosmos_predict2.data.action_conditioned.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
    rotvec2rotm,
    rotm2rotvec,
    merge_pointclouds,
    merge_pointclouds_batch,
    pointclouds_world2cam,
    campose_to_pytorch3d_mat,
    campose_to_pytorch3d_mat_batch,
    pytorch3d_pointcloud_to_open3d
)



class AnyViewDataset(Dataset):
    def __init__(
        self,
        train_annotation_path,
        val_annotation_path,
        test_annotation_path,
        video_path,
        sequence_interval,
        num_frames,
        cam_ids,
        gt_cams,
        pred_cams,
        accumulate_action,
        video_size,
        val_start_frame_interval,
        debug=False,
        normalize=False,
        pre_encode=False,
        do_evaluate=False,
        load_t5_embeddings=False,
        load_action=True,
        load_state=True,
        mode="train",
    ):
        """Dataset class for loading 3D robot action-conditioned data.

        This dataset loads robot trajectories consisting of RGB video frames, robot states (arm positions and gripper states),
        and computes relative actions between consecutive frames.

        Args:
            train_annotation_path (str): Path to training annotation files
            val_annotation_path (str): Path to validation annotation files
            test_annotation_path (str): Path to test annotation files
            video_path (str): Base path to video files
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            cam_ids (list): List of camera IDs to sample from
            gt_cams (list): Known Cameras
            pred_cams (list): Cameras need to be predicted
            accumulate_action (bool): Whether to accumulate actions relative to first frame
            video_size (list): Target size [H,W] for video frames
            val_start_frame_interval (int): Frame sampling interval for validation/test
            debug (bool, optional): If True, only loads subset of data. Defaults to False.
            normalize (bool, optional): Whether to normalize video frames. Defaults to False.
            pre_encode (bool, optional): Whether to pre-encode video frames. Defaults to False.
            do_evaluate (bool, optional): Whether in evaluation mode. Defaults to False.
            load_t5_embeddings (bool, optional): Whether to load T5 embeddings. Defaults to False.
            load_action (bool, optional): Whether to load actions. Defaults to True.
            mode (str, optional): Dataset mode - 'train', 'val' or 'test'. Defaults to 'train'.

        The dataset loads robot trajectories and computes:
        - RGB video frames from specified camera views
        - Robot arm states (xyz position + euler angles)
        - Gripper states (binary open/closed)
        - Relative actions between consecutive frames

        Actions are computed as relative transforms between frames:
        - Translation: xyz offset in previous frame's coordinate frame
        - Rotation: euler angles of relative rotation
        - Gripper: binary gripper state

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - action: Action tensor [T-1,7]
            - video_name: Dict with episode/frame metadata
            - latent: Pre-encoded video features if pre_encode=True
        """

        super().__init__()
        if mode == "train":
            self.data_path = train_annotation_path
            self.start_frame_interval = 1
        elif mode == "val":
            self.data_path = val_annotation_path
            self.start_frame_interval = val_start_frame_interval
        elif mode == "test":
            self.data_path = test_annotation_path
            self.start_frame_interval = val_start_frame_interval
        self.video_path = video_path
        self.sequence_interval = sequence_interval
        self.mode = mode
        self.sequence_length = num_frames
        self.normalize = normalize
        self.pre_encode = pre_encode
        self.load_t5_embeddings = load_t5_embeddings
        self.load_action = load_action
        self.load_state = load_state

        self.cam_ids = cam_ids
        self.gt_cams = gt_cams
        self.pred_cams = pred_cams
        self.accumulate_action = accumulate_action

        self.action_dim = 7  # ee xyz (3) + ee rotvec (3) + gripper(1)
        self.c_act_scaler = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0]
        self.c_act_scaler = torch.tensor(self.c_act_scaler, dtype=float)
        self.ann_files = self._init_anns(self.data_path)

        print(f"{len(self.ann_files)} trajectories in total")
        self.samples = self._init_sequences(self.ann_files)

        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))
        if debug and not do_evaluate:
            self.samples = self.samples[0:10]
        print(f"{len(self.ann_files)} trajectories in total")
        print(f"{len(self.samples)} samples in total")
        # with open('./samples_16.pkl','wb') as file:
        #     pickle.dump(self.samples,file)
        self.wrong_number = 0
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.training = False
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(video_size)),  # 288 512
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])
        self.depth_preprocess = Resize_Preprocess(tuple(video_size))

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
        return ann_files

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {
                executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files
            }
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):
        samples = []
        with open(ann_file, "r") as f:
            ann = json.load(f)

        n_frames = len(ann["state"])
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample["ann_file"] = ann_file
            sample["frame_ids"] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval
            # make sure there are sequence_length number of frames
            if len(sample["frame_ids"]) == self.sequence_length:
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, frame_ids):
        from decord import VideoReader, cpu  # Importing here due to malloc errors on ARM when importing on top level

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data

    def _get_frames(self, label, frame_ids, cam_id, pre_encode):
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
        else:
            video_path = label["videos"][cam_id]["video_path"]
            video_path = os.path.join(self.video_path, video_path)
            frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)

            def printvideo(videos, filename):
                t_videos = rearrange(videos, "f c h w -> f h w c")
                t_videos = (
                    ((t_videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                print(t_videos.shape)
                writer = imageio.get_writer(filename, fps=4)  # fps 是帧率
                for frame in t_videos:
                    writer.append_data(frame)  # 1 4 13 23 # fp16 24 76 456 688

            if self.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode)
        return frames, temp_cam_id
    
    def _get_depth(self, label, frame_ids, cam_id):
        depth_path = label["videos"][cam_id]["depth_path"]
        depth_path = os.path.join(self.video_path, depth_path)
        all_depths = np.load(depth_path)
        depths = all_depths[frame_ids]
        depths = torch.tensor(depths)
    
        # depths = torch.flip(depths, dims=[-3])
        depths = depths.permute(0, 3, 1, 2)
        depths = self.depth_preprocess(depths)  # [T, C, H, W]
        return depths

    def _get_robot_states(self, label, frame_ids):
        all_states = np.array(label["state"])
        # all_cont_gripper_states = np.array(label["continuous_gripper_state"])
        all_cont_gripper_states = np.array(label["state"])[:, 6]
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        assert arm_states.shape[0] == self.sequence_length
        assert cont_gripper_states.shape[0] == self.sequence_length
        return arm_states, cont_gripper_states

    def _get_all_robot_states(self, label, frame_ids):
        all_states = np.array(label["state"])
        # all_cont_gripper_states = np.array(label["continuous_gripper_state"])
        all_cont_gripper_states = np.array(label["state"])[:, 6]
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        return arm_states, cont_gripper_states

    def _get_all_actions(self, arm_states, gripper_states, accumulate_action):
        action_num = arm_states.shape[0] - 1
        action = np.zeros((action_num, self.action_dim))
        if accumulate_action:
            first_xyz = arm_states[0, 0:3]
            first_rpy = arm_states[0, 3:6]
            first_rotm = rotvec2rotm(first_rpy)
            for k in range(1, action_num + 1):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = rotvec2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2rotvec(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        else:
            for k in range(1, action_num + 1):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = rotvec2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = rotvec2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2rotvec(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = np.zeros((self.sequence_length - 1, self.action_dim))
        if accumulate_action:
            first_xyz = arm_states[0, 0:3]
            first_rpy = arm_states[0, 3:6]
            first_rotm = rotvec2rotm(first_rpy)
            for k in range(1, self.sequence_length):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = rotvec2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2rotvec(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        else:
            for k in range(1, self.sequence_length):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = rotvec2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = rotvec2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2rotvec(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)
    
    def _get_robot_actions(self, label, frame_ids):
        all_actions = np.array(label["action"])
        actions = all_actions[frame_ids]
        arm_actions = actions[:, :7]
        assert arm_actions.shape[0] == self.sequence_length - 1
        return torch.from_numpy(arm_actions)

    def _get_all_robot_actions(self, label, frame_ids):
        all_actions = np.array(label["action"])
        actions = all_actions[frame_ids]
        arm_actions = actions[:, :7]
        return torch.from_numpy(arm_actions)
    
    def _get_cam_parameters(self, label, cam_id, frame_ids):
        all_extrinsic_matrixs = label["extrinsic_matrix"][cam_id]
        all_extrinsic_matrixs = torch.tensor(all_extrinsic_matrixs)
        extrinsic_matrixs = all_extrinsic_matrixs[frame_ids]
        
        all_intrinsic_matrixs = label["intrinsic_matrix"][cam_id]
        all_intrinsic_matrixs = torch.tensor(all_intrinsic_matrixs)
        intrinsic_matrixs = all_intrinsic_matrixs[frame_ids]
        
        return extrinsic_matrixs, intrinsic_matrixs  # (T, 4, 4)
    
    def _rgbd_to_pointcloud_batch(self, rgb_batch, depth_batch, intrinsics_batch, extrinsics_batch, depth_scale=1):
        """
        批量构建点云（世界坐标系）
        inputs:
            rgb_batch: torch.Tensor [B, C, H, W]  # 批量RGB图像
            depth_batch: torch.Tensor [B, 1, H, W]  # 批量深度图
            intrinsics_batch: torch.Tensor [B, 3, 3]  # 批量内参
            extrinsics_batch: torch.Tensor [B, 4, 4]  # 批量外参（相机到世界）
        return:
            pointclouds: Pointclouds对象，包含B个点云（带batch维度）
        """
        device = rgb_batch.device
        rgb_batch = rgb_batch / 255.0
        B, C, H, W = rgb_batch.shape

        # 生成像素坐标网格（批量处理）
        u = torch.linspace(0, W-1, W, device=device)
        v = torch.linspace(0, H-1, H, device=device)
        u, v = torch.meshgrid(u, v, indexing='xy')  # (H, W)
        u = u.flatten()  # (H*W,)
        v = v.flatten()  # (H*W,)
        # 扩展到batch维度：(B, H*W)
        u_batch = u.unsqueeze(0).repeat(B, 1)
        v_batch = v.unsqueeze(0).repeat(B, 1)

        # 深度值处理（批量）
        z_batch = depth_batch.squeeze(1).reshape(B, -1) / depth_scale  # (B, H*W)
        valid_mask = (z_batch > 0) & (z_batch < 10)  # (B, H*W)

        # 提取内参（批量）
        fx_batch = intrinsics_batch[:, 0, 0]  # (B,)
        fy_batch = intrinsics_batch[:, 1, 1]  # (B,)
        cx_batch = intrinsics_batch[:, 0, 2]  # (B,)
        cy_batch = intrinsics_batch[:, 1, 2]  # (B,)

        # 转换为相机坐标系3D点（批量）
        x = (u_batch - cx_batch.unsqueeze(1)) * z_batch / fx_batch.unsqueeze(1)  # (B, H*W)
        y = (v_batch - cy_batch.unsqueeze(1)) * z_batch / fy_batch.unsqueeze(1)  # (B, H*W)
        points_cam = torch.stack([x, y, z_batch], dim=2)  # (B, H*W, 3)

        # 过滤无效点（批量）
        valid_points_cam = []
        valid_colors = []
        for b in range(B):
            mask = valid_mask[b]
            # 相机坐标系点
            valid_points = points_cam[b, mask]  # (N_b, 3)
            valid_points_cam.append(valid_points)
            # 对应颜色
            rgb = rgb_batch[b].permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
            valid_colors.append(rgb[mask])  # (N_b, 3)

        # 转换到世界坐标系（批量外参应用）
        world_points_list = []
        for b in range(B):
            points_hom = torch.hstack([
                valid_points_cam[b],
                torch.ones(valid_points_cam[b].shape[0], 1, device=device)
            ])  # (N_b, 4)
            world_points = (extrinsics_batch[b] @ points_hom.T).T[:, :3]  # (N_b, 3)
            world_points_list.append(world_points)

        # 构建带batch的Pointclouds对象
        pointclouds = Pointclouds(
            points=world_points_list,  # 列表中每个元素为一个点云 (N_b, 3)
            features=valid_colors      # 对应颜色 (N_b, 3)
        )
        return pointclouds
    
    def _rgbd_to_pointcloud(self, rgb, depth, intrinsics, extrinsics, depth_scale=1):
        """
        rgb: torch.Tensor [C, H, W]
        depth: torch.Tensor [1, H, W]
        intrinsics: torch.Tensor [3, 3]
        extrinsics: torch.Tensor [4, 4]
        """
        # 读取图像
        rgb = rgb / 255.0  # 转换为[0, 1]范围内的浮点数
        device = rgb.device
        
        # 获取图像尺寸
        _, H, W = rgb.shape
        
        # 生成像素坐标网格
        u = torch.linspace(0, W-1, W)
        v = torch.linspace(0, H-1, H)
        u, v = torch.meshgrid(u, v, indexing='xy')  # (H, W)
        u = u.flatten().to(device)  # 展平为一维
        v = v.flatten().to(device)
        
        # 深度值转换为米
        z = depth.squeeze().flatten() / depth_scale  # (H*W,)
        
        # 过滤掉无效深度值
        valid_mask = z > 0
        u = u[valid_mask]
        v = v[valid_mask]
        z = z[valid_mask]
        
        # 相机内参
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # 转换为3D坐标 (x, y, z)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # 构建点云坐标并根据外参切换到世界坐标系
        points = torch.stack([x, y, z], dim=1)  # (N, 3)
        points_homogeneous = torch.hstack((points, torch.ones((points.shape[0], 1)).to(device)))  # (N, 4)
        world_points = (extrinsics @ points_homogeneous.T).T[:, :3]
        
        # 获取对应点的颜色
        rgb = rgb.permute(1, 2, 0)
        rgb = rgb.reshape(-1, 3)    # (H*W, 3)
        colors = rgb[valid_mask]    # (N, 3)
        
        # 创建PyTorch3D点云对象
        pointclouds = Pointclouds(points=[world_points], features=[colors])
        # pointclouds = Pointclouds(points=[points], features=[colors])
        
        return pointclouds
    
    def _render_pointcloud(self, pointclouds, extrinsic_matrix, intrinsic_matrix, width=256, height=256, 
                     point_size=1.0, background_color=(0, 0, 0), return_depth=False,
                     device=None, znear=0.01, points_per_pixel=10):
        """
        使用PyTorch3D渲染点云并可视化结果
        
        参数:
            pointclouds (pytorch3d.structures.Pointclouds): 点云数据
            extrinsic_matrix (torch.Tensor): 外参矩阵，形状为[4, 4]
            intrinsic_matrix (torch.Tensor): 内参矩阵，形状为[3, 3]
            width (int): 渲染图像宽度，默认为256
            height (int): 渲染图像高度，默认为256
            point_size (float): 点的大小，默认为1.0
            background_color (tuple): 背景颜色，默认为黑色(0, 0, 0)
            return_depth (bool): 是否返回深度图，默认为False
        
        返回:
            dict: 包含渲染结果的字典，键包括'image'(RGB图像)和'depth'(深度图，可选)
        """
        # 确保输入数据是PyTorch张量并在GPU上
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        rgb = images[0].mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        
        # 获取深度信息（如果需要）
        if return_depth:
            fragments = rasterizer(pointclouds)
            depth = fragments.zbuf[0, ..., 0].unsqueeze(2)
            return rgb, depth
        
        # 准备返回结果
        return rgb

    def _render_pointcloud_batch(self, pointclouds, extrinsic_matrixs, intrinsic_matrixs, 
                            point_size=1.0, height=128, width=128, background_color=(1, 1, 1), return_depth=False, device=None):
        """
        批量渲染点云
        inputs:
            pointclouds: Pointclouds对象（含B个点云）
            extrinsic_matrixs: torch.Tensor [B, 4, 4]  # 批量外参（世界到相机）
            intrinsic_matrixs: torch.Tensor [B, 3, 3]  # 批量内参
        return:
            render_images: torch.Tensor [B, C, H, W]  # 批量渲染图像
            render_depths: torch.Tensor [B, 1, H, W]  # 批量深度图（可选）
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B = extrinsic_matrixs.shape[0]

        # 提取相机参数（批量）
        fx = intrinsic_matrixs[:, 0, 0]  # (B,)
        fy = intrinsic_matrixs[:, 1, 1]  # (B,)
        cx = intrinsic_matrixs[:, 0, 2]  # (B,)
        cy = intrinsic_matrixs[:, 1, 2]  # (B,)

        # 计算视野角（批量）
        fov_y = 2 * torch.atan2(cy, fy) * 180 / torch.pi  # (B,)

        # 提取旋转矩阵和平移向量（批量）
        R = extrinsic_matrixs[:, :3, :3]  # (B, 3, 3)
        T = extrinsic_matrixs[:, :3, 3]   # (B, 3)

        # 创建批量相机
        cameras = FoVPerspectiveCameras(
            device=device,
            R=R,  # 直接使用批量旋转矩阵
            T=T,  # 直接使用批量平移向量
            fov=fov_y,
            znear=0.01
        )

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
        render_images = render_images.permute(0, 3, 1, 2)  # (B, C, H, W)

        # 批量提取深度（如果需要）
        render_depths = None
        if return_depth:
            fragments = rasterizer(pointclouds)  # 批量获取光栅化结果
            render_depths = fragments.zbuf[..., 0].unsqueeze(1)  # (B, 1, H, W)

        return render_images, render_depths
    
    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]
            with open(ann_file, "r") as f:
                label = json.load(f)
            arm_states, gripper_states = self._get_robot_states(label, frame_ids)
            # actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
            actions = self._get_robot_actions(label, frame_ids[:-1]) * 0  ## !!!!!!!!!!!!!!!!!!!!!!! remove action !!!!!!!!!!!!!!!!!!!!!!!
            # actions *= self.c_act_scaler

            data = dict()
            if self.load_state:
                data["state"] = np.concatenate([arm_states, gripper_states[..., None]], axis=1)
                
            if self.load_action:
                data["action"] = actions.float()
            
            pointclouds_dict = {}
            # data["video"] = dict()
            # data["depth"] = dict()
            # data["extrinsic_matrix"] = dict()
            # data["intrinsic_matrix"] = dict()
            renderer = BatchCameraRenderer()

            for cam_id in self.cam_ids:
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3).cuda()  # Rearrange from [T, C, H, W] to [C, T, H, W]
                
                depth = self._get_depth(label, frame_ids, cam_id)  # [T, C, H, W]
                depth = depth.permute(1, 0, 2, 3).cuda()  # [1, T, H, W]
                
                extrinsic_matrixs, intrinsic_matrixs = self._get_cam_parameters(label, cam_id, frame_ids)
                extrinsic_matrixs = extrinsic_matrixs.cuda()  # [T, 4, 4]
                intrinsic_matrixs = intrinsic_matrixs.cuda()  # [T, 4, 4]
                
                # Build 3D Scene from GT cams
                if cam_id in self.gt_cams:
                    video_T_C_H_W = video.permute(1, 0, 2, 3)
                    depth_T_C_H_W = depth.permute(1, 0, 2, 3)
                    pointclouds_T = self._rgbd_to_pointcloud_batch(video_T_C_H_W, depth_T_C_H_W, intrinsic_matrixs, extrinsic_matrixs).cuda()
                    pointclouds_dict[cam_id] = pointclouds_T
                    
                if cam_id in self.pred_cams:
                    data["video"] = video.to(dtype=torch.uint8)
                    data["depth"] = depth.to(dtype=torch.float32)
                    data["extrinsic_matrix"] = extrinsic_matrixs
                    data["intrinsic_matrix"] = intrinsic_matrixs
            
            height = video.shape[2]
            width = video.shape[3]
            
            # Merge gt cams' pointclouds
            gt_pointclouds = []
            for gt_cam in self.gt_cams:
                gt_pointclouds.append(pointclouds_dict[gt_cam])
            merged_pointclouds = merge_pointclouds_batch(gt_pointclouds).cuda()
            # o3d_pc = pytorch3d_pointcloud_to_open3d(merged_pointclouds)
            # o3d.io.write_point_cloud("output/aaa.ply", o3d_pc)
            
            # Get pred cams' render images
            # data["pred_video"] = dict()
            for pred_cam in self.pred_cams:
                extrinsic_matrixs = data["extrinsic_matrix"]
                intrinsic_matrixs = data["intrinsic_matrix"]
                # extrinsic_matrixs[-1] = extrinsic_matrixs[-2]
                extrinsic_matrixs = campose_to_pytorch3d_mat_batch(extrinsic_matrixs)

                render_image, render_depth = self._render_pointcloud_batch(merged_pointclouds, 
                                                        extrinsic_matrixs=extrinsic_matrixs,
                                                        intrinsic_matrixs=intrinsic_matrixs,
                                                        point_size=4,
                                                        height=height,
                                                        width=width,
                                                        background_color=(1, 1, 1),
                                                        return_depth=True)  # [T, H, W, C]
                render_image = render_image.permute(1, 0, 2, 3)  # [C, T, H, W]
                render_depth = render_depth.permute(1, 0, 2, 3)  # [C, T, H, W]
                
                pred_video = render_image
                pred_depth = render_depth
                data["pred_video"] = pred_video
                data["pred_depth"] = pred_depth
            
            data["annotation_file"] = ann_file

            # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
            if "episode_id" in label:
                data["__key__"] = label["episode_id"]
            else:
                data["__key__"] = label["original_path"]

            # Just add these to fit the interface
            if self.load_t5_embeddings:
                t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
                data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings).cuda()
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16).cuda()
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4).cuda()  # TODO: Does this matter?
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, height, width).cuda()
            data["num_conditional_frames"] = self.sequence_length
            data["base_pos"] = np.array(label["base_pos"])
            data["base_quat"] = np.array(label["base_quat"])

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]
        
        
if __name__ == "__main__":
    base_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa/PnPCounterToSink_10/"
    train_annotation_path = os.path.join(base_path, "annotation/train")
    val_annotation_path = os.path.join(base_path, "annotation/val")
    test_annotation_path = os.path.join(base_path, "annotation/test")

    train_dataset = AnyViewDataset(
        train_annotation_path=train_annotation_path,
        val_annotation_path=val_annotation_path,
        test_annotation_path=test_annotation_path,
        video_path=base_path,
        sequence_interval=1,
        num_frames=17,
        cam_ids=['robot0_agentview_center', 'robot0_eye_in_hand', 'robot0_activeview'],
        gt_cams=['robot0_agentview_center', 'robot0_eye_in_hand'],
        pred_cams=['robot0_activeview'],
        accumulate_action=False,
        video_size=[128, 128],
        val_start_frame_interval=1,
        mode="train",
    )
    
    data = train_dataset[40]
    gt_video = data["video"].permute(1, 2, 3, 0)
    pred_video = data["pred_video"].permute(1, 2, 3, 0)
    gt_video = gt_video.detach().cpu().numpy()
    pred_video = pred_video.detach().cpu().numpy()
    import mediapy
    mediapy.write_video("output/pppp.mp4", pred_video, fps=2)
    mediapy.write_video("output/gggg.mp4", gt_video, fps=2)