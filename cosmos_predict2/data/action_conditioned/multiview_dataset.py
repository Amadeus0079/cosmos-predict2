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

import imageio
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.io import save_ply
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

from cosmos_predict2.data.action_conditioned.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
)


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


class MultiViewDataset(Dataset):
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

        self.cam_ids = cam_ids
        self.accumulate_action = accumulate_action

        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)
        self.c_act_scaler = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0]
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
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
        return depths

    def _get_robot_states(self, label, frame_ids):
        all_states = np.array(label["state"])
        all_cont_gripper_states = np.array(label["continuous_gripper_state"])
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        assert arm_states.shape[0] == self.sequence_length
        assert cont_gripper_states.shape[0] == self.sequence_length
        return arm_states, cont_gripper_states

    def _get_all_robot_states(self, label, frame_ids):
        all_states = np.array(label["state"])
        all_cont_gripper_states = np.array(label["continuous_gripper_state"])
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
            first_rotm = euler2rotm(first_rpy)
            for k in range(1, action_num + 1):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        else:
            for k in range(1, action_num + 1):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = np.zeros((self.sequence_length - 1, self.action_dim))
        if accumulate_action:
            first_xyz = arm_states[0, 0:3]
            first_rpy = arm_states[0, 3:6]
            first_rotm = euler2rotm(first_rpy)
            for k in range(1, self.sequence_length):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        else:
            for k in range(1, self.sequence_length):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)
    
    def _get_cam_parameters(self, label, cam_id, frame_ids):
        all_extrinsic_matrixs = label["extrinsic_matrix"][cam_id]
        all_extrinsic_matrixs = torch.tensor(all_extrinsic_matrixs)
        extrinsic_matrixs = all_extrinsic_matrixs[frame_ids]
        
        all_intrinsic_matrixs = label["intrinsic_matrix"][cam_id]
        all_intrinsic_matrixs = torch.tensor(all_intrinsic_matrixs)
        intrinsic_matrixs = all_intrinsic_matrixs[frame_ids]
        
        return extrinsic_matrixs, intrinsic_matrixs  # (T, 4, 4)
    
    def _rgbd_to_pointcloud(self, rgb, depth, intrinsics, extrinsics, depth_scale=1):
        """
        从RGB图像和深度图生成点云
        
        参数:
            rgb_path: RGB图像路径
            depth_path: 深度图路径
            intrinsics: 相机内参矩阵 (3x3)
            extrinsics: 相机外参矩阵 (4x4)
            depth_scale: 深度图缩放因子，将像素值转换为实际深度(米)
            
        返回:
            pointcloud: PyTorch3D的Pointclouds对象
        """
        # 读取图像
        rgb = rgb / 255.0  # 转换为[0, 1]范围内的浮点数
        
        # 获取图像尺寸
        _, H, W = rgb.shape
        
        # 生成像素坐标网格
        u = torch.linspace(0, W-1, W)
        v = torch.linspace(0, H-1, H)
        u, v = torch.meshgrid(u, v, indexing='xy')  # (H, W)
        u = u.flatten()  # 展平为一维
        v = v.flatten()
        
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
        points_homogeneous = torch.hstack((points, torch.ones((points.shape[0], 1))))  # (N, 4)
        world_points = (extrinsics @ points_homogeneous.T).T[:, :3]
        
        # 获取对应点的颜色
        rgb = rgb.permute(1, 2, 0)
        rgb = rgb.reshape(-1, 3)    # (H*W, 3)
        colors = rgb[valid_mask]    # (N, 3)
        
        # 创建PyTorch3D点云对象
        pointclouds = Pointclouds(points=[world_points], features=[colors])
        
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
        
        return result

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
            actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
            actions *= self.c_act_scaler

            data = dict()
            if self.load_action:
                data["action"] = actions.float()

            for cam_id in self.cam_ids:
                data[cam_id] = dict()
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                data[cam_id]["video"] = video.to(dtype=torch.uint8)
                
                depth = self._get_depth(label, frame_ids, cam_id)
                depth = depth.permute(3, 0, 1, 2)  # [1, T, H, W]
                data[cam_id]["depth"] = depth.to(dtype=torch.float32)  # []
                
                extrinsic_matrixs, intrinsic_matrixs = self._get_cam_parameters(label, cam_id, frame_ids)
                data[cam_id]["extrinsic_matrix"] = extrinsic_matrixs
                data[cam_id]["intrinsic_matrix"] = intrinsic_matrixs
                
                # Build 3D Scene from GT cams Render Images by Pred Cams
                first_rgb = video[:, 0]
                first_depth = depth[:, 0]
                height = first_rgb.shape[1]
                width = first_rgb.shape[2]
                first_extrinsics = extrinsic_matrixs[0]
                first_instrinsics = intrinsic_matrixs[0]
                pointclouds = self._rgbd_to_pointcloud(first_rgb, first_depth, first_instrinsics, first_extrinsics)
                data[cam_id]["pointcloud"] = pointclouds
                save_pointcloud_to_ply(pointclouds, "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/output/aaa.ply")
                
                # Render Images by Pred Cams
                render_image = self._render_pointcloud(pointclouds,
                                        first_extrinsics,
                                        first_instrinsics,
                                        width,
                                        height,
                                        point_size=15,
                                        background_color=(1, 1, 1))
                
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
            data["padding_mask"] = torch.zeros(1, 256, 256).cuda()

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
    base_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_multiview/"
    train_annotation_path = os.path.join(base_path, "annotation/train")
    val_annotation_path = os.path.join(base_path, "annotation/val")
    test_annotation_path = os.path.join(base_path, "annotation/test")

    train_dataset = MultiViewDataset(
        train_annotation_path=train_annotation_path,
        val_annotation_path=val_annotation_path,
        test_annotation_path=test_annotation_path,
        video_path=base_path,
        sequence_interval=1,
        num_frames=13,
        cam_ids=['robot0_agentview_right', 'robot0_eye_in_hand', 'robot0_handview_right', 'robot0_handview_front'],
        gt_cams=['robot0_agentview_right', 'robot0_eye_in_hand'],
        pred_cams=['robot0_eye_in_hand', 'robot0_handview_right', 'robot0_handview_front'],
        accumulate_action=False,
        video_size=[256, 256],
        val_start_frame_interval=1,
        mode="train",
    )
    
    data = train_dataset[0]
