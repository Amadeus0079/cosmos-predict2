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
NovelViewDataset: Multi-view dataset that uses agentview_center and eye_in_hand as GT cameras,
and treats each of 4 random views as a separate prediction camera for training.

Each data sequence generates 4 samples, one for each randomview used as pred_cam.
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


class NovelViewDataset(Dataset):
    def __init__(
        self,
        train_annotation_path,
        val_annotation_path,
        test_annotation_path,
        video_path,
        sequence_interval,
        num_frames,
        cam_ids,
        gt_cams=None,
        pred_cams=None,
        randomview_names=None,
        accumulate_action=False,
        video_size=[128, 128],
        val_start_frame_interval=1,
        debug=False,
        normalize=False,
        pre_encode=False,
        do_evaluate=False,
        load_t5_embeddings=False,
        load_action=True,
        load_state=True,
        mode="train",
    ):
        """Dataset class for loading 3D robot action-conditioned data with multiple novel views.

        This dataset loads robot trajectories and generates 4 samples per sequence,
        one for each random view used as the prediction camera.

        Args:
            train_annotation_path (str): Path to training annotation files
            val_annotation_path (str): Path to validation annotation files
            test_annotation_path (str): Path to test annotation files
            video_path (str): Base path to video files
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            cam_ids (list): All available camera IDs
            gt_cams (list): Ground truth cameras for pointcloud reconstruction.
                           Defaults to ['robot0_agentview_center', 'robot0_eye_in_hand']
            pred_cams (list): Deprecated (use randomview_names instead)
            randomview_names (list): Names of the 4 random view cameras to use as pred_cams.
                                   Defaults to ['robot0_randomview_0', 'robot0_randomview_1',
                                              'robot0_randomview_2', 'robot0_randomview_3']
            accumulate_action (bool): Whether to accumulate actions relative to first frame
            video_size (list): Target size [H,W] for video frames
            val_start_frame_interval (int): Frame sampling interval for validation/test
            debug (bool, optional): If True, only loads subset of data
            normalize (bool, optional): Whether to normalize video frames
            pre_encode (bool, optional): Whether to pre-encode video frames
            do_evaluate (bool, optional): Whether in evaluation mode
            load_t5_embeddings (bool, optional): Whether to load T5 embeddings
            load_action (bool, optional): Whether to load actions
            load_state (bool, optional): Whether to load states
            mode (str, optional): Dataset mode - 'train', 'val' or 'test'

        The dataset generates 4x the number of original samples:
        - Each sequence yields 4 samples, one per randomview as pred_cam
        - All samples share the same GT pointcloud (from agentview_center + eye_in_hand)

        Returns dict with:
            - video: RGB frames from pred_cam (one of the random views) [C,T,H,W]
            - pred_video: Rendered frames from GT pointcloud at pred_cam view [C,T,H,W]
            - pred_depth: Rendered depth from GT pointcloud at pred_cam view [C,T,H,W]
            - extrinsic_matrix: Pred camera extrinsics [T,4,4]
            - intrinsic_matrix: Pred camera intrinsics [T,3,3]
            - action: Action tensor [T-1,7]
            - state: Robot state tensor [T,7]
            - randomview_id: Index of which randomview is used (0-3)
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

        # Set default cameras
        self.cam_ids = cam_ids
        if gt_cams is None:
            self.gt_cams = ['robot0_agentview_center', 'robot0_eye_in_hand']
        else:
            self.gt_cams = gt_cams

        if randomview_names is None:
            self.randomview_names = [
                'robot0_randomview_0',
                'robot0_randomview_1',
                'robot0_randomview_2',
                'robot0_randomview_3'
            ]
        else:
            self.randomview_names = randomview_names

        self.num_randomviews = len(self.randomview_names)

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
        print(f"{len(self.samples)} base sequences in total")
        print(f"{len(self) * self.num_randomviews} total samples (x{self.num_randomviews} for randomviews)")

        self.wrong_number = 0
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.training = False
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(video_size)),
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
        # Total length = number of sequences x number of randomviews
        return len(self.samples) * self.num_randomviews

    def _get_sample_index(self, index):
        """Convert global index to (sample_idx, randomview_id) tuple."""
        sample_idx = index // self.num_randomviews
        randomview_id = index % self.num_randomviews
        return sample_idx, randomview_id

    def _load_video(self, video_path, frame_ids):
        from decord import VideoReader, cpu

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

        depths = depths.permute(0, 3, 1, 2)
        depths = self.depth_preprocess(depths)  # [T, C, H, W]
        return depths

    def _get_robot_states(self, label, frame_ids):
        all_states = np.array(label["state"])
        all_cont_gripper_states = np.array(label["state"])[:, 6]
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        assert arm_states.shape[0] == self.sequence_length
        assert cont_gripper_states.shape[0] == self.sequence_length
        return arm_states, cont_gripper_states

    def _get_all_robot_states(self, label, frame_ids):
        all_states = np.array(label["state"])
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
        Batch construct pointclouds in world coordinates
        inputs:
            rgb_batch: torch.Tensor [B, C, H, W]
            depth_batch: torch.Tensor [B, 1, H, W]
            intrinsics_batch: torch.Tensor [B, 3, 3]
            extrinsics_batch: torch.Tensor [B, 4, 4]
        return:
            pointclouds: Pointclouds object with B pointclouds
        """
        device = rgb_batch.device
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

    def _rgbd_to_pointcloud(self, rgb, depth, intrinsics, extrinsics, depth_scale=1):
        """
        rgb: torch.Tensor [C, H, W]
        depth: torch.Tensor [1, H, W]
        intrinsics: torch.Tensor [3, 3]
        extrinsics: torch.Tensor [4, 4]
        """
        rgb = rgb / 255.0
        device = rgb.device

        _, H, W = rgb.shape

        u = torch.linspace(0, W-1, W)
        v = torch.linspace(0, H-1, H)
        u, v = torch.meshgrid(u, v, indexing='xy')
        u = u.flatten().to(device)
        v = v.flatten().to(device)

        z = depth.squeeze().flatten() / depth_scale

        valid_mask = z > 0
        u = u[valid_mask]
        v = v[valid_mask]
        z = z[valid_mask]

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = torch.stack([x, y, z], dim=1)
        points_homogeneous = torch.hstack((points, torch.ones((points.shape[0], 1)).to(device)))
        world_points = (extrinsics @ points_homogeneous.T).T[:, :3]

        rgb = rgb.permute(1, 2, 0).reshape(-1, 3)
        colors = rgb[valid_mask]

        pointclouds = Pointclouds(points=[world_points], features=[colors])

        return pointclouds

    def _render_pointcloud(self, pointclouds, extrinsic_matrix, intrinsic_matrix, width=256, height=256,
                     point_size=1.0, background_color=(0, 0, 0), return_depth=False,
                     device=None, znear=0.01, points_per_pixel=10):
        """Render pointcloud using PyTorch3D."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        fov_y = 2 * torch.atan2(
            torch.tensor(height / 2.0, dtype=torch.float32, device=device),
            fy
        ) * 180 / np.pi
        fov_x = 2 * torch.atan2(
            torch.tensor(width / 2.0, dtype=torch.float32, device=device),
            fx
        ) * 180 / np.pi

        R = extrinsic_matrix[:3, :3]
        T = extrinsic_matrix[:3, 3]

        cameras = FoVPerspectiveCameras(
            device=device,
            R=R.unsqueeze(0),
            T=T.unsqueeze(0),
            fov=fov_y.unsqueeze(0),
            znear=znear
        )

        raster_settings = PointsRasterizationSettings(
            image_size=(height, width),
            radius=point_size / max(width, height),
            points_per_pixel=points_per_pixel
        )

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=background_color)
        )

        images = renderer(pointclouds)
        rgb = images[0].mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

        if return_depth:
            fragments = rasterizer(pointclouds)
            depth = fragments.zbuf[0, ..., 0].unsqueeze(2)
            return rgb, depth

        return rgb

    def _render_pointcloud_batch(self, pointclouds, extrinsic_matrixs, intrinsic_matrixs,
                            point_size=1.0, height=128, width=128, background_color=(1, 1, 1), return_depth=False, device=None):
        """
        Batch render pointclouds
        inputs:
            pointclouds: Pointclouds object with B pointclouds
            extrinsic_matrixs: torch.Tensor [B, 4, 4]
            intrinsic_matrixs: torch.Tensor [B, 3, 3]
        return:
            render_images: torch.Tensor [B, C, H, W]
            render_depths: torch.Tensor [B, 1, H, W]
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B = extrinsic_matrixs.shape[0]

        fx = intrinsic_matrixs[:, 0, 0]
        fy = intrinsic_matrixs[:, 1, 1]
        cx = intrinsic_matrixs[:, 0, 2]
        cy = intrinsic_matrixs[:, 1, 2]

        fov_y = 2 * torch.atan2(cy, fy) * 180 / torch.pi

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
            points_per_pixel=10,
            bin_size=0
        )

        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=background_color)
        )

        images = renderer(pointclouds)  # (B, H, W, 3)
        render_images = images.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        render_images = render_images.permute(0, 3, 1, 2)  # (B, C, H, W)

        render_depths = None
        if return_depth:
            fragments = rasterizer(pointclouds)
            render_depths = fragments.zbuf[..., 0].unsqueeze(1)  # (B, 1, H, W)

        return render_images, render_depths

    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        try:
            # Convert global index to sample index and randomview id
            sample_idx, randomview_id = self._get_sample_index(index)

            sample = self.samples[sample_idx]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]

            with open(ann_file, "r") as f:
                label = json.load(f)

            # Get current randomview as pred_cam
            pred_cam = self.randomview_names[randomview_id]

            arm_states, gripper_states = self._get_robot_states(label, frame_ids)
            actions = self._get_robot_actions(label, frame_ids[:-1]) * 0

            data = dict()

            if self.load_state:
                data["state"] = np.concatenate([arm_states, gripper_states[..., None]], axis=1)

            if self.load_action:
                data["action"] = actions.float()

            pointclouds_dict = {}
            renderer = BatchCameraRenderer()

            # Load GT cameras and build pointclouds
            for gt_cam in self.gt_cams:
                video, _ = self._get_obs(label, frame_ids, gt_cam, pre_encode=False)
                video = video.permute(1, 0, 2, 3).cuda()  # [C, T, H, W]

                depth = self._get_depth(label, frame_ids, gt_cam)  # [T, C, H, W]
                depth = depth.permute(1, 0, 2, 3).cuda()  # [1, T, H, W]

                extrinsic_matrixs, intrinsic_matrixs = self._get_cam_parameters(label, gt_cam, frame_ids)
                extrinsic_matrixs = extrinsic_matrixs.cuda()  # [T, 4, 4]
                intrinsic_matrixs = intrinsic_matrixs.cuda()  # [T, 4, 4]

                # Build pointcloud from GT camera
                video_T_C_H_W = video.permute(1, 0, 2, 3)
                depth_T_C_H_W = depth.permute(1, 0, 2, 3)
                pointclouds_T = self._rgbd_to_pointcloud_batch(
                    video_T_C_H_W, depth_T_C_H_W,
                    intrinsic_matrixs, extrinsic_matrixs
                ).cuda()
                pointclouds_dict[gt_cam] = pointclouds_T

            # Merge all GT pointclouds
            gt_pointclouds = []
            for gt_cam in self.gt_cams:
                gt_pointclouds.append(pointclouds_dict[gt_cam])
            merged_pointclouds = merge_pointclouds_batch(gt_pointclouds).cuda()

            # Load pred camera (randomview) data
            video, _ = self._get_obs(label, frame_ids, pred_cam, pre_encode=False)
            video = video.permute(1, 0, 2, 3).cuda()  # [C, T, H, W]

            depth = self._get_depth(label, frame_ids, pred_cam)  # [T, C, H, W]
            depth = depth.permute(1, 0, 2, 3).cuda()  # [1, T, H, W]

            extrinsic_matrixs, intrinsic_matrixs = self._get_cam_parameters(label, pred_cam, frame_ids)
            extrinsic_matrixs = extrinsic_matrixs.cuda()  # [T, 4, 4]
            intrinsic_matrixs = intrinsic_matrixs.cuda()  # [T, 3, 3]

            height = video.shape[2]
            width = video.shape[3]

            # Store actual video from pred camera (GT target)
            data["video"] = video.to(dtype=torch.uint8)
            data["depth"] = depth.to(dtype=torch.float32)
            data["extrinsic_matrix"] = extrinsic_matrixs
            data["intrinsic_matrix"] = intrinsic_matrixs

            # Render merged pointcloud from pred camera view
            extrinsic_matrixs_render = campose_to_pytorch3d_mat_batch(extrinsic_matrixs)

            render_image, render_depth = self._render_pointcloud_batch(
                merged_pointclouds,
                extrinsic_matrixs=extrinsic_matrixs_render,
                intrinsic_matrixs=intrinsic_matrixs,
                point_size=4,
                height=height,
                width=width,
                background_color=(1, 1, 1),
                return_depth=True
            )

            render_image = render_image.permute(1, 0, 2, 3)  # [C, T, H, W]
            render_depth = render_depth.permute(1, 0, 2, 3)  # [C, T, H, W]

            data["pred_video"] = render_image
            data["pred_depth"] = render_depth
            data["randomview_id"] = randomview_id
            data["randomview_name"] = pred_cam
            data["annotation_file"] = ann_file

            # Unique key for sample
            if "episode_id" in label:
                data["__key__"] = f"{label['episode_id']}_rv{randomview_id}"
            else:
                data["__key__"] = f"{label['original_path']}_rv{randomview_id}"

            # T5 embeddings placeholder
            if self.load_t5_embeddings:
                t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
                data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings).cuda()
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16).cuda()
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, height, width).cuda()
            data["num_conditional_frames"] = self.sequence_length
            data["base_pos"] = np.array(label["base_pos"])
            data["base_quat"] = np.array(label["base_quat"])

            return data

        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index // self.num_randomviews]['ann_file']}. Skipped "
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    import mediapy
    import torch.nn.functional as F

    base_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_pnp"
    train_annotation_path = os.path.join(base_path, "annotation/train")
    val_annotation_path = os.path.join(base_path, "annotation/val")
    test_annotation_path = os.path.join(base_path, "annotation/test")

    output_dir = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/output"
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = NovelViewDataset(
        train_annotation_path=train_annotation_path,
        val_annotation_path=val_annotation_path,
        test_annotation_path=test_annotation_path,
        video_path=base_path,
        sequence_interval=1,
        num_frames=17,
        cam_ids=[
            'robot0_agentview_center',
            'robot0_eye_in_hand',
            'robot0_randomview_0',
            'robot0_randomview_1',
            'robot0_randomview_2',
            'robot0_randomview_3',
        ],
        gt_cams=['robot0_agentview_center', 'robot0_eye_in_hand'],
        randomview_names=[
            'robot0_randomview_0',
            'robot0_randomview_1',
            'robot0_randomview_2',
            'robot0_randomview_3',
        ],
        accumulate_action=False,
        video_size=[256, 256],
        val_start_frame_interval=1,
        mode="train",
    )

    print(f"\nDataset size: {len(train_dataset)}")
    print(f"Base sequences: {len(train_dataset.samples)}")
    print(f"Random views: {train_dataset.num_randomviews}")

    # Test loading samples and save videos
    num_samples_to_save = 8
    for i in range(num_samples_to_save):
        data = train_dataset[i]
        print(f"\nSample {i}:")
        print(f"  randomview_id: {data['randomview_id']}")
        print(f"  randomview_name: {data['randomview_name']}")
        print(f"  __key__: {data['__key__']}")
        print(f"  video shape: {data['video'].shape}")
        print(f"  pred_video shape: {data['pred_video'].shape}")

        # Get video and pred_video: [C, T, H, W]
        video = data["video"]  # [C, T, H, W]
        pred_video = data["pred_video"]  # [C, T, H, W]

        # Convert to numpy and permute to [T, H, W, C]
        video_np = video.permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.uint8)
        pred_video_np = pred_video.permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.uint8)

        # Concatenate horizontally (video on left, pred_video on right)
        combined_video = np.concatenate([video_np, pred_video_np], axis=2)  # [T, H, 2*W, C]

        # Save video
        output_path = os.path.join(output_dir, f"sample_{i}_rv{data['randomview_id']}_{data['randomview_name']}.mp4")
        mediapy.write_video(output_path, combined_video, fps=4)
        print(f"  Saved to: {output_path}")

    print(f"\nAll {num_samples_to_save} samples saved to {output_dir}")
