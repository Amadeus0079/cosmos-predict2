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

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.action_conditioned.action_conditioned_dataset import ActionConditionedDataset
from cosmos_predict2.data.action_conditioned.multiview_dataset import MultiViewDataset
from imaginaire.lazy_config import LazyCall as L

base_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_eff/"
train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")
test_annotation_path = os.path.join(base_path, "annotation/test")


robocasa_train_dataset = L(MultiViewDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=9,
    cam_ids=['robot0_agentview_right', 'robot0_eye_in_hand', 'robot0_handview_right', 'robot0_handview_front'],
    gt_cams=['robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_eye_in_hand',],
    accumulate_action=False,
    video_size=[256, 256],
    val_start_frame_interval=1,
    mode="train",
)

robocasa_val_dataset = L(MultiViewDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=9,
    cam_ids=['robot0_agentview_right', 'robot0_eye_in_hand', 'robot0_handview_right', 'robot0_handview_front'],
    gt_cams=['robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_eye_in_hand'],
    accumulate_action=False,
    video_size=[256, 256],
    val_start_frame_interval=1,
    mode="val",
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


robocasa_train_dataloader = L(DataLoader)(
    dataset=robocasa_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_val_dataloader = L(DataLoader)(
    dataset=robocasa_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_training_and_val_data_action_conditioned():
    cs = ConfigStore.instance()

    # for local dataset
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_train",
        node=robocasa_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_val",
        node=robocasa_val_dataloader,
    )
