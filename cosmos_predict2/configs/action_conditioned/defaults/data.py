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
from cosmos_predict2.data.action_conditioned.multicam_dataset import MultiCamDataset
from cosmos_predict2.data.action_conditioned.multiview_gripper_dataset import MultiViewGripperDataset
from cosmos_predict2.data.action_conditioned.anyview_dataset import AnyViewDataset
from cosmos_predict2.data.action_conditioned.novelview_dataset import NovelViewDataset
from cosmos_predict2.data.action_conditioned.cheatview_dataset import CheatViewDataset
from imaginaire.lazy_config import LazyCall as L

base_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_128/"
train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")
test_annotation_path = os.path.join(base_path, "annotation/test")

bestview_base_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_bestview/"
bestview_train_annotation_path = os.path.join(bestview_base_path, "annotation/train")
bestview_val_annotation_path = os.path.join(bestview_base_path, "annotation/val")
bestview_test_annotation_path = os.path.join(bestview_base_path, "annotation/test")

anyview_base_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa/PnPCounterToSink"
anyview_train_annotation_path = os.path.join(anyview_base_path, "annotation/train")
anyview_val_annotation_path = os.path.join(anyview_base_path, "annotation/val")
anyview_test_annotation_path = os.path.join(anyview_base_path, "annotation/test")

novelview_base_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_pnp"
novelview_train_annotation_path = os.path.join(novelview_base_path, "annotation/train")
novelview_val_annotation_path = os.path.join(novelview_base_path, "annotation/val")
novelview_test_annotation_path = os.path.join(novelview_base_path, "annotation/test")

robocasa_train_dataset = L(MultiCamDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=9,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_eye_in_hand'],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="train",
)

robocasa_val_dataset = L(MultiCamDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=9,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_eye_in_hand'],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="val",
)

robocasa_long16_train_dataset = L(MultiViewDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_eye_in_hand',],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="train",
)

robocasa_long16_val_dataset = L(MultiViewDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_eye_in_hand'],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="val",
)

robocasa_bestview_train_dataset = L(MultiViewDataset)(
    train_annotation_path=bestview_train_annotation_path,
    val_annotation_path=bestview_val_annotation_path,
    test_annotation_path=bestview_test_annotation_path,
    video_path=bestview_base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand', "robot0_agentview_center"],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand', "robot0_agentview_center"],
    pred_cams=['robot0_agentview_center',],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="train",
)

robocasa_bestview_val_dataset = L(MultiViewDataset)(
    train_annotation_path=bestview_train_annotation_path,
    val_annotation_path=bestview_val_annotation_path,
    test_annotation_path=bestview_test_annotation_path,
    video_path=bestview_base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand', "robot0_agentview_center"],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand', "robot0_agentview_center"],
    pred_cams=['robot0_agentview_center'],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="val",
)

robocasa_gripper_train_dataset = L(MultiViewGripperDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_eye_in_hand',],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="train",
)

robocasa_gripper_val_dataset = L(MultiViewGripperDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_eye_in_hand'],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="val",
)

robocasa_handview_train_dataset = L(MultiViewDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand', 'robot0_handview_right'],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_handview_right',],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="train",
)

robocasa_handview_val_dataset = L(MultiViewDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand', 'robot0_handview_right'],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_handview_right',],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="val",
)

robocasa_frontview_train_dataset = L(MultiViewDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand', 'robot0_handview_front'],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_handview_front',],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="train",
)

robocasa_frontview_val_dataset = L(MultiViewDataset)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    sequence_interval=1,
    num_frames=17,
    cam_ids=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand', 'robot0_handview_front'],
    gt_cams=['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand',],
    pred_cams=['robot0_handview_front',],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="val",
)

robocasa_anyview_train_dataset = L(AnyViewDataset)(
    train_annotation_path=anyview_train_annotation_path,
    val_annotation_path=anyview_val_annotation_path,
    test_annotation_path=anyview_test_annotation_path,
    video_path=anyview_base_path,
    sequence_interval=2,
    num_frames=9,
    cam_ids=['robot0_agentview_center', 'robot0_eye_in_hand', 'robot0_activeview'],
    gt_cams=['robot0_agentview_center', 'robot0_eye_in_hand'],
    pred_cams=['robot0_activeview'],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="train",
)

robocasa_anyview_val_dataset = L(AnyViewDataset)(
    train_annotation_path=anyview_train_annotation_path,
    val_annotation_path=anyview_val_annotation_path,
    test_annotation_path=anyview_test_annotation_path,
    video_path=anyview_base_path,
    sequence_interval=2,
    num_frames=9,
    cam_ids=['robot0_agentview_center', 'robot0_eye_in_hand', 'robot0_activeview'],
    gt_cams=['robot0_agentview_center', 'robot0_eye_in_hand'],
    pred_cams=['robot0_activeview'],
    accumulate_action=False,
    video_size=[128, 128],
    val_start_frame_interval=1,
    mode="val",
)

robocasa_novelview_train_dataset = L(NovelViewDataset)(
    train_annotation_path=novelview_train_annotation_path,
    val_annotation_path=novelview_val_annotation_path,
    test_annotation_path=novelview_test_annotation_path,
    video_path=novelview_base_path,
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

robocasa_novelview_val_dataset = L(NovelViewDataset)(
    train_annotation_path=novelview_train_annotation_path,
    val_annotation_path=novelview_val_annotation_path,
    test_annotation_path=novelview_test_annotation_path,
    video_path=novelview_base_path,
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
    mode="val",
)

robocasa_novelview_short_train_dataset = L(NovelViewDataset)(
    train_annotation_path=novelview_train_annotation_path,
    val_annotation_path=novelview_val_annotation_path,
    test_annotation_path=novelview_test_annotation_path,
    video_path=novelview_base_path,
    sequence_interval=1,
    num_frames=9,
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

robocasa_novelview_short_val_dataset = L(NovelViewDataset)(
    train_annotation_path=novelview_train_annotation_path,
    val_annotation_path=novelview_val_annotation_path,
    test_annotation_path=novelview_test_annotation_path,
    video_path=novelview_base_path,
    sequence_interval=1,
    num_frames=9,
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
    mode="val",
)

robocasa_cheatview_train_dataset = L(CheatViewDataset)(
    train_annotation_path=novelview_train_annotation_path,
    val_annotation_path=novelview_val_annotation_path,
    test_annotation_path=novelview_test_annotation_path,
    video_path=novelview_base_path,
    sequence_interval=1,
    num_frames=9,
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

robocasa_cheatview_val_dataset = L(CheatViewDataset)(
    train_annotation_path=novelview_train_annotation_path,
    val_annotation_path=novelview_val_annotation_path,
    test_annotation_path=novelview_test_annotation_path,
    video_path=novelview_base_path,
    sequence_interval=1,
    num_frames=9,
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

robocasa_long16_train_dataloader = L(DataLoader)(
    dataset=robocasa_long16_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_long16_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_long16_val_dataloader = L(DataLoader)(
    dataset=robocasa_long16_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_long16_val_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_bestview_train_dataloader = L(DataLoader)(
    dataset=robocasa_bestview_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_bestview_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_bestview_val_dataloader = L(DataLoader)(
    dataset=robocasa_bestview_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_bestview_val_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_gripper_train_dataloader = L(DataLoader)(
    dataset=robocasa_gripper_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_gripper_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_gripper_val_dataloader = L(DataLoader)(
    dataset=robocasa_gripper_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_gripper_val_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_handview_train_dataloader = L(DataLoader)(
    dataset=robocasa_handview_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_handview_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_handview_val_dataloader = L(DataLoader)(
    dataset=robocasa_handview_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_handview_val_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_frontview_train_dataloader = L(DataLoader)(
    dataset=robocasa_frontview_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_frontview_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_frontview_val_dataloader = L(DataLoader)(
    dataset=robocasa_frontview_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_frontview_val_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_anyview_train_dataloader = L(DataLoader)(
    dataset=robocasa_anyview_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_anyview_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_anyview_val_dataloader = L(DataLoader)(
    dataset=robocasa_anyview_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_anyview_val_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_novelview_train_dataloader = L(DataLoader)(
    dataset=robocasa_novelview_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_novelview_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_novelview_val_dataloader = L(DataLoader)(
    dataset=robocasa_novelview_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_novelview_val_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_novelview_short_train_dataloader = L(DataLoader)(
    dataset=robocasa_novelview_short_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_novelview_short_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_novelview_short_val_dataloader = L(DataLoader)(
    dataset=robocasa_novelview_short_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_novelview_short_val_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_cheatview_train_dataloader = L(DataLoader)(
    dataset=robocasa_cheatview_train_dataset,
    sampler=L(get_sampler)(dataset=robocasa_cheatview_train_dataset),
    batch_size=1,
    drop_last=True,
)

robocasa_cheatview_val_dataloader = L(DataLoader)(
    dataset=robocasa_cheatview_val_dataset,
    sampler=L(get_sampler)(dataset=robocasa_cheatview_val_dataset),
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
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_gripper_train",
        node=robocasa_gripper_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_gripper_val",
        node=robocasa_gripper_val_dataloader,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_long16_train",
        node=robocasa_long16_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_long16_val",
        node=robocasa_long16_val_dataloader,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_bestview_train",
        node=robocasa_bestview_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_bestview_val",
        node=robocasa_bestview_val_dataloader,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_handview_train",
        node=robocasa_handview_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_handview_val",
        node=robocasa_handview_val_dataloader,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_frontview_train",
        node=robocasa_frontview_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_frontview_val",
        node=robocasa_frontview_val_dataloader,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_anyview_train",
        node=robocasa_anyview_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_anyview_val",
        node=robocasa_anyview_val_dataloader,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_novelview_train",
        node=robocasa_novelview_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_novelview_val",
        node=robocasa_novelview_val_dataloader,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_novelview_short_train",
        node=robocasa_novelview_short_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_novelview_short_val",
        node=robocasa_novelview_short_val_dataloader,
    )
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="robocasa_cheatview_train",
        node=robocasa_cheatview_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="robocasa_cheatview_val",
        node=robocasa_cheatview_val_dataloader,
    )