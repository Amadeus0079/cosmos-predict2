#!/usr/bin/env python3
"""
测试脚本：为单个episode生成静态场景的参考视频
"""

import sys
sys.path.insert(0, '/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/scripts')

from generate_camera_moving_videos import process_episode_ref_videos

# Test parameters
task_name = "PnPCounterToMicrowave"
episode_id = 79
split = "train"
hdf5_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/mg/2024-05-04-22-13-21_and_2024-05-07-07-41-17/demo_im256_ep100_cam2+4.hdf5"
annotation_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+4/PnPCounterToMicrowave/annotation/train/79.json"
output_dir = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+4"

print(f"Testing episode {episode_id} from {task_name}")
print(f"HDF5: {hdf5_path}")
print(f"Annotation: {annotation_path}")
print(f"Output: {output_dir}")

result = process_episode_ref_videos(
    task_name=task_name,
    episode_id=episode_id,
    split=split,
    hdf5_path=hdf5_path,
    annotation_path=annotation_path,
    output_dir=output_dir,
    camera_width=256,
    camera_height=256
)

if result:
    print(f"\nResult: {result}")
else:
    print("\n✓ Success!")
