#!/usr/bin/env python3
"""
合并多个 RoboCasa 数据集为一个大数据集

将 robocasa_im256_ep100_cam2+4 下的 8 个子任务数据集合并为一个数据集 robocasa_im256_ep100_pnp
"""

import os
import json
import shutil
from tqdm import tqdm
from pathlib import Path


def merge_datasets():
    # 配置路径
    base_dir = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets"
    source_dir = os.path.join(base_dir, "robocasa_im256_ep100_cam2+4")
    output_dir = os.path.join(base_dir, "robocasa_im256_ep100_pnp")

    # 8个子任务名称
    tasks = [
        "PnPCabToCounter",
        "PnPCounterToCab",
        "PnPCounterToMicrowave",
        "PnPCounterToSink",
        "PnPCounterToStove",
        "PnPMicrowaveToCounter",
        "PnPSinkToCounter",
        "PnPStoveToCounter",
    ]

    splits = ["train", "test", "val"]

    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"子任务数量: {len(tasks)}")
    print(f"数据划分: {splits}")

    # 创建输出目录结构
    for split in splits:
        os.makedirs(os.path.join(output_dir, "annotation", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "videos", split), exist_ok=True)

    print("\n开始合并数据集...")

    # 统计信息
    total_episodes = 0
    episode_id_mapping = {}  # 记录 episode_id 到 (task, original_id) 的映射
    task_start_ids = {}  # 记录每个任务的起始 episode_id

    current_episode_id = 0

    # 处理每个任务
    for task_name in tasks:
        print(f"\n{'='*60}")
        print(f"处理任务: {task_name}")
        print(f"{'='*60}")

        task_dir = os.path.join(source_dir, task_name)
        task_start_ids[task_name] = current_episode_id

        # 统计该任务的文件数
        task_counts = {}
        for split in splits:
            ann_dir = os.path.join(task_dir, "annotation", split)
            if os.path.exists(ann_dir):
                task_counts[split] = len([f for f in os.listdir(ann_dir) if f.endswith(".json")])
            else:
                task_counts[split] = 0

        print(f"任务统计 - Train: {task_counts.get('train', 0)}, Test: {task_counts.get('test', 0)}, Val: {task_counts.get('val', 0)}")

        # 处理每个 split
        for split in splits:
            ann_dir = os.path.join(task_dir, "annotation", split)
            video_dir = os.path.join(task_dir, "videos", split)

            if not os.path.exists(ann_dir):
                continue

            ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")],
                              key=lambda x: int(x.replace(".json", "")))

            print(f"\n处理 {split} 集, 共 {len(ann_files)} 条轨迹")

            for ann_file in tqdm(ann_files, desc=f"{task_name}/{split}"):
                # 读取原始 annotation
                ann_path = os.path.join(ann_dir, ann_file)
                with open(ann_path, "r") as f:
                    data = json.load(f)

                # 获取原始 episode_id
                original_id = data["episode_id"]

                # 更新 episode_id
                new_episode_id = str(current_episode_id)
                data["episode_id"] = new_episode_id

                # 记录映射关系
                episode_id_mapping[new_episode_id] = {
                    "task": task_name,
                    "original_id": original_id,
                    "split": split
                }

                # 更新视频路径
                for cam_name in data["videos"]:
                    old_video_path = data["videos"][cam_name]["video_path"]
                    old_depth_path = data["videos"][cam_name]["depth_path"]

                    # 更新路径格式: videos/train/0/robot0_eye_in_hand.mp4
                    data["videos"][cam_name]["video_path"] = f"videos/{split}/{new_episode_id}/{cam_name}.mp4"
                    data["videos"][cam_name]["depth_path"] = f"videos/{split}/{new_episode_id}/{cam_name}.npy"

                # 保存新的 annotation 文件
                new_ann_path = os.path.join(output_dir, "annotation", split, f"{new_episode_id}.json")
                with open(new_ann_path, "w") as f:
                    json.dump(data, f, indent=4)

                # 复制视频和深度文件
                old_video_folder = os.path.join(video_dir, original_id)
                new_video_folder = os.path.join(output_dir, "videos", split, new_episode_id)
                os.makedirs(new_video_folder, exist_ok=True)

                # 复制该文件夹下的所有视频和深度文件
                if os.path.exists(old_video_folder):
                    for file_name in os.listdir(old_video_folder):
                        src_file = os.path.join(old_video_folder, file_name)
                        dst_file = os.path.join(new_video_folder, file_name)
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, dst_file)

                current_episode_id += 1
                total_episodes += 1

    # 保存 episode_id 映射关系（可选，用于调试）
    mapping_path = os.path.join(output_dir, "episode_id_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({
            "episode_id_mapping": episode_id_mapping,
            "task_start_ids": task_start_ids,
            "total_episodes": total_episodes,
            "tasks": tasks
        }, f, indent=4)

    # 输出统计信息
    print(f"\n{'='*60}")
    print("合并完成!")
    print(f"{'='*60}")
    print(f"总轨迹数: {total_episodes}")
    print(f"\n任务起始 ID 映射:")
    for task_name, start_id in task_start_ids.items():
        if task_name in task_start_ids:
            next_task_idx = list(task_start_ids.keys()).index(task_name) + 1
            if next_task_idx < len(task_start_ids):
                next_task = list(task_start_ids.keys())[next_task_idx]
                end_id = task_start_ids[next_task] - 1
            else:
                end_id = total_episodes - 1
            print(f"  {task_name}: {start_id} - {end_id}")

    print(f"\n输出目录: {output_dir}")
    print(f"映射文件: {mapping_path}")

    # 验证输出
    print(f"\n验证输出文件...")
    for split in splits:
        ann_dir = os.path.join(output_dir, "annotation", split)
        video_dir = os.path.join(output_dir, "videos", split)
        ann_count = len([f for f in os.listdir(ann_dir) if f.endswith(".json")]) if os.path.exists(ann_dir) else 0
        video_folders = len(os.listdir(video_dir)) if os.path.exists(video_dir) else 0
        print(f"  {split}: {ann_count} annotations, {video_folders} video folders")


if __name__ == "__main__":
    merge_datasets()
