#!/usr/bin/env python3
"""
合并多个 RoboCasa 数据集为一个大数据集 (使用硬链接优化版)
"""

import os
import json
import shutil
import time
from tqdm import tqdm
from pathlib import Path

def merge_datasets():
    # --- 配置路径 ---
    base_dir = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets"
    source_dir = os.path.join(base_dir, "robocasa_im256_ep100_cam2+8_fov75")
    output_dir = os.path.join(base_dir, "robocasa_im256_ep100_pnpall_fov75")

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

    # --- 计时器初始化 ---
    script_start_time = time.time()
    
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"优化策略: 优先使用硬链接 (Hard Link)")

    # 创建输出目录结构
    for split in splits:
        os.makedirs(os.path.join(output_dir, "annotation", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "videos", split), exist_ok=True)

    total_episodes = 0
    episode_id_mapping = {}
    task_start_ids = {}
    current_episode_id = 0

    # 处理每个任务
    for task_name in tasks:
        task_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"处理任务: {task_name}")
        print(f"{'='*60}")

        task_dir = os.path.join(source_dir, task_name)
        task_start_ids[task_name] = current_episode_id

        # 细分计时变量
        json_total_time = 0
        link_total_time = 0

        for split in splits:
            ann_dir = os.path.join(task_dir, "annotation", split)
            video_dir = os.path.join(task_dir, "videos", split)

            if not os.path.exists(ann_dir):
                continue

            ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".json")],
                              key=lambda x: int(x.replace(".json", "")))

            pbar = tqdm(ann_files, desc=f"{split}")
            for ann_file in pbar:
                loop_start = time.time()

                # --- 1. 处理 Annotation (JSON) ---
                t0 = time.time()
                ann_path = os.path.join(ann_dir, ann_file)
                with open(ann_path, "r") as f:
                    data = json.load(f)

                original_id = data["episode_id"]
                new_episode_id = str(current_episode_id)
                data["episode_id"] = new_episode_id

                episode_id_mapping[new_episode_id] = {
                    "task": task_name,
                    "original_id": original_id,
                    "split": split
                }

                # 更新路径
                for cam_name in data["videos"]:
                    data["videos"][cam_name]["video_path"] = f"videos/{split}/{new_episode_id}/{cam_name}.mp4"
                    data["videos"][cam_name]["depth_path"] = f"videos/{split}/{new_episode_id}/{cam_name}.npy"

                # 写入新的 JSON (取消缩进以加快速度，如需美观可加 indent=4)
                new_ann_path = os.path.join(output_dir, "annotation", split, f"{new_episode_id}.json")
                with open(new_ann_path, "w") as f:
                    json.dump(data, f)
                
                t1 = time.time()
                json_total_time += (t1 - t0)

                # --- 2. 处理视频文件 (硬链接优化) ---
                t2 = time.time()
                old_video_folder = os.path.join(video_dir, original_id)
                new_video_folder = os.path.join(output_dir, "videos", split, new_episode_id)
                os.makedirs(new_video_folder, exist_ok=True)

                if os.path.exists(old_video_folder):
                    for file_name in os.listdir(old_video_folder):
                        src_file = os.path.join(old_video_folder, file_name)
                        dst_file = os.path.join(new_video_folder, file_name)
                        
                        if os.path.isfile(src_file):
                            if os.path.exists(dst_file):
                                os.remove(dst_file) # 如果已存在则删除，确保链接成功
                            
                            try:
                                # 使用硬链接代替拷贝
                                os.link(src_file, dst_file)
                            except OSError as e:
                                # 如果跨文件系统(Error 18)，则退回到物理拷贝
                                if e.errno == 18:
                                    shutil.copy2(src_file, dst_file)
                                else:
                                    raise e
                
                t3 = time.time()
                link_total_time += (t3 - t2)

                current_episode_id += 1
                total_episodes += 1

                # 实时更新状态
                if current_episode_id % 10 == 0:
                    pbar.set_postfix({
                        "JSON": f"{t1-t0:.3f}s",
                        "Link": f"{t3-t2:.3f}s",
                        "Speed": f"{(t3-loop_start):.3f}s/it"
                    })

        # 任务总结
        task_duration = time.time() - task_start_time
        print(f"  [任务完成] 耗时: {task_duration:.2f}s | "
              f"Link平均: {link_total_time/(len(ann_files)+1e-6):.4f}s")

    # 保存映射关系
    mapping_path = os.path.join(output_dir, "episode_id_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({
            "episode_id_mapping": episode_id_mapping,
            "task_start_ids": task_start_ids,
            "total_episodes": total_episodes,
            "tasks": tasks
        }, f, indent=4)

    # 最终汇总
    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    
    print(f"\n{'='*60}")
    print(f"所有任务合并完成！")
    print(f"总耗时: {total_duration/60:.2f} 分钟 ({total_duration:.2f} 秒)")
    print(f"总轨迹数: {total_episodes}")
    print(f"理论处理速度: {total_episodes/total_duration:.2f} it/s")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    merge_datasets()