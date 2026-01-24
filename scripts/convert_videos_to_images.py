#!/usr/bin/env python3
"""
Convert videos from robocasa_im256_ep100_cam2+8_fov60 dataset to images
and save them to robocasa_im256_ep100_pnpall_fov60 dataset structure.
Multi-processing version.
"""

import json
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def read_mapping(mapping_file):
    """Read the episode ID mapping from JSON file."""
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    return data['episode_id_mapping']


def convert_video_worker(args):
    """
    工作线程函数：处理单个视频的转换。
    args: (source_video, target_dir, fps)
    """
    source_video, target_dir, fps = args
    
    # 检查源文件是否存在
    if not source_video.exists():
        return "skip", str(source_video)

    # 创建输出目录
    os.makedirs(target_dir, exist_ok=True)

    # ffmpeg 命令
    cmd = [
        'ffmpeg',
        '-i', str(source_video),
        '-vf', f'fps={fps}',
        '-q:v', '2',
        f'{target_dir}/%06d.png',
        '-y',           # 覆盖已存在的文件
        '-loglevel', 'error' # 减少日志输出，避免多进程混杂
    ]

    try:
        subprocess.run(cmd, check=True)
        return "success", None
    except subprocess.CalledProcessError as e:
        return "error", f"{source_video}: {str(e)}"


def main():
    # --- 配置区域 ---
    base_dir = Path('/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2')
    mapping_file = base_dir / 'datasets/robocasa_im256_ep100_pnpall_fov60/episode_id_mapping.json'
    source_base = base_dir / 'datasets/robocasa_im256_ep100_cam2+8_fov75'
    target_base = base_dir / 'datasets/robocasa_im256_ep100_pnpall_fov75'
    
    cam_num = 8
    fps = 20
    # 建议进程数：如果是IO密集型，可以设为 cpu_count()；
    # 如果 ffmpeg 负载很高，可以设为 cpu_count() // 2
    num_workers = cpu_count() // 2
    # ----------------

    # 1. 读取映射
    print("Reading episode ID mapping...")
    mapping = read_mapping(mapping_file)
    print(f"Found {len(mapping)} episodes. Total potential videos: {len(mapping) * cam_num}")

    # 2. 构建任务列表
    tasks = []
    for idx, episode_info in mapping.items():
            
        task = episode_info['task']
        original_id = episode_info['original_id']
        split = episode_info['split']
        
        # if task != "PnPCounterToMicrowave":
        #     continue
        
        # if original_id != "19":
        #     continue

        for cam_id in range(cam_num):
            source_video = source_base / task / 'videos' / split / original_id / f'robot0_randomview_{cam_id}_ref.mp4'
            cam_name = f'robot0_randomview_{cam_id}'
            target_dir = target_base / 'videos' / split / idx / f'{cam_name}_ref'
            
            tasks.append((source_video, target_dir, fps))

    # 3. 并行执行
    print(f"Starting pool with {num_workers} workers...")
    success_count = 0
    skip_count = 0
    error_count = 0
    error_logs = []

    # 使用 imap_unordered 可以更早地看到进度条更新
    with Pool(processes=num_workers) as pool:
        pbar = tqdm(total=len(tasks), desc="Converting videos")
        for status, info in pool.imap_unordered(convert_video_worker, tasks):
            if status == "success":
                success_count += 1
            elif status == "skip":
                skip_count += 1
                # tqdm.write(f"Skipped: {info}") # 如果不想屏幕太乱可以注释掉
            else:
                error_count += 1
                error_logs.append(info)
            pbar.update(1)
        pbar.close()

    # 4. 打印总结
    print("\n" + "="*50)
    print("Conversion Summary:")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Successfully converted: {success_count}")
    print(f"  Skipped (not found): {skip_count}")
    print(f"  Errors: {error_count}")
    
    if error_logs:
        print("\nFirst 5 Error Details:")
        for log in error_logs[:5]:
            print(f"  - {log}")
    print("="*50)


if __name__ == '__main__':
    # 注意：在 Linux 上多进程通常不需要 if __name__ == '__main__':，
    # 但为了兼容性和防止递归启动，这是最佳实践。
    main()