import os
import subprocess
import time
import argparse
from pathlib import Path

# ================= 配置区域 =================
# 原脚本的文件名
SCRIPT_NAME = "generate_camera_moving_videos.py"  
# 你的数据集根目录 (用来自动扫描这8个任务的名字)
DATASET_DIR = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+8_fov75"
# 可用的 GPU ID 列表
GPU_IDS = [1,2,3]
# 想要并行的任务总数 (如果不指定，则跑文件夹下所有)
TARGET_TASK_COUNT = 8
# 单个脚本内部的线程数 (因为一张卡跑两个脚本，建议把这个调小一点，防止CPU爆炸)
WORKERS_PER_SCRIPT = 5
# ===========================================

def get_task_list(dataset_dir):
    """扫描目录下所有的子文件夹作为任务名"""
    path = Path(dataset_dir)
    tasks = [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    tasks.sort() # 排序保证分配确定性
    return tasks

def main():
    # 1. 获取任务列表
    all_tasks = get_task_list(DATASET_DIR)
    
    if len(all_tasks) == 0:
        print(f"错误: 在 {DATASET_DIR} 下未找到任何任务文件夹")
        return

    # 如果任务很多，只取前8个，或者根据你的需求修改
    tasks_to_run = all_tasks[:TARGET_TASK_COUNT] if TARGET_TASK_COUNT else all_tasks
    
    print(f"检测到 {len(all_tasks)} 个任务，准备处理前 {len(tasks_to_run)} 个任务。")
    print(f"可用 GPU: {GPU_IDS}")
    
    processes = []
    
    # 2. 循环分配任务
    for i, task_name in enumerate(tasks_to_run):
        # 使用取余操作分配 GPU (Round-Robin 策略)
        # 任务 0 -> GPU 0
        # 任务 1 -> GPU 1
        # ...
        # 任务 4 -> GPU 0 (实现了一张卡跑多个)
        gpu_idx = i % len(GPU_IDS)
        gpu_id = GPU_IDS[gpu_idx]
        
        print(f"🚀 [任务 {i+1}/{len(tasks_to_run)}] 分配任务 '{task_name}' 到 GPU {gpu_id}")
        
        # 3. 构建环境变量
        env = os.environ.copy()
        # 关键：指定该进程可见的 GPU
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # 关键：MuJoCo 使用 EGL 后端进行无头渲染
        env["MUJOCO_GL"] = "egl" 
        
        # 4. 构建命令
        cmd = [
            "python3", SCRIPT_NAME,
            "--task-name", task_name,
            "--max-workers", str(WORKERS_PER_SCRIPT),
            "--dataset-dir", DATASET_DIR
            # 如果有其他参数可以在这里追加
        ]
        
        # 5. 异步启动进程 (Popen 不会阻塞主线程)
        # stdout/stderr 可以重定向到文件以防终端输出混乱，这里为了简单直接打印
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        
        # 稍微暂停一下，避免瞬间同时启动冲击 CPU/IO
        time.sleep(2)

    print("\n✅ 所有任务已启动，正在后台运行...")
    print("请等待所有子进程结束 (按 Ctrl+C 可强行终止所有)...")

    # 6. 等待所有进程结束
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\n⚠️  检测到中断，正在终止所有子进程...")
        for p in processes:
            p.terminate()
            
    print("🎉 所有任务处理完成！")

if __name__ == "__main__":
    main()