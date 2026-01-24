#!/usr/bin/env python3
"""
批量处理 RoboCasa kitchen_pnp 任务的脚本
自动查找所有 kitchen_pnp 任务下的 demo_im256_ep100_cam2+4.hdf5 文件并执行转换
"""

import os
import subprocess
import sys
from pathlib import Path

# 基础路径配置
ROBOCASA_BASE = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/robocasa/datasets_hdd/v0.1/single_stage/kitchen_pnp"
OUTPUT_DIR = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+8_fov75"
SCRIPT_PATH = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/scripts/get_robocasa_data.py"

# 定义任务列表（手动指定，确保顺序和完整性）
TASKS = [
    "PnPCabToCounter",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "PnPMicrowaveToCounter",
    "PnPSinkToCounter",
    "PnPStoveToCounter",
]

def find_hdf5_file(task_name):
    """在指定任务目录下查找 demo_im256_ep100_cam2+8_fov60.hdf5 文件"""
    task_dir = os.path.join(ROBOCASA_BASE, task_name)

    for root, dirs, files in os.walk(task_dir):
        for file in files:
            if file == "demo_im256_ep100_cam2+8_fov75.hdf5":
                return os.path.join(root, file)
    return None

def run_conversion(task_name, hdf5_path):
    """执行单个任务的转换"""
    cmd = [
        "python3",
        SCRIPT_PATH,
        "--hdf5-path", hdf5_path,
        "--output-dir", OUTPUT_DIR,
        "--domain-name", task_name,
        "--train-ratio", "0.8",
        "--test-ratio", "0.1",
        "--val-ratio", "0.1",
        "--max-workers", "5",
    ]

    print(f"\n{'='*60}")
    print(f"开始处理任务: {task_name}")
    print(f"输入文件: {hdf5_path}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"任务 {task_name} 处理成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"任务 {task_name} 处理失败，返回码: {e.returncode}")
        return False
    except Exception as e:
        print(f"任务 {task_name} 处理时发生异常: {e}")
        return False

def main():
    """主函数"""
    print(f"RoboCasa kitchen_pnp 批量处理脚本")
    print(f"基础目录: {ROBOCASA_BASE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"共 {len(TASKS)} 个任务待处理\n")

    # 检查脚本是否存在
    if not os.path.exists(SCRIPT_PATH):
        print(f"错误: 脚本文件不存在: {SCRIPT_PATH}")
        sys.exit(1)

    results = {}

    for task in TASKS:
        # 查找 hdf5 文件
        hdf5_path = find_hdf5_file(task)

        if not hdf5_path:
            print(f"警告: 未找到任务 {task} 的 demo_im256_ep100_cam2+8_fov60.hdf5 文件，跳过")
            results[task] = "未找到文件"
            continue

        # 执行转换
        success = run_conversion(task, hdf5_path)
        results[task] = "成功" if success else "失败"

    # 打印汇总结果
    print(f"\n{'='*60}")
    print("处理汇总:")
    print(f"{'='*60}")
    for task, status in results.items():
        print(f"  {task}: {status}")

    success_count = sum(1 for s in results.values() if s == "成功")
    print(f"\n总计: {success_count}/{len(TASKS)} 个任务成功")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
