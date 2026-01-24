import os
import math

# --- 配置区域 ---
dataset_dir = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+8_fov60"
robocasa_base = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp"
script_name = "generate_single.py" # 上面那个Python文件的名字

# 你的可用 GPU ID 列表
# gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7] 
gpu_ids = [0, 1, 2, 3] # 如果只有4张卡

# 每个任务有多少个 episode (你说是100)
total_episodes = 100 
# --- 配置结束 ---

def generate_commands():
    # 获取所有任务名
    tasks = [d for d in os.listdir(dataset_dir) 
             if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')]
    tasks.sort()

    print(f"找到 {len(tasks)} 个任务。")
    print(f"可用 GPU: {len(gpu_ids)}")
    print("-" * 60)
    print("请复制以下命令到不同的 tmux 窗口中运行：")
    print("-" * 60)

    # 策略：
    # 方案 A: 如果任务数 >= GPU数，每个 GPU 分几个完整的任务 (最简单)
    # 方案 B: 你的需求是将同一个任务的不同 episode 分到不同卡。我们采用方案 B。
    
    # 我们将为每个 GPU 生成一段长命令，这段命令会依次处理：
    # Task1_Part1, Task2_Part1 ... 
    
    commands_per_gpu = {gpu: [] for gpu in gpu_ids}
    
    # 将 0-99 分配给 GPU
    # 例如 8张卡，每张卡分 100/8 = 12.5 个 episode
    episodes = list(range(total_episodes))
    chunk_size = math.ceil(total_episodes / len(gpu_ids))
    
    # 为每个任务生成分片
    for task in tasks:
        for i, gpu in enumerate(gpu_ids):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_episodes)
            
            if start_idx >= total_episodes:
                continue
                
            # 生成 ID 列表字符串 "0,1,2,3..."
            subset_ids = episodes[start_idx:end_idx]
            if not subset_ids: continue
            
            ids_str = ",".join(map(str, subset_ids))
            
            cmd = (
                f"python {script_name} "
                f"--task-name {task} "
                f"--episode-ids {ids_str} "
                f"--dataset-dir {dataset_dir} "
                f"--robocasa-base {robocasa_base}"
            )
            commands_per_gpu[gpu].append(cmd)

    # 打印结果
    for gpu in gpu_ids:
        print(f"\n### Tmux Window / Terminal for GPU {gpu} ###")
        print(f"export CUDA_VISIBLE_DEVICES={gpu}")
        
        # 将所有命令用 '&&' 连接，这样一个跑完跑下一个
        full_command = " && \\\n".join(commands_per_gpu[gpu])
        print(full_command)
        print("\n")

if __name__ == "__main__":
    generate_commands()