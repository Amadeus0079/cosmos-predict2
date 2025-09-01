# convert_robocasa.py
# 最终版本：一个完整的脚本，用于将RoboCasa HDF5数据集转换为
# Cosmos-Predict2动作条件化模型所需的视频和JSON动作文件格式。

import h5py
import numpy as np
import cv2  # OpenCV库，用于视频写入
import os
import json
import argparse
from tqdm import tqdm # 用于显示漂亮的进度条

def convert_robocasa_to_cosmos_format(hdf5_path, output_dir, domain_name):
    """
    读取RoboCasa HDF5文件，并将其转换为Cosmos-Predict2训练所需的格式。

    Args:
        hdf5_path (str): 原始RoboCasa HDF5文件的路径。
        output_dir (str): 输出文件夹的根目录。
        domain_name (str): 为这个数据集指定的机器人类型/领域名称。
    """
    print(f"开始处理文件: {hdf5_path}")
    print(f"输出将保存在: {output_dir}")

    # 阶段一：创建输出目录结构
    domain_output_dir = os.path.join(output_dir, domain_name)
    videos_dir = os.path.join(domain_output_dir, "videos")
    actions_dir = os.path.join(domain_output_dir, "actions")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(actions_dir, exist_ok=True)

    try:
        # 阶段二：读取与遍历轨迹
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                print(f"错误: HDF5文件中找不到 'data' 组。")
                return

            data_group = f['data']
            episodes = sorted([key for key in data_group.keys() if key.startswith('demo_')])
            print(f"在文件中找到 {len(episodes)} 条轨迹。")

            # 循环处理每一条轨迹
            for ep_name in tqdm(episodes, desc="正在处理轨迹"):
                episode = data_group[ep_name]
                
                # 检查必需的数据是否存在
                if 'actions' not in episode or 'obs' not in episode:
                    print(f"警告: 轨迹 {ep_name} 缺少 'actions' 或 'obs' 组，已跳过。")
                    continue

                # --- 阶段三：核心数据转换 ---
                # 1. 提取和转换动作 (Delta Pose)
                original_actions = episode['actions'][:][:-1] # 形状: (T, 12)
                if 'actions_abs' not in episode or 'obs' not in episode:
                    print(f"警告: 轨迹 {ep_name} 缺少 'actions' 或 'obs' 组，已跳过。")
                    continue
                episode_id_str = ep_name.replace('demo_', '')

                # --- 阶段三：核心数据转换 ---
                # 1. 提取和转换动作 (Delta Pose)
                original_states = episode['actions_abs'][:] # 形状: (T, 12)
                # 关键逻辑：从12维中提取我们需要的6维末端执行器位移
                # 这是一个基于Robomimic/Robosuite标准格式的合理假设
                # 如果RoboCasa有特殊定义，只需修改这里的切片索引
                eef_pose_delta = original_actions[:, :6]

                # 2. 提取连续的Gripper状态
                if 'robot0_gripper_qpos' in episode['obs']:
                    # gripper_qpos 通常是 [left_finger_pos, right_finger_pos] (T, 2)
                    # 我们取第一个值或平均值作为代表性的开合状态
                    gripper_qpos = episode['obs']['robot0_gripper_qpos'][:] 
                    
                    # 提取单一一维作为代表状态
                    gripper_state_sequence = gripper_qpos[:, 0:1]-gripper_qpos[:,1:] # 保持(T, 1)的二维形状
                    
                    # 构造一个长度为T+1的列表，第一个是初始状态
                    initial_gripper_state = gripper_state_sequence[0:1] # 依然是(1, 1)
                    continuous_gripper_state_list = np.vstack([gripper_state_sequence]).flatten().tolist()
                else:
                    print(f"警告: 在轨迹 {ep_name} 中找不到 'robot0_gripper_qpos'，将使用0作为占位符。")
                    num_states = len(original_actions) + 1
                    continuous_gripper_state_list = [[0.0]] * num_states
                episode_video_dir = os.path.join(videos_dir, episode_id_str)
                os.makedirs(episode_video_dir, exist_ok=True)
                
                # 视频的相对路径也相应更新
                video_filename_rel = os.path.join("videos", episode_id_str, "rgb.mp4")

                # 3. 创建并保存动作JSON文件
                action_data = {
                    # 注意：这里的action只包含6维的eef pose delta
                    "action": original_actions.tolist(),
                    # continuous_gripper_state 是一个独立的列表
                    "state": original_states.tolist(), # 推理不需要，设为空列表
                    "continuous_gripper_state": continuous_gripper_state_list,
                    
                }
                
                json_filename = f"{ep_name.replace('demo_', '')}.json"
                json_path = os.path.join(actions_dir, json_filename)
                with open(json_path, 'w') as json_file:
                    json.dump(action_data, json_file, indent=4)

                # 4. 提取视频帧并保存为 .mp4
                if 'robot0_agentview_right_image' in episode['obs']:
                    video_frames = episode['obs']['robot0_agentview_right_image'][:]
                    
                    num_frames, height, width, _ = video_frames.shape
                    # 更新视频的绝对保存路径
                    video_path_abs = os.path.join(domain_output_dir, video_filename_rel)
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_fps = 20.0
                    out = cv2.VideoWriter(video_path_abs, fourcc, output_fps, (width, height))
                    
                    for frame in video_frames:
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(bgr_frame)
                    out.release()
                else:
                    print(f"警告: 在轨迹 {ep_name} 中找不到图像数据，已跳过视频生成。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

    print("\n数据转换完成！")


if __name__ == '__main__':
    # --- 阶段四：启动与执行 ---
    parser = argparse.ArgumentParser(description="将RoboCasa HDF5数据集转换为Cosmos-Predict2训练格式。")
    parser.add_argument("--hdf5-path", type=str, default ="datasets/robocasa_dataset/demo_gentex_im128_randcams.hdf5/demo_gentex_im128_randcams.hdf5", help="输入的RoboCasa HDF5文件路径。")
    parser.add_argument("--output-dir", type=str, default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets", help="转换后数据集的输出根目录。")
    parser.add_argument("--domain-name", type=str, default="test", help="为这个新数据集指定的领域名称。")
    args = parser.parse_args()

    convert_robocasa_to_cosmos_format(args.hdf5_path, args.output_dir, args.domain_name)