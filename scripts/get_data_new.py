import h5py
import numpy as np
import cv2
import os
import json
import argparse
import random
import concurrent.futures
from tqdm import tqdm
import math

INTERVAL = 1

def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def split_episodes(episodes, train_ratio, test_ratio, val_ratio):
    """将轨迹列表划分为训练集、测试集和验证集"""
    total = train_ratio + test_ratio + val_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"划分比例之和必须为1，当前为{total}")
    
    shuffled = random.sample(episodes, len(episodes))
    total_count = len(shuffled)
    train_count = int(total_count * train_ratio)
    test_count = int(total_count * test_ratio)
    
    return shuffled[:train_count], shuffled[train_count:train_count+test_count], shuffled[train_count+test_count:]

def process_episode(ep_name, data_group, episode_to_id, set_videos_dir, set_annotations_dir, split, cam_names):
    """处理单个轨迹的函数，供多线程调用"""
    
    episode = data_group[ep_name]
    
    ep_meta = json.loads(episode.attrs["ep_meta"])       # get meta data for episode
    lang = ep_meta["lang"]     
    
    # 检查必需的数据是否存在
    if 'actions' not in episode or 'obs' not in episode:
        return f"警告: 轨迹 {ep_name} 缺少 'actions' 或 'obs' 组，已跳过。"
    
    # 获取该轨迹的唯一序号ID
    episode_id = episode_to_id[ep_name]
    episode_id_str = str(episode_id)

    traj_len = len(episode['actions'][:])
    
    # 定义要取的序列
    index = list(range(1, traj_len, INTERVAL))
    prev_index = list(range(0, traj_len - 1, INTERVAL))
    
    # 提取动作和状态数据
    original_actions = episode['actions'][:][index][:-1]
    # if 'actions_abs' not in episode:
    #     return f"警告: 轨迹 {ep_name} 缺少 'actions_abs' 组，已跳过。"
    
    ### Change
    abs_actions = episode['actions_abs'][:][prev_index]
    original_pos = episode['obs/robot0_base_to_eef_pos'][:][index]
    original_quat = episode['obs/robot0_base_to_eef_quat'][:][index]
    original_ori = np.stack([quat2axisangle(original_quat[i]) for i in range(len(original_quat))], axis=0)
    original_states = np.concatenate([original_pos, original_ori, abs_actions[:, 6:]], axis=-1)
    
    # 提取抓取器状态
    if 'robot0_gripper_qpos' in episode['obs']:
        gripper_qpos = episode['obs']['robot0_gripper_qpos'][:][index]
        gripper_state_sequence = gripper_qpos[:, 0:1] - gripper_qpos[:, 1:]
        continuous_gripper_state_list = np.vstack([gripper_state_sequence]).flatten().tolist()
    else:
        num_states = len(original_actions) + 1
        continuous_gripper_state_list = [0.0] * num_states
        
    # 提取Base状态
    base_pos = episode['obs']['robot0_base_pos'][:][0]
    base_quat = episode['obs']['robot0_base_quat'][:][0]
    
    # 保存标注文件
    action_data = {
        "episode_id": episode_id_str,
        "task": "robot_trajectory_prediction",
        "texts": [lang],
        "videos": {cam_name: {"video_path": f"videos/{split}/{episode_id}/{cam_name}.mp4", "depth_path": f"videos/{split}/{episode_id}/{cam_name}.npy"} for cam_name in cam_names},
        "action": original_actions.tolist(),
        "state": original_states.tolist(),
        "continuous_gripper_state": continuous_gripper_state_list,
        "extrinsic_matrix": {cam_name: episode['cam_info'][cam_name]['extrinsic_matrix'][:].tolist() for cam_name in cam_names},
        "intrinsic_matrix": {cam_name: episode['cam_info'][cam_name]['intrinsic_matrix'][:].tolist() for cam_name in cam_names},
        "base_pos": base_pos.tolist(),
        "base_quat": base_quat.tolist(),
    }
    json_path = os.path.join(set_annotations_dir, f"{episode_id}.json")
    with open(json_path, 'w') as json_file:
        json.dump(action_data, json_file, indent=4)
    
    for cam_name in cam_names:
        video_frames = episode['obs'][cam_name + '_image'][:]
        depth_frames = episode['obs'][cam_name + '_depth'][:]
        num_frames, height, width, _ = video_frames.shape
        # height = 256  # debug
        # width = 256
        
        episode_video_dir = os.path.join(set_videos_dir, f"{episode_id}")
        os.makedirs(episode_video_dir, exist_ok=True)
        video_path_abs = os.path.join(episode_video_dir, f"{cam_name}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = 20.0
        out = cv2.VideoWriter(video_path_abs, fourcc, output_fps, (width, height))
        
        for frame in video_frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        out.release()
        
        depth_path_abs = os.path.join(episode_video_dir, f"{cam_name}.npy")
        np.save(depth_path_abs, depth_frames)
    
    return

def convert_robocasa_to_cosmos_format_format(hdf5_path, output_dir, domain_name, 
           train_ratio=0.7, test_ratio=0.2, val_ratio=0.1,
           random_seed=42, max_workers=None, cam_names=None):
    """主函数：读取HDF5文件并转换为目标格式，支持多线程加速"""
    print(f"开始处理文件: {hdf5_path}")
    print(f"输出将保存在: {output_dir}")
    print(f"数据集划分比例 - 训练集: {train_ratio}, 测试集: {test_ratio}, 验证集: {val_ratio}")
    print(f"使用多线程加速，线程数: {max_workers if max_workers else '自动'}")
    
    random.seed(random_seed)

    # 创建输出目录结构
    domain_output_dir = os.path.join(output_dir, domain_name)
    videos_dir = os.path.join(domain_output_dir, "videos")
    annotations_dir = os.path.join(domain_output_dir, "annotation")
    
    # 子目录结构
    dirs = {
        "train": (os.path.join(videos_dir, "train"), os.path.join(annotations_dir, "train")),
        "test": (os.path.join(videos_dir, "test"), os.path.join(annotations_dir, "test")),
        "val": (os.path.join(videos_dir, "val"), os.path.join(annotations_dir, "val"))
    }
    
    # 创建所有目录
    for video_dir, annot_dir in dirs.values():
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(annot_dir, exist_ok=True)

    try:
        f = h5py.File(hdf5_path, 'r')
        if 'data' not in f:
            print("错误: HDF5文件中找不到 'data' 组。")
            return

        data_group = f['data']
        episodes = sorted([key for key in data_group.keys() if key.startswith('demo_')])
        episodes = episodes[:300]  # debug
        print(f"在文件中找到 {len(episodes)} 条轨迹。")
        
        # 划分数据集
        train_episodes, test_episodes, val_episodes = split_episodes(
            episodes, train_ratio, test_ratio, val_ratio
        )
        
        print(f"划分结果 - 训练集: {len(train_episodes)}, 测试集: {len(test_episodes)}, 验证集: {len(val_episodes)}")

        # 为每个轨迹分配唯一ID
        episode_to_id = {episode: idx for idx, episode in enumerate(episodes)}

        # 处理所有子集
        all_episodes = [
            ("train", train_episodes, dirs["train"][0], dirs["train"][1]),
            ("test", test_episodes, dirs["test"][0], dirs["test"][1]),
            ("val", val_episodes, dirs["val"][0], dirs["val"][1])
        ]
        
        for set_name, set_episodes, set_videos_dir, set_annotations_dir in all_episodes:
            print(f"\n开始处理{set_name}集，共{len(set_episodes)}条轨迹")
            
            # 使用线程池并行处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 准备任务列表
                futures = [
                    executor.submit(
                        process_episode,
                        ep_name,
                        data_group,
                        episode_to_id,
                        set_videos_dir,
                        set_annotations_dir,
                        set_name,
                        cam_names
                    ) for ep_name in set_episodes
                ]
                
                # 监控进度并收集结果
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                    total=len(set_episodes), 
                                    desc=f"处理{set_name}集轨迹"):
                    result = future.result()
                    if result:
                        print(f"\n{result}")  # 输出警告或错误信息

    except Exception as e:
        print(f"处理文件时发生错误: {e}")

    print("\n数据转换完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将RoboCasa HDF5数据集转换为Cosmos-Predict2训练格式，支持多线程加速")
    parser.add_argument("--hdf5-path", type=str, 
                  default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_im128_depth_intvl1.hdf5", 
                  help="输入的RoboCasa HDF5文件路径。")
    parser.add_argument("--output-dir", type=str, 
                  default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets", 
                  help="转换后数据集的输出根目录。")
    parser.add_argument("--domain-name", type=str, default="robocasa_128",
                  help="为这个新数据集指定的领域名称。")
    parser.add_argument("--train-ratio", type=float, default=0.8, 
                  help="训练集所占比例 (默认: 0.7)")
    parser.add_argument("--test-ratio", type=float, default=0.1, 
                  help="测试集所占比例 (默认: 0.2)")
    parser.add_argument("--val-ratio", type=float, default=0.1, 
                  help="验证集所占比例 (默认: 0.1)")
    parser.add_argument("--random-seed", type=int, default=42, 
                  help="随机种子，确保划分结果可复现 (默认: 42)")
    parser.add_argument("--max-workers", type=int, default=10,
                  help="最大线程数，默认自动根据CPU核心数确定")
    parser.add_argument("--cam-names",
                        type=str,
                        nargs="+",
                        default=[
                            "robot0_agentview_left",
                            "robot0_agentview_right",
                            "robot0_eye_in_hand",
                            # "robot0_handview_left",
                            "robot0_handview_right",
                            "robot0_handview_front",
                            # "robot0_agentview_center",
                            # "robot0_frontview",
                            # "robot0_robotview",
                        ],
                        help="最大线程数，默认自动根据CPU核心数确定")
    
    args = parser.parse_args()

    convert_robocasa_to_cosmos_format_format(
        args.hdf5_path, 
        args.output_dir, 
        args.domain_name,
        args.train_ratio,
        args.test_ratio,
        args.val_ratio,
        args.random_seed,
        args.max_workers,
        args.cam_names,
    )