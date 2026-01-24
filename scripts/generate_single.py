#!/usr/bin/env python3
"""
生成静态场景的参考视频 (串行版)
支持指定 episode_ids，适合用于外部并行调度 (如 tmux/slurm)
"""

import os
# 必须在导入 mujoco/robosuite 之前设置
os.environ["MUJOCO_GL"] = "egl"

import h5py
import numpy as np
import cv2
import json
import argparse
import traceback
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Import robocasa dependencies
try:
    import robocasa
    from robocasa.utils.env_utils import create_env
    import robosuite
except ImportError as e:
    raise ImportError("RoboCasa not found.") from e


# --- Helper Functions (保持不变) ---

def _nearest_rotation(R_mat: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(R_mat)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho

def invert_camera_extrinsic_matrix(R_extrinsic: np.ndarray):
    R_extrinsic = np.asarray(R_extrinsic, dtype=float)
    C = np.diag([1.0, -1.0, -1.0, 1.0])
    pose = R_extrinsic @ C
    camera_rot_raw = pose[:3, :3]
    camera_pos = pose[:3, 3].copy()
    camera_rot = _nearest_rotation(camera_rot_raw)
    return camera_pos, camera_rot

def set_camera_pose_world(sim, cam_id, pos_world, rot_world_3x3):
    quat_wxyz = R.from_matrix(rot_world_3x3).as_quat()
    quat_wxyz = np.roll(quat_wxyz, 1)
    sim.model.cam_pos[cam_id] = np.asarray(pos_world, dtype=np.float64)
    sim.model.cam_quat[cam_id] = np.asarray(quat_wxyz, dtype=np.float64)
    sim.forward()

def reset_to(env, state):
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)

        env.generative_textures = "100p"
        env.reset()
        
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml
            xml = postprocess_model_xml(state["model"])
        else:
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()

    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()

    if hasattr(env, "update_sites"):
        env.update_sites()
    return None

def find_hdf5_file(task_name, base_dir):
    task_dir = os.path.join(base_dir, task_name)
    if not os.path.exists(task_dir): return None
    for root, dirs, files in os.walk(task_dir):
        for file in files:
            if file == "demo_im256_ep100_cam2+4.hdf5":
                return os.path.join(root, file)
    return None

def get_demo_key_from_episode_id(hdf5_path, episode_id):
    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f: return None
        data_group = f['data']
        episodes = sorted([key for key in data_group.keys() if key.startswith('demo_')])
        if episode_id < 0 or episode_id >= len(episodes): return None
        return episodes[episode_id]

# --- Main Processing Logic (串行化) ---

def process_episode(task_name, episode_id, split, hdf5_path, annotation_path, output_dir, width, height):
    env = None
    try:
        # 1. Load Annotation
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # 2. Get Demo Key & Data
        demo_key = get_demo_key_from_episode_id(hdf5_path, episode_id)
        if demo_key is None:
            return f"[Skipped] Ep {episode_id}: Demo key not found"

        initial_state = {}
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f or demo_key not in f['data']:
                return f"[Error] Ep {episode_id}: Key missing in HDF5"
            demo_data = f['data'][demo_key]
            initial_state["states"] = f[f"data/{demo_key}/states"][()][0]
            initial_state["model"] = demo_data.attrs["model_file"]
            initial_state["ep_meta"] = demo_data.attrs.get("ep_meta", None)

        # 3. Create Env
        env = create_env(
            env_name=task_name,
            robots="PandaMobile",
            camera_names=["robot0_randomview_0", "robot0_agentview_center"],
            camera_widths=width,
            camera_heights=height,
            seed=42,
            render_onscreen=False,
        )

        # 4. Reset
        reset_to(env, initial_state)
        # env.reset()

        # 5. Check Camera
        available_cam_name = None
        for test_cam in ["robot0_randomview_0", "robot0_agentview_center", "robot0_eye_in_hand"]:
            try:
                env.sim.model.camera_name2id(test_cam)
                available_cam_name = test_cam
                break
            except: continue
        
        if not available_cam_name:
            env.close()
            return f"[Error] Ep {episode_id}: No valid camera"

        cam_id = env.sim.model.camera_name2id(available_cam_name)

        # 6. Render Loop
        video_output_dir = os.path.join(output_dir, task_name, "videos", split, str(episode_id))
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Precompute base transform
        T_base = np.eye(4)
        T_base[:3, 3] = np.array(annotation['base_pos'])
        T_base[:3, :3] = R.from_quat(np.array(annotation['base_quat'])).as_matrix()
        T_base_inv = np.linalg.inv(T_base)

        randomview_cameras = [c for c in annotation['videos'].keys() if c.startswith('robot0_randomview_')]
        generated = 0
        
        for cam_name in randomview_cameras:
            out_path = os.path.join(video_output_dir, f"{cam_name}_ref.mp4")
            if os.path.exists(out_path): continue # Skip existing

            if cam_name not in annotation['extrinsic_matrix']: continue
            extrinsic_matrices = np.array(annotation['extrinsic_matrix'][cam_name])
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))

            for mat in extrinsic_matrices:
                mat_world = T_base_inv @ mat
                pos, rot = invert_camera_extrinsic_matrix(mat_world)
                set_camera_pose_world(env.sim, cam_id, pos, rot)
                
                frame = env.sim.render(camera_name=available_cam_name, width=width, height=height, depth=False)
                out.write(cv2.cvtColor(frame[::-1], cv2.COLOR_RGB2BGR))
            
            out.release()
            generated += 1

        env.close()
        return None if generated > 0 else f"Ep {episode_id}: Already done or skipped"

    except Exception as e:
        if env: env.close()
        return f"[Exception] Ep {episode_id}: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str,
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+8_fov60",
                       help="数据集根目录")
    parser.add_argument("--robocasa-base", type=str,
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp",
                       help="RoboCasa数据集基础目录")
    parser.add_argument("--task-name", type=str, default="PnPCabToCounter",
                       help="指定任务名称（如果为None则处理所有任务）")
    parser.add_argument("--episode-ids", type=str, default=None,
                       help="指定要处理的 IDs，例如 '5' 或 '5,7,8'。不传则处理所有。")
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    args = parser.parse_args()

    # 解析 target_ids
    target_ids = None
    if args.episode_ids:
        try:
            target_ids = set([int(x.strip()) for x in args.episode_ids.split(',') if x.strip()])
            print(f"模式: 仅处理指定的 {len(target_ids)} 个 Episodes: {target_ids}")
        except ValueError:
            print("Error: --episode-ids 格式错误，应为数字列表如 1,2,3")
            return

    # 获取任务列表
    if args.task_name:
        tasks = [args.task_name]
    else:
        tasks = [d.name for d in Path(args.dataset_dir).iterdir() if d.is_dir() and not d.name.startswith('.')]

    for task in tasks:
        print(f"\n{'='*40}\nTask: {task}\n{'='*40}")
        
        hdf5_path = find_hdf5_file(task, args.robocasa_base)
        if not hdf5_path:
            print(f"No HDF5 found for {task}")
            continue

        task_anno_dir = os.path.join(args.dataset_dir, task, "annotation")
        
        # 收集该任务下所有符合条件的 episodes
        todo_list = []
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(task_anno_dir, split)
            if not os.path.exists(split_dir): continue
            
            for f in os.listdir(split_dir):
                if f.endswith(".json"):
                    ep_id = int(f.replace(".json", ""))
                    
                    # 过滤逻辑
                    if target_ids is not None:
                        if ep_id not in target_ids:
                            continue
                    
                    todo_list.append((task, ep_id, split, os.path.join(split_dir, f)))

        todo_list.sort(key=lambda x: x[1])
        print(f"待处理: {len(todo_list)} 个 episodes")

        # 纯串行循环
        pbar = tqdm(todo_list)
        for item in pbar:
            t_name, e_id, spl, anno_path = item
            pbar.set_description(f"Ep {e_id}")
            
            msg = process_episode(t_name, e_id, spl, hdf5_path, anno_path, 
                                  args.dataset_dir, args.camera_width, args.camera_height)
            
            if msg:
                # 只打印异常或重要信息
                if "Error" in msg or "Exception" in msg:
                    tqdm.write(msg)

if __name__ == "__main__":
    main()