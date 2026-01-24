#!/usr/bin/env python3
"""
生成静态场景的参考视频 (多进程高稳定版)
包含 IO锁 和 初始化锁，彻底隔离环境创建冲突
"""

import os
# 必须在任何 robosuite/mujoco 导入之前设置
os.environ["MUJOCO_GL"] = "egl"

import json
import argparse
import concurrent.futures
import multiprocessing as mp
import traceback
import time
from pathlib import Path

import h5py
import numpy as np
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# 延迟导入，防止全局污染
try:
    import robocasa
    from robocasa.utils.env_utils import create_env
    import robosuite
    import robosuite.utils.camera_utils as camera_utils
    # 尝试导入纹理相关，用于确保路径正确
    try:
        from robocasa.utils.texture_utils import TEXTURES_DIR
    except ImportError:
        TEXTURES_DIR = None # 后续处理
except ImportError as e:
    raise ImportError("RoboCasa/Robosuite not found.") from e


# --- Helper Functions ---

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
    """
    重置环境状态。
    注意：此函数现在假设在 reset_lock 保护下运行。
    """
    if "model" in state:
        # 处理 ep_meta
        if state.get("ep_meta", None) is not None:
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
            
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)

        # 关键修改：
        # 如果从 HDF5 读取的 model xml 已经是替换过纹理的（例如 demo_im256.hdf5），
        # 再次运行 generative_textures="100p" 可能会导致找不到原始材质名而报错。
        # 但 Robocasa 通常需要这一步来加载贴图。
        # 我们保持原样，但在外部捕获错误。
        env.generative_textures = "100p"
        
        # 这一步会触发 replace_wall_texture 等逻辑
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

def process_episode_ref_videos(task_name, episode_id, split, hdf5_path, annotation_path,
                               output_dir, camera_width, camera_height, 
                               io_lock, reset_lock):
    """
    worker function
    io_lock: 保护 HDF5 读取
    reset_lock: 保护 env.reset() 和纹理生成
    """
    env = None
    try:
        # 1. 准备数据容器
        initial_state = {}
        
        # 2. 读取 Annotation
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # 3. 读取 HDF5 (加锁)
        # 这里的锁是为了防止 HDF5 库在多进程下读取变长字符串(XML)时发生内存截断
        with io_lock:
            with h5py.File(hdf5_path, 'r') as f:
                if 'data' not in f: return f"Error: Bad HDF5 {hdf5_path}"
                data_group = f['data']
                episodes = sorted([k for k in data_group.keys() if k.startswith('demo_')])
                if not (0 <= episode_id < len(episodes)): return f"Error: Ep ID {episode_id} out of range"
                
                demo_key = episodes[episode_id]
                demo_data = data_group[demo_key]
                
                initial_state["states"] = f[f"data/{demo_key}/states"][()][0]
                initial_state["model"] = demo_data.attrs["model_file"] # XML string
                initial_state["ep_meta"] = demo_data.attrs.get("ep_meta", None)

        if not initial_state.get("model"):
            return "Error: Empty XML model"

        # 4. 环境初始化 (加锁！！！)
        # 这是解决 "wall_tex_name is None" 和纹理读写冲突的关键
        # 我们强制串行执行环境创建和重置，但在渲染时并行
        with reset_lock:
            # 创建环境
            env = create_env(
                env_name=task_name,
                robots="PandaMobile",
                camera_names=["robot0_randomview_0", "robot0_agentview_center"],
                camera_widths=camera_width,
                camera_heights=camera_height,
                seed=42,
                render_onscreen=False,
            )
            
            # 尝试重置
            try:
                reset_to(env, initial_state)
            except AssertionError as e:
                # 捕获特定的 wall texture 错误
                if "wall_tex_name" in str(e) or "None" in str(e):
                    # 将出错的 XML 保存下来用于调试
                    debug_path = os.path.join(output_dir, f"debug_failed_xml_ep{episode_id}.xml")
                    with open(debug_path, "w") as f:
                        f.write(initial_state["model"])
                    raise ValueError(f"纹理替换失败，XML结构可能不匹配。已保存出错XML至: {debug_path}") from e
                else:
                    raise e
            
            # env.reset() # 再次reset确保状态同步
            
            # 获取相机 ID (在锁内完成)
            available_cam_name = None
            for test_cam in ["robot0_randomview_0", "robot0_agentview_center", "robot0_eye_in_hand"]:
                try:
                    env.sim.model.camera_name2id(test_cam)
                    available_cam_name = test_cam
                    break
                except: continue
            
            if not available_cam_name:
                raise ValueError("No valid camera found")
            
            cam_id = env.sim.model.camera_name2id(available_cam_name)

        # --- 锁释放，以下是耗时的渲染过程，可以安全并行 ---

        # 准备输出目录
        video_output_dir = os.path.join(output_dir, task_name, "videos", split, str(episode_id))
        os.makedirs(video_output_dir, exist_ok=True)

        # 准备坐标变换
        T_base = np.eye(4)
        base_pos = np.array(annotation['base_pos'])
        base_quat = np.array(annotation['base_quat'])
        T_base[:3, 3] = base_pos
        T_base[:3, :3] = R.from_quat(base_quat).as_matrix()
        T_base_inv = np.linalg.inv(T_base)

        randomview_cameras = [c for c in annotation['videos'].keys() if c.startswith('robot0_randomview_')]
        generated_count = 0

        for cam_name in randomview_cameras:
            ref_video_path = os.path.join(video_output_dir, f"{cam_name}_ref.mp4")
            
            # 简单检查跳过
            if os.path.exists(ref_video_path): continue
            if cam_name not in annotation['extrinsic_matrix']: continue

            extrinsic_matrices = np.array(annotation['extrinsic_matrix'][cam_name])
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(ref_video_path, fourcc, 20.0, (camera_width, camera_height))

            for frame_idx in range(len(extrinsic_matrices)):
                # 坐标变换
                extrinsic_matrix = extrinsic_matrices[frame_idx]
                extrinsic_matrix_world = T_base_inv @ extrinsic_matrix
                
                # 设置相机
                camera_pos, camera_rot = invert_camera_extrinsic_matrix(extrinsic_matrix_world)
                set_camera_pose_world(env.sim, cam_id, camera_pos, camera_rot)

                # 渲染
                frame = env.sim.render(
                    camera_name=available_cam_name,
                    width=camera_width,
                    height=camera_height,
                    depth=False
                )
                
                # 保存
                bgr_frame = cv2.cvtColor(frame[::-1], cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            out.release()
            generated_count += 1

        env.close()
        return None

    except Exception as e:
        if env: env.close()
        # 打印详细堆栈
        tb = traceback.format_exc()
        return f"CRITICAL ERROR Ep {episode_id}: {str(e)}\n{tb}"

def main():
    parser = argparse.ArgumentParser(description="生成静态场景的参考视频")
    parser.add_argument("--dataset-dir", type=str,
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+8_fov60",
                       help="数据集根目录")
    parser.add_argument("--robocasa-base", type=str,
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp",
                       help="RoboCasa数据集基础目录")
    parser.add_argument("--task-name", type=str, default="PnPCounterToMicrowave",
                       help="指定任务名称（如果为None则处理所有任务）")
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--max-workers", type=int, default=4,
                       help="并行处理的最大进程数") # 建议不要超过CPU核心数或GPU显存限制
    args = parser.parse_args()

    # 初始化管理器和锁
    manager = mp.Manager()
    io_lock = manager.Lock()    # 用于 HDF5 读取
    reset_lock = manager.Lock() # 用于 Environment Reset (最关键的锁)

    # 获取任务列表
    if args.task_name:
        tasks = [args.task_name]
    else:
        tasks = [d.name for d in Path(args.dataset_dir).iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"开始处理 {len(tasks)} 个任务 (使用 {args.max_workers} 进程)")

    for task in tasks:
        print(f"\nProcessing Task: {task}")
        
        hdf5_path = find_hdf5_file(task, args.robocasa_base)
        if not hdf5_path:
            print(f"Skipping {task}: No HDF5 found")
            continue

        task_anno_dir = os.path.join(args.dataset_dir, task, "annotation")
        if not os.path.exists(task_anno_dir): continue

        # 收集所有 episodes
        work_items = []
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(task_anno_dir, split)
            if not os.path.exists(split_dir): continue
            
            for f in os.listdir(split_dir):
                if f.endswith(".json"):
                    ep_id = int(f.replace(".json", ""))
                    work_items.append((task, ep_id, split, os.path.join(split_dir, f)))
        
        work_items.sort(key=lambda x: x[1])
        print(f"Found {len(work_items)} episodes.")

        # 并行处理
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for item in work_items:
                futures.append(executor.submit(
                    process_episode_ref_videos,
                    item[0], item[1], item[2], hdf5_path, item[3],
                    args.dataset_dir, args.camera_width, args.camera_height,
                    io_lock, reset_lock
                ))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(work_items)):
                res = future.result()
                if res:
                    tqdm.write(res) # 打印错误信息

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()