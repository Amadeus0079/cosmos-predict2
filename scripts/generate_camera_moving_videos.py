#!/usr/bin/env python3
"""
生成静态场景的参考视频
保持场景（initial state）不变，使用原视频中的相机参数逐帧渲染
"""

# #!/usr/bin/env python3
# import os
# import sys
# import shutil
# import tempfile
# import atexit

# # ==============================================================================
# # 【核心修复】进程隔离层
# # 在导入任何 robosuite/mujoco 库之前，强制隔离临时目录
# # ==============================================================================

# # 1. 创建一个专属的临时目录，例如 /tmp/robocasa_proc_12345
# # pid 是进程ID，绝对唯一
# unique_tmp_dir = tempfile.mkdtemp(prefix=f"robocasa_proc_{os.getpid()}_")

# print(f"🔒 [Process {os.getpid()}] file system isolation activated.")
# print(f"📂 Temp dir: {unique_tmp_dir}")

# # 2. 强制修改所有可能涉及临时文件的环境变量
# # Python 的 tempfile 模块和很多 C++ 库都会遵循这些变量
# os.environ["TMPDIR"] = unique_tmp_dir
# os.environ["TEMP"] = unique_tmp_dir
# os.environ["TMP"] = unique_tmp_dir

# # 3. 专门针对 MuJoCo/Robosuite 可能的缓存路径进行隔离
# # 如果库在当前目录写文件，我们把工作目录也切过去（可选，视库的具体行为而定）
# # os.chdir(unique_tmp_dir) # 慎用，可能会导致找不到相对路径的 dataset

# # 4. 注册退出函数，程序结束时自动清理垃圾
# def cleanup_temp_dir():
#     try:
#         if os.path.exists(unique_tmp_dir):
#             shutil.rmtree(unique_tmp_dir)
#             # print(f"🧹 [Process {os.getpid()}] Cleaned up temp dir.")
#     except Exception as e:
#         print(f"⚠️ Cleanup failed: {e}")

# atexit.register(cleanup_temp_dir)

# # ==============================================================================
# # 隔离完成，设置渲染后端
# # ==============================================================================
# os.environ["MUJOCO_GL"] = "egl"

import h5py
import numpy as np
import cv2
import os
import json
import argparse
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Import robocasa dependencies
try:
    import robocasa
    from robocasa.utils.env_utils import create_env
    import robosuite
    import robosuite.utils.camera_utils as camera_utils
except ImportError as e:
    raise ImportError(
        "RoboCasa not found. Please install RoboCasa in your environment."
    ) from e


# --- Helper Functions ---

def _nearest_rotation(R_mat: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the nearest proper rotation (SO(3)) using SVD."""
    U, S, Vt = np.linalg.svd(R_mat)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def invert_camera_extrinsic_matrix(R_extrinsic: np.ndarray):
    """Invert the camera extrinsic matrix to get camera position and rotation."""
    R_extrinsic = np.asarray(R_extrinsic, dtype=float)
    if R_extrinsic.shape != (4, 4):
        raise ValueError(f"R_extrinsic must be 4x4, got shape {R_extrinsic.shape}")

    C = np.diag([1.0, -1.0, -1.0, 1.0])
    pose = R_extrinsic @ C
    camera_rot_raw = pose[:3, :3]
    camera_pos = pose[:3, 3].copy()
    camera_rot = _nearest_rotation(camera_rot_raw)
    return camera_pos, camera_rot


def set_camera_pose_world(sim, cam_id, pos_world, rot_world_3x3):
    """Set camera pose in MuJoCo simulation."""
    quat_wxyz = R.from_matrix(rot_world_3x3).as_quat()  # returns (x, y, z, w)
    quat_wxyz = np.roll(quat_wxyz, 1)  # Convert to (w, x, y, z)
    sim.model.cam_pos[cam_id] = np.asarray(pos_world, dtype=np.float64)
    sim.model.cam_quat[cam_id] = np.asarray(quat_wxyz, dtype=np.float64)
    sim.forward()


def reset_to(env, state):
    """Reset environment to a specific simulator state."""
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}

        # 关键修复：处理 gen_textures
        # 问题：HDF5 中的 ep_meta.gen_textures 可能是空字典 {}
        # 解决方案：如果 gen_textures 为空，我们需要确保使用一致的纹理
        if ep_meta.get("gen_textures") is None or ep_meta.get("gen_textures") == {}:
            # 检查环境是否已经生成了纹理（来自 create_env 时的初始 reset）
            if hasattr(env, "_curr_gen_fixtures") and env._curr_gen_fixtures:
                # 使用环境已生成的纹理（第一次 reset 时生成）
                ep_meta["gen_textures"] = env._curr_gen_fixtures
            else:
                # 如果环境还没有生成纹理，设置一个固定的随机种子
                # 这样每次调用 edit_model_xml 时都会生成相同的纹理
                # 使用 episode 相关的信息作为种子（如果可用）
                import hashlib
                seed_str = json.dumps(ep_meta.get("layout_id", 0)) + json.dumps(ep_meta.get("style_id", 0))
                seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)

                # 创建一个带固定种子的 RNG 并生成纹理
                from robocasa.utils.texture_swap import get_random_textures
                rng = np.random.RandomState(seed_hash)
                ep_meta["gen_textures"] = get_random_textures(rng)

        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)

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
    if hasattr(env, "update_state"):
        env.update_state()
    return None


def find_hdf5_file(task_name, base_dir):
    """Find HDF5 file for a given task."""
    task_dir = os.path.join(base_dir, task_name)
    for root, dirs, files in os.walk(task_dir):
        for file in files:
            if file == "demo_im256_ep100_cam2+8_fov75.hdf5":
                return os.path.join(root, file)
    return None


def get_demo_key_from_episode_id(hdf5_path, episode_id):
    """
    从HDF5文件中获取对应episode_id的demo_key
    episode_id就是排序后demo列表的索引
    """
    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            return None

        data_group = f['data']
        episodes = sorted([key for key in data_group.keys() if key.startswith('demo_')])

        if episode_id < 0 or episode_id >= len(episodes):
            return None

        return episodes[episode_id]


def process_episode_ref_videos(task_name, episode_id, split, hdf5_path, annotation_path,
                                output_dir, camera_width=256, camera_height=256):
    """
    为单个episode生成静态场景的参考视频
    使用annotation中的相机参数
    """
    try:
        # Read annotation to get camera list and extrinsic matrices
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # if episode_id != 19:
        #     return
        
        # Get demo_key from episode_id
        demo_key = get_demo_key_from_episode_id(hdf5_path, episode_id)
        if demo_key is None:
            return f"警告: episode_id {episode_id} 未找到对应的demo"

        # Load initial state from HDF5
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f or demo_key not in f['data']:
                return f"错误: demo_key {demo_key} 不存在于HDF5文件中"

            demo_data = f['data'][demo_key]
            states = f[f"data/{demo_key}/states"][()]

            initial_state = dict(states=states[0])
            initial_state["model"] = demo_data.attrs["model_file"]
            initial_state["ep_meta"] = demo_data.attrs.get("ep_meta", None)

        # Create environment
        # 重要: generative_textures 必须在创建环境时设置，而不是在 reset_to 中设置
        # 这样才能确保 _load_model() 时 self._curr_gen_fixtures 从 ep_meta 中正确获取 gen_textures
        env = create_env(
            env_name=task_name,
            robots="PandaMobile",
            camera_names=["robot0_randomview_0", "robot0_agentview_center"],
            camera_widths=camera_width,
            camera_heights=camera_height,
            generative_textures="100p",  # 关键: 在 create_env 时设置
            seed=42,
            render_onscreen=False,
        )

        # Reset to initial state
        reset_to(env, initial_state)
        # env.reset()

        # Find a camera to use for rendering
        available_cam_name = None
        for test_cam in ["robot0_randomview_0", "robot0_agentview_center", "robot0_eye_in_hand"]:
            try:
                cam_id = env.sim.model.camera_name2id(test_cam)
                available_cam_name = test_cam
                break
            except:
                continue

        if available_cam_name is None:
            return f"错误: 未找到可用的相机进行渲染"

        # Process each randomview camera that exists in the annotation
        video_output_dir = os.path.join(output_dir, task_name, "videos", split, str(episode_id))
        os.makedirs(video_output_dir, exist_ok=True)

        generated_count = 0
        skipped_count = 0

        # Get all randomview cameras from annotation
        randomview_cameras = [cam_name for cam_name in annotation['videos'].keys()
                             if cam_name.startswith('robot0_randomview_')]

        for cam_name in randomview_cameras:
            # Check if ref video already exists
            ref_video_path = os.path.join(video_output_dir, f"{cam_name}_ref.mp4")
            ref_depth_path = os.path.join(video_output_dir, f"{cam_name}_ref.npy")

            # if os.path.exists(ref_video_path) and os.path.exists(ref_depth_path):
            if os.path.exists(ref_video_path):
                skipped_count += 1
                continue

            # Get extrinsic matrices for this camera
            if cam_name not in annotation['extrinsic_matrix']:
                return f"错误: {cam_name} 的extrinsic_matrix不存在于annotation中"

            extrinsic_matrices = np.array(annotation['extrinsic_matrix'][cam_name])
            num_frames = len(extrinsic_matrices)

            # Use the available camera for rendering
            cam_id = env.sim.model.camera_name2id(available_cam_name)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_fps = 20.0
            out = cv2.VideoWriter(ref_video_path, fourcc, output_fps, (camera_width, camera_height))

            # depth_frames = []
            
            T_base = np.eye(4)
            base_pos = np.array(annotation['base_pos'])
            base_quat = np.array(annotation['base_quat'])
            base_mat = R.from_quat(base_quat).as_matrix()
            T_base[:3, 3] = base_pos
            T_base[:3, :3] = base_mat

            # Render each frame with the camera pose from extrinsic matrix
            for frame_idx in range(num_frames):
                extrinsic_matrix = extrinsic_matrices[frame_idx]
                extrinsic_matrix = np.linalg.inv(T_base) @ extrinsic_matrix
                camera_pos, camera_rot = invert_camera_extrinsic_matrix(extrinsic_matrix)
                set_camera_pose_world(env.sim, cam_id, camera_pos, camera_rot)

                # Render frame
                frame = env.sim.render(
                    camera_name=available_cam_name,
                    width=camera_width,
                    height=camera_height,
                    depth=False
                )

                frame = frame[::-1]

                # Get depth
                # depth = env.sim.render(
                #     camera_name=available_cam_name,
                #     width=camera_width,
                #     height=camera_height,
                #     depth=True
                # )

                # Convert to BGR for video writing
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
                # depth_frames.append(depth)

            out.release()

            # Save depth
            # depth_array = np.stack(depth_frames, axis=0)
            # np.save(ref_depth_path, depth_array)

            generated_count += 1

        if generated_count > 0 or skipped_count > 0:
            return f"Episode {episode_id}: 生成 {generated_count} 个ref视频, 跳过 {skipped_count} 个已存在ref视频"
        return None

    except Exception as e:
        import traceback
        return f"错误处理 episode {episode_id}: {str(e)}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="生成静态场景的参考视频")
    parser.add_argument("--dataset-dir", type=str,
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_im256_ep100_cam2+8_fov75",
                       help="数据集根目录")
    parser.add_argument("--robocasa-base", type=str,
                       default="/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/robocasa/datasets_hdd/v0.1/single_stage/kitchen_pnp",
                       help="RoboCasa数据集基础目录")
    parser.add_argument("--task-name", type=str, default="PnPCounterToMicrowave",
                       help="指定任务名称（如果为None则处理所有任务）")
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--max-workers", type=int, default=1,
                       help="并行处理的最大线程数")

    args = parser.parse_args()

    # Find all tasks
    if args.task_name:
        tasks = [args.task_name]
    else:
        dataset_path = Path(args.dataset_dir)
        tasks = [d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    print(f"开始处理 {len(tasks)} 个任务")

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"处理任务: {task}")
        print(f"{'='*60}")

        # Find HDF5 file
        hdf5_path = find_hdf5_file(task, args.robocasa_base)
        if not hdf5_path:
            print(f"警告: 未找到任务 {task} 的HDF5文件，跳过")
            continue

        print(f"HDF5文件: {hdf5_path}")

        # Find all annotation files for this task
        task_annotation_dir = os.path.join(args.dataset_dir, task, "annotation")
        if not os.path.exists(task_annotation_dir):
            print(f"警告: 未找到任务 {task} 的annotation目录，跳过")
            continue

        episodes_to_process = []
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(task_annotation_dir, split)
            if not os.path.exists(split_dir):
                continue

            for json_file in os.listdir(split_dir):
                if json_file.endswith(".json"):
                    episode_id = int(json_file.replace(".json", ""))
                    annotation_path = os.path.join(split_dir, json_file)
                    episodes_to_process.append((task, episode_id, split, annotation_path))

        print(f"找到 {len(episodes_to_process)} 个episodes需要处理")

        # Process episodes
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    process_episode_ref_videos,
                    task, episode_id, split, hdf5_path, annotation_path,
                    args.dataset_dir, args.camera_width, args.camera_height
                ) for task, episode_id, split, annotation_path in episodes_to_process
            ]

            for future in tqdm(concurrent.futures.as_completed(futures),
                             total=len(episodes_to_process),
                             desc=f"处理{task}"):
                result = future.result()
                if result:
                    print(f"\n{result}")

    print("\n所有任务处理完成！")


if __name__ == "__main__":
    main()