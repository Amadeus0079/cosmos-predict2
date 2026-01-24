#!/usr/bin/env python3
"""
Convert LeRobot dataset from openpi/datasets/real_merge/PanAll to cosmos-predict2 training format.

This script:
1. Reads GT camera data (head, side) from LeRobot dataset
2. Reads rendered camera data (8 random views) from collection_ref
3. Converts to cosmos-predict2 format with train/test/val split
4. Saves videos, depth, reference images, and annotations
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm
import concurrent.futures
import shutil

# Add parent directory to path to import LeRobotDataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "openpi"))
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Default paths
DEFAULT_LEROOT_PATH = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/openpi/datasets/real_merge/PanAll"
DEFAULT_OUTPUT_DIR = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets"
DEFAULT_DOMAIN_NAME = "lerobot_panall"

# GT camera names from LeRobot
GT_CAMERAS = ["wrist", "side"]

# Rendered camera names (8 cameras)
RENDERED_CAMERAS = [f"randomview_{i}" for i in range(8)]

# All camera names
ALL_CAMERAS = GT_CAMERAS + RENDERED_CAMERAS

# FPS for output videos
OUTPUT_FPS = 30


def extract_ref_images_as_png(ref_video_path: str, output_dir: str) -> int:
    """
    Extract frames from ref_image.mp4 and save as PNG files.

    Args:
        ref_video_path: Path to the reference video file
        output_dir: Directory to save PNG frames

    Returns:
        Number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(ref_video_path)

    if not cap.isOpened():
        print(f"Warning: Could not open {ref_video_path}")
        return 0

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame as PNG with 6-digit zero-padded numbering
        output_path = os.path.join(output_dir, f"{frame_idx:06d}.png")
        cv2.imwrite(output_path, frame)
        frame_idx += 1

    cap.release()
    return frame_idx - 1


def split_episodes(episodes: List[int], train_ratio: float, test_ratio: float, val_ratio: float,
                   random_seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Split episode indices into train, test, and val sets.

    Args:
        episodes: List of episode indices
        train_ratio: Ratio for training set
        test_ratio: Ratio for test set
        val_ratio: Ratio for validation set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_episodes, test_episodes, val_episodes)
    """
    total = train_ratio + test_ratio + val_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    random.seed(random_seed)
    shuffled = episodes.copy()
    random.shuffle(shuffled)

    total_count = len(shuffled)
    train_count = int(total_count * train_ratio)
    test_count = int(total_count * test_ratio)

    train_episodes = shuffled[:train_count]
    test_episodes = shuffled[train_count:train_count + test_count]
    val_episodes = shuffled[train_count + test_count:]

    return train_episodes, test_episodes, val_episodes


def process_gt_cameras(dataset: LeRobotDataset, episode_idx: int, start_idx: int, end_idx: int,
                       output_video_dir: str, cam_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process GT camera data from LeRobot dataset.

    Args:
        dataset: LeRobotDataset instance
        episode_idx: Episode index
        start_idx: Start frame index
        end_idx: End frame index
        output_video_dir: Directory to save video
        cam_name: Camera name (head or side)

    Returns:
        Tuple of (depth_array, intrinsics, extrinsics)
    """
    video_path = os.path.join(output_video_dir, f"{cam_name}.mp4")

    # Get frame dimensions
    first_frame = dataset[start_idx]
    rgb = first_frame[f"observation.images.{cam_name}"]
    height, width = rgb.shape[1], rgb.shape[2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, OUTPUT_FPS, (width, height))

    # Collect depth, intrinsics, extrinsics
    depth_list = []
    intrinsics_list = []
    extrinsics_list = []

    # Process each frame
    for frame_idx in range(start_idx, end_idx):
        data = dataset[frame_idx]

        # Get RGB image - LeRobot stores as CHW, convert to HWC
        rgb = data[f"observation.images.{cam_name}"]
        # Convert to numpy if tensor
        if hasattr(rgb, 'cpu'):
            rgb = rgb.cpu().numpy()
        if rgb.ndim == 3 and rgb.shape[0] == 3:  # CHW format
            rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (rgb * 255).astype(np.uint8)

        # Get depth - LeRobot stores as float32 in meters
        depth = data[f"observation.depth.{cam_name}"]
        if hasattr(depth, 'cpu'):
            depth = depth.cpu().numpy() / 1000.0
        if depth.ndim == 3:
            depth = depth.squeeze()
        depth = depth.astype(np.float32)
        depth_list.append(depth)

        # Get intrinsics - reshape from (9,) to (3, 3)
        intr = data[f"observation.intrinsics.{cam_name}"]
        if hasattr(intr, 'cpu'):
            intr = intr.cpu().numpy()
        intr = intr.reshape(3, 3)
        intrinsics_list.append(intr)

        # Get extrinsics - reshape from (16,) to (4, 4) and extract 3x4
        extr = data[f"observation.extrinsics.{cam_name}"]
        if hasattr(extr, 'cpu'):
            extr = extr.cpu().numpy()
        extr = extr.reshape(4, 4)
        # Extract 3x4 portion (first 3 rows)
        # extr_3x4 = extr[:3, :]
        extrinsics_list.append(extr)

        # Write RGB frame (convert RGB to BGR for OpenCV)
        bgr_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()

    # Save depth as .npy
    depth_array = np.stack(depth_list, axis=0)
    depth_path = os.path.join(output_video_dir, f"{cam_name}.npy")
    np.save(depth_path, depth_array)

    # For intrinsics and extrinsics, use the first frame (they should be constant)
    intrinsics = intrinsics_list[0] if intrinsics_list else np.eye(3)
    extrinsics = np.stack(extrinsics_list, axis=0) if extrinsics_list else np.zeros((0, 3, 4))

    return depth_array, intrinsics, extrinsics


def process_rendered_cameras(collection_ref_root: str, episode_idx: int, output_video_dir: str) -> Dict[str, Dict]:
    """
    Process rendered camera data from collection_ref.

    Args:
        collection_ref_root: Path to collection_ref directory
        episode_idx: Episode index
        output_video_dir: Directory to save videos

    Returns:
        Dictionary mapping camera names to (depth, intrinsics, extrinsics)
    """
    episode_name = f"episode_{episode_idx:06d}"
    episode_dir = os.path.join(collection_ref_root, episode_name)

    result = {}

    for i in range(8):
        cam_dir = os.path.join(episode_dir, f"cam{i}")
        cam_name = f"randomview_{i}"

        if not os.path.exists(cam_dir):
            print(f"Warning: Camera directory not found: {cam_dir}")
            continue

        # Paths in collection_ref
        images_mp4 = os.path.join(cam_dir, "images.mp4")
        ref_mp4 = os.path.join(cam_dir, "ref_image.mp4")
        data_npz = os.path.join(cam_dir, "data.npz")

        if not all(os.path.exists(p) for p in [images_mp4, ref_mp4, data_npz]):
            print(f"Warning: Missing files in {cam_dir}")
            continue

        # Copy images.mp4 to output
        output_mp4 = os.path.join(output_video_dir, f"{cam_name}.mp4")
        shutil.copy(images_mp4, output_mp4)

        # Extract reference images to PNG
        ref_dir = os.path.join(output_video_dir, f"{cam_name}_ref")
        num_ref_frames = extract_ref_images_as_png(ref_mp4, ref_dir)

        # Load data.npz
        data = np.load(data_npz)
        depth_int16 = data['depth']  # Shape: (num_frames, 480, 640), int16 in mm
        extrinsics = data['extrinsics']  # Shape: (num_frames, 4, 4)
        intrinsics = data['intrinsics']  # Shape: (3, 3)

        # Convert depth from int16 (mm) to float32 (meters)
        depth_float32 = depth_int16.astype(np.float32) / 1000.0

        # Save depth as .npy
        depth_path = os.path.join(output_video_dir, f"{cam_name}.npy")
        np.save(depth_path, depth_float32)

        # Extract 3x4 portion from extrinsics
        # extrinsics_3x4 = extrinsics[:, :3, :]  # Shape: (num_frames, 3, 4)

        result[cam_name] = {
            'depth': depth_float32,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'num_ref_frames': num_ref_frames
        }

    return result


def process_episode(episode_idx: int, dataset: LeRobotDataset, collection_ref_root: str,
                   output_videos_dir: str, output_annotations_dir: str, split: str,
                   new_episode_id: int) -> Optional[str]:
    """
    Process a single episode.

    Args:
        episode_idx: Original episode index in LeRobot dataset
        dataset: LeRobotDataset instance
        collection_ref_root: Path to collection_ref directory
        output_videos_dir: Output videos directory
        output_annotations_dir: Output annotations directory
        split: Dataset split (train/test/val)
        new_episode_id: New episode ID for output

    Returns:
        Error message if failed, None otherwise
    """
    try:
        # Get frame indices for this episode
        start_idx = int(dataset.episode_data_index['from'][episode_idx])
        end_idx = int(dataset.episode_data_index['to'][episode_idx])

        # Create output directory
        episode_output_dir = os.path.join(output_videos_dir, split, str(new_episode_id))
        os.makedirs(episode_output_dir, exist_ok=True)

        # Collect state and action data
        states = []
        actions = []

        for frame_idx in range(start_idx, end_idx):
            data = dataset[frame_idx]

            # Get state and action (raw format, shape [8])
            state = data['observation.state'].cpu().numpy() if hasattr(data['observation.state'], 'cpu') else data['observation.state']
            action = data['action'].cpu().numpy() if hasattr(data['action'], 'cpu') else data['action']

            states.append(state.tolist())
            actions.append(action.tolist())

        # Process GT cameras (head, side)
        camera_data = {}

        for cam_name in GT_CAMERAS:
            depth, intrinsics, extrinsics = process_gt_cameras(
                dataset, episode_idx, start_idx, end_idx, episode_output_dir, cam_name
            )
            camera_data[cam_name] = {
                'depth': depth,
                'intrinsics': intrinsics,
                'extrinsics': extrinsics
            }

        # Process rendered cameras (randomview_0-7)
        rendered_data = process_rendered_cameras(
            collection_ref_root, episode_idx, episode_output_dir
        )

        for cam_name, data in rendered_data.items():
            camera_data[cam_name] = data

        # Create JSON annotation
        video_paths = {}
        for cam_name in ALL_CAMERAS:
            if cam_name in camera_data:
                video_paths[cam_name] = {
                    'video_path': f"videos/{split}/{new_episode_id}/{cam_name}.mp4",
                    'depth_path': f"videos/{split}/{new_episode_id}/{cam_name}.npy"
                }

        # Build extrinsic_matrix and intrinsic_matrix
        extrinsic_matrix = {}
        intrinsic_matrix = {}

        for cam_name in ALL_CAMERAS:
            if cam_name in camera_data:
                intrinsic_matrix[cam_name] = camera_data[cam_name]['intrinsics'].tolist()

                # For GT cameras, use first extrinsic; for rendered, use all
                if cam_name in GT_CAMERAS:
                    extrinsic_matrix[cam_name] = camera_data[cam_name]['extrinsics'].tolist()
                else:
                    extrinsic_matrix[cam_name] = camera_data[cam_name]['extrinsics'].tolist()

        annotation = {
            'episode_id': str(new_episode_id),
            'task': 'robot_trajectory_prediction',
            'texts': [''],
            'videos': video_paths,
            'action': actions,
            'state': states,
            'continuous_gripper_state': [],
            'extrinsic_matrix': extrinsic_matrix,
            'intrinsic_matrix': intrinsic_matrix,
            'base_pos': [0, 0, 0],
            'base_quat': [0, 0, 0, 1]
        }

        # Save annotation
        annotation_path = os.path.join(output_annotations_dir, split, f"{new_episode_id}.json")
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        with open(annotation_path, 'w') as f:
            json.dump(annotation, f, indent=4)

        return None

    except Exception as e:
        import traceback
        return f"Error processing episode {episode_idx}: {e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(
        description='Convert LeRobot dataset to cosmos-predict2 training format'
    )
    parser.add_argument(
        '--leroot-path',
        type=str,
        default=DEFAULT_LEROOT_PATH,
        help='Path to LeRobot dataset root directory'
    )
    parser.add_argument(
        '--collection-ref-path',
        type=str,
        default=None,
        help='Path to collection_ref directory (default: {leroot-path}/collection_ref)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for converted dataset'
    )
    parser.add_argument(
        '--domain-name',
        type=str,
        default=DEFAULT_DOMAIN_NAME,
        help='Domain name for the output dataset'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio of training set'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.0,
        help='Ratio of test set'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Ratio of validation set'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for splitting'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Maximum number of worker threads'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        nargs='+',
        default=None,
        help='Episode indices to process (default: all)'
    )

    args = parser.parse_args()

    # Set collection_ref path
    if args.collection_ref_path is None:
        collection_ref_path = os.path.join(args.leroot_path, 'collection_ref')
    else:
        collection_ref_path = args.collection_ref_path

    # Verify input paths exist
    if not os.path.exists(args.leroot_path):
        print(f"Error: LeRobot dataset path not found: {args.leroot_path}")
        return

    if not os.path.exists(collection_ref_path):
        print(f"Error: collection_ref path not found: {collection_ref_path}")
        return

    print(f"Converting LeRobot dataset from: {args.leroot_path}")
    print(f"collection_ref path: {collection_ref_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Domain name: {args.domain_name}")
    print(f"Split ratios - train: {args.train_ratio}, test: {args.test_ratio}, val: {args.val_ratio}")

    # Load LeRobot dataset to get total episodes
    dataset = LeRobotDataset(
        repo_id="real_merge/test",
        root=args.leroot_path,
        force_cache_sync=False
    )

    total_episodes = 90
    print(f"Total episodes in dataset: {total_episodes}")

    # Determine which episodes to process
    if args.episodes is None:
        all_episodes = list(range(total_episodes))
    else:
        all_episodes = args.episodes

    print(f"Processing {len(all_episodes)} episodes")

    # Split episodes
    train_episodes, test_episodes, val_episodes = split_episodes(
        all_episodes, args.train_ratio, args.test_ratio, args.val_ratio, args.random_seed
    )

    print(f"Split - train: {len(train_episodes)}, test: {len(test_episodes)}, val: {len(val_episodes)}")

    # Create output directories
    domain_output_dir = os.path.join(args.output_dir, args.domain_name)
    videos_dir = os.path.join(domain_output_dir, "videos")
    annotations_dir = os.path.join(domain_output_dir, "annotation")

    for split in ['train', 'test', 'val']:
        os.makedirs(os.path.join(videos_dir, split), exist_ok=True)
        os.makedirs(os.path.join(annotations_dir, split), exist_ok=True)

    # Process all splits
    all_splits = [
        ('train', train_episodes),
        ('test', test_episodes),
        ('val', val_episodes)
    ]

    # Assign new episode IDs
    new_episode_id = 0
    episode_id_mapping = {}

    for split_name, episodes in all_splits:
        for ep_idx in episodes:
            episode_id_mapping[ep_idx] = (split_name, new_episode_id)
            new_episode_id += 1

    # Process episodes with multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []

        for split_name, episodes in all_splits:
            for ep_idx in episodes:
                _, new_id = episode_id_mapping[ep_idx]
                future = executor.submit(
                    process_episode,
                    ep_idx, dataset, collection_ref_path,
                    videos_dir, annotations_dir, split_name, new_id
                )
                futures.append((ep_idx, future))

        # Wait for completion and show progress
        for future in tqdm(concurrent.futures.as_completed([f for _, f in futures]),
                            total=len(futures), desc="Processing episodes"):
            result = future.result()
            if result:
                print(f"\n{result}")

    print("\nConversion completed!")
    print(f"Output saved to: {domain_output_dir}")


if __name__ == '__main__':
    main()
