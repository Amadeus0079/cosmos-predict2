# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import pdb

import mediapy as mp
import numpy as np

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state

from cosmos_predict2.configs.action_conditioned.config import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED,
)
from cosmos_predict2.configs.action_conditioned.config_concat import PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED_CONCAT
from cosmos_predict2.data.action_conditioned.multiview_dataset import MultiViewDataset
from cosmos_predict2.configs.action_conditioned.defaults.data import robocasa_val_dataset
from cosmos_predict2.pipelines.video2world_multiview import Video2WorldMultiviewPipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video
from imaginaire.lazy_config import instantiate


def get_action_sequence(annotation_path):
    with open(annotation_path, "r") as file:
        data = json.load(file)

    # rescale the action to the original scale
    action_ee = np.array(data["action"])[:, :6] * 20
    gripper = np.array(data["continuous_gripper_state"])[1:,None]

    # concatenate the end-effector displacement and gripper width
    action = np.concatenate([action_ee, gripper], axis=1)
    return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=["2B"],
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--condition_strategy",
        choices=["concat", "replace"],
        default="concat",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Path to input image or video for conditioning (include file extension)",
    )
    parser.add_argument(
        "--input_annotation",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Path to input image or video for conditioning (include file extension)",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        choices=[1],
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=12,
        help="Chunk size",
    )
    parser.add_argument("--autoregressive", action="store_true", help="Use autoregressive mode")
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference (should be a divisor of the total frames)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace):
    log.info(f"Using model size: {args.model_size}")
    if args.model_size == "2B":
        if args.condition_strategy == "concat":
            config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED_CONCAT
        elif args.condition_strategy == "replace":
            config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED
        dit_path = "checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned/model-480p-4fps.pt"
    else:
        raise ValueError("Invalid model size. Choose either '2B' or '14B'.")
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path

    # text_encoder_path = "checkpoints/google-t5/t5-11b"
    text_encoder_path = ""

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False

    # Load models
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    pipe = Video2WorldMultiviewPipeline.from_config(
        config=config,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_prompt_refiner=True,
        is_train=False,
    )

    return pipe


def read_first_frame(video_path):
    video = mp.read_video(video_path)  # Returns (T, H, W, C) numpy array
    return video[0]  # Return first frame as numpy array


def process_single_generation(
    pipe, input_video, input_actions, input_condition, output_path, guidance, seed, chunk_size, autoregressive
):
    actions = input_actions.cpu().detach().numpy()
    frame_num = input_video.shape[1]
    # input_video[:, 1:, :, :] = 0
    print(f"input_video: {input_video.shape}")
    print(f"action: {actions[:chunk_size].shape}")
    print(f"input_condition: {input_condition.shape}")

    video = pipe(
        input_video,
        actions[:chunk_size],
        input_condition,
        num_conditional_frames=frame_num,
        guidance=guidance,
        seed=seed,
    )
    
    # Visualize the original Image
    normalized_input_video = (input_video.unsqueeze(0).to(video.device) / 255.0) * 2 - 1
    normalized_input_condition = (input_condition.unsqueeze(0).to(video.device) / 255.0) * 2 - 1
    video = torch.cat((normalized_input_video, video, normalized_input_condition), dim=4)

    if video is not None:
        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated video to: {output_path}")
        save_image_or_video(video, output_path, fps=4)
        log.success(f"Successfully saved video to: {output_path}")
        return True
    return False


def generate_video(args: argparse.Namespace, pipe: Video2WorldMultiviewPipeline, batch_data) -> None:
    for i in range(0, 1000, 100):
        batch_data = val_dataset[i]
        dit_ckpt = os.path.basename(args.dit_path)
        dit_name = os.path.splitext(dit_ckpt)[0]
        process_single_generation(
            pipe=pipe,
            input_video=batch_data['video'],
            input_actions=batch_data['action'],
            input_condition=batch_data['pred_video'],
            output_path=f"output/multiview/{args.condition_strategy}_{dit_name}_eyeinhand_{i}.mp4",
            guidance=args.guidance,
            seed=args.seed,
            chunk_size=args.chunk_size,
            autoregressive=args.autoregressive,
        )
    return


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    val_dataset = instantiate(robocasa_val_dataset)
    try:
        pipe = setup_pipeline(args)
        generate_video(args, pipe, val_dataset)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
