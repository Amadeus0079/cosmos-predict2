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
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIVIEW_LONG16,
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIVIEW_CONCAT,
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIVIEW_GRIPPER
)
from cosmos_predict2.data.action_conditioned.novelview_dataset import NovelViewDataset
from cosmos_predict2.configs.action_conditioned.defaults.data import (
    robocasa_novelview_val_dataset,
    robocasa_cheatview_val_dataset
)
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
    parser = argparse.ArgumentParser(description="Video-to-World Generation with Cosmos Predict2 (Novel View)")
    parser.add_argument(
        "--model_type",
        choices=["concat", "replace", "long16", "gripper"],
        default="concat",
        help="Type of the model to use for video-to-world generation",
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
        "--num_sampling_step",
        type=int,
        default=35,
        help="Number of steps for the pipeline to sample",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        help="Chunk size",
    )
    parser.add_argument("--autoregressive", action="store_true", help="Use autoregressive mode")
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/novelview/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference (should be a divisor of the total frames)",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=200,
        help="Interval between samples to process",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Maximum number of samples to process",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace):
    if args.model_type == "concat":
        config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIVIEW_CONCAT
    elif args.model_type == "replace":
        config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED
    elif args.model_type == 'long16':
        config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIVIEW_LONG16
    elif args.model_type == 'gripper':
        config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIVIEW_GRIPPER
    dit_path = "checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned/model-480p-4fps.pt"

    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path

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
    pipe, input_video, input_actions, input_condition, output_path, guidance, seed, chunk_size, autoregressive, num_sampling_step
):
    actions = input_actions.cpu().detach().numpy()
    frame_num = input_video.shape[1]
    print(f"input_video: {input_video.shape}")
    print(f"ori actions: {actions.shape}")
    print(f"actions: {actions[:chunk_size].shape}")
    print(f"input_condition: {input_condition.shape}")

    video = pipe(
        input_video,
        actions[:chunk_size]*0,
        input_condition,
        num_conditional_frames=frame_num,
        guidance=guidance,
        seed=seed,
        num_sampling_step=num_sampling_step,
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


def generate_video(args: argparse.Namespace, pipe: Video2WorldMultiviewPipeline, val_dataset: NovelViewDataset) -> None:
    """Generate videos for novel view synthesis validation.

    NovelViewDataset generates 4x samples (one per randomview), so we iterate through
    all samples and use the randomview_name to organize outputs.
    """
    dit_ckpt = os.path.basename(args.dit_path)
    dit_name = os.path.splitext(dit_ckpt)[0]

    # Calculate total samples considering randomview multiplier
    total_samples = len(val_dataset)
    num_randomviews = val_dataset.num_randomviews

    log.info(f"Total samples in dataset: {total_samples}")
    log.info(f"Number of randomviews per sequence: {num_randomviews}")
    log.info(f"Processing samples with interval: {args.sample_interval}")

    for j in range(0, min(args.max_samples, total_samples), args.sample_interval):
        for i in range(j, j + 4):
            batch_data = val_dataset[i]

            # Get randomview information from the data
            randomview_id = batch_data['randomview_id']
            randomview_name = batch_data['randomview_name']

            # Construct output path with randomview info
            output_path = f"output/cheatview/{args.model_type}_{dit_name}_sample{args.num_sampling_step}_{randomview_name}_idx{i}_rv{randomview_id}.mp4"

            log.info(f"Processing sample {i}/{total_samples}, randomview: {randomview_name} (id: {randomview_id})")

            process_single_generation(
                pipe=pipe,
                input_video=batch_data['video'],
                input_actions=batch_data['action'],
                input_condition=batch_data['pred_video'],
                output_path=output_path,
                guidance=args.guidance,
                seed=args.seed,
                chunk_size=args.chunk_size,
                autoregressive=args.autoregressive,
                num_sampling_step=args.num_sampling_step,
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
    # val_dataset = instantiate(robocasa_novelview_val_dataset)
    val_dataset = instantiate(robocasa_cheatview_val_dataset)

    log.info(f"Loaded CheatViewDataset with {len(val_dataset)} samples")
    log.info(f"Base sequences: {len(val_dataset.samples)}")
    log.info(f"Randomviews per sequence: {val_dataset.num_randomviews}")

    try:
        pipe = setup_pipeline(args)
        generate_video(args, pipe, val_dataset)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
