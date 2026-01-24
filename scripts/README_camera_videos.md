# 静态场景参考视频生成脚本使用说明

## 功能描述

该脚本用于生成静态场景的参考视频（ref videos）。它会：
1. **保持场景的 initial state 不变**（机器人和物体位置固定在第一帧）
2. **使用原视频中的相机参数**（从 annotation JSON 的 extrinsic_matrix 读取）
3. 逐帧改变相机位置，渲染静态场景
4. 为所有已存在的 randomview 相机生成对应的 ref 视频
5. **自动跳过已存在的 ref 视频，不会覆盖**

## 原视频 vs 参考视频

- **原视频（如 `robot0_randomview_0.mp4`）**：
  - 场景是动态的（机器人在运动，物体在移动）
  - 相机位置也在变化
  - 展示完整的任务执行过程

- **参考视频（如 `robot0_randomview_0_ref.mp4`）**：
  - 场景是静态的（保持 initial state）
  - 相机位置与原视频完全相同
  - 用作参考，展示从不同角度观察静态初始场景

## 工作原理

脚本会：
1. 扫描 `cosmos-predict2/datasets/robocasa_im256_ep100_cam2+4/{task}/annotation/{split}/` 目录
2. 读取 `{episode_id}.json` 文件，获取：
   - 已有的 randomview 相机列表（从 `videos` 字段）
   - 每个相机的 extrinsic_matrix（每一帧的相机位置）
3. 从 HDF5 文件加载对应 demo 的 initial state
4. 使用 initial state 创建静态场景
5. 按照 extrinsic_matrix 中的相机参数逐帧渲染
6. 保存为 `robot0_randomview_X_ref.mp4`

## 文件说明

- `generate_camera_moving_videos.py`: 主脚本，支持批量处理所有任务
- `test_camera_video_single.py`: 测试脚本，用于测试单个 episode

## 使用方法

### 1. 测试单个 episode

首先建议使用测试脚本验证功能：

```bash
cd /inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/scripts
python3 test_camera_video_single.py
```

测试脚本会为 PnPCounterToMicrowave 任务的 episode 79 生成所有 randomview 相机的 ref 视频。

### 2. 批量处理所有任务

确认测试成功后，可以运行主脚本处理所有任务：

```bash
python3 generate_camera_moving_videos.py \
    --camera-width 256 \
    --camera-height 256 \
    --max-workers 1
```

### 3. 处理特定任务

如果只想处理某个特定任务：

```bash
python3 generate_camera_moving_videos.py \
    --task-name PnPCounterToSink \
    --max-workers 1
```

## 参数说明

- `--dataset-dir`: 数据集根目录（默认：cosmos-predict2/datasets/robocasa_im256_ep100_cam2+4）
- `--robocasa-base`: RoboCasa 原始 HDF5 文件的基础目录（默认：robocasa/datasets/v0.1/single_stage/kitchen_pnp）
- `--task-name`: 指定单个任务名称（可选，如果不指定则处理所有任务）
- `--camera-width`: 视频宽度（默认：256）
- `--camera-height`: 视频高度（默认：256）
- `--max-workers`: 并行处理的最大线程数（默认：1，建议先用 1 测试）

## 重要说明

### 1. 只为已存在的相机生成 ref 视频

脚本会自动检测 annotation 中已有的 randomview 相机，只为这些相机生成 ref 视频。
- 如果原数据只有 4 个 randomview 相机（0-3），就只生成 4 个 ref 视频
- 如果有 8 个，就生成 8 个

### 2. 使用原视频的相机参数

每个 ref 视频的相机轨迹与原视频完全相同：
- 帧数相同
- 每一帧的相机位置（extrinsic_matrix）相同
- 只是场景是静态的（initial state）

### 3. Episode ID 到 Demo 的映射

- Episode ID = 排序后 HDF5 demo 列表的索引
- 例如：episode_id 79 = 排序后的第 79 个 demo

### 4. 不会覆盖已存在的视频

脚本会自动检查 ref 视频文件是否已存在，如果存在则跳过，不会覆盖。这样可以安全地重复运行脚本。

### 5. 输出结构

生成的 ref 视频会保存在：
```
{dataset-dir}/{task_name}/videos/{split}/{episode_id}/robot0_randomview_{i}_ref.mp4
{dataset-dir}/{task_name}/videos/{split}/{episode_id}/robot0_randomview_{i}_ref.npy
```

例如：
```
cosmos-predict2/datasets/robocasa_im256_ep100_cam2+4/
└── PnPCounterToSink/
    └── videos/
        └── train/
            └── 79/
                ├── robot0_randomview_0.mp4          # 原视频（动态场景）
                ├── robot0_randomview_0_ref.mp4      # ref视频（静态场景）
                ├── robot0_randomview_0.npy
                ├── robot0_randomview_0_ref.npy
                ├── robot0_randomview_1.mp4
                ├── robot0_randomview_1_ref.mp4
                ...
```

## 使用场景

这些 ref 视频可以用于：
1. **视觉对比**：对比动态场景和静态场景，理解机器人的运动轨迹
2. **相机校准**：验证相机参数是否正确
3. **数据增强**：为模型提供静态场景的参考视角
4. **调试**：检查 initial state 是否正确加载

## 故障排查

### 1. 找不到 demo_key

如果出现"未找到对应的 demo"错误，检查：
- HDF5 文件路径是否正确
- Episode ID 是否超出 demo 数量范围

### 2. 找不到相机

如果出现"未找到可用的相机"错误，说明环境中没有可用的相机进行渲染。

### 3. extrinsic_matrix 不存在

如果出现"extrinsic_matrix 不存在"错误，说明 annotation JSON 文件不完整。

### 4. 环境创建失败

确保已正确安装 RoboCasa 和相关依赖。

## 性能建议

1. 先使用 `--max-workers 1` 进行单线程处理，确保稳定
2. 如果系统资源充足，可以逐步增加 `--max-workers` 的值
3. 先用测试脚本测试单个 episode

## 完整示例

```bash
# 1. 测试单个episode
python3 test_camera_video_single.py

# 2. 为所有任务生成ref视频
python3 generate_camera_moving_videos.py --camera-width 256 --camera-height 256 --max-workers 1

# 3. 只处理特定任务
python3 generate_camera_moving_videos.py --task-name PnPCounterToMicrowave
```

## 技术细节

### 相机参数转换

脚本使用 `invert_camera_extrinsic_matrix` 函数将 extrinsic matrix 转换为相机位置和旋转：
1. 读取 4x4 extrinsic matrix
2. 应用坐标系转换（OpenGL 到 MuJoCo）
3. 提取相机位置和旋转矩阵
4. 使用 `set_camera_pose_world` 设置相机位置

### 静态场景保持

通过 `reset_to(env, initial_state)` 函数：
1. 加载 demo 的 initial state（第一帧的状态）
2. 重置环境到这个状态
3. 在整个渲染过程中保持场景不变
4. 只改变相机位置
