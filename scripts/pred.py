import h5py
import imageio
import numpy as np
import os
import cv2
from PIL import Image
import open3d as o3d
from tqdm import tqdm
# from p_tqdm import p_map, t_map
from multiprocessing import Pool
import matplotlib.pyplot as plt
import torch
from render import pytorch3d_render

TASK = "PnPCounterToSink"  # Change this to your task name

def to_next(sequence):
    new_data = sequence[-1]
    new_data_expanded = np.expand_dims(new_data, axis=0)
    sequence = np.concatenate((sequence, new_data_expanded), axis=0)
    sequence = sequence[1:]
    return sequence 
###类似滑动窗口


def normalize_depth(depth_frames):
    new_frames = []
    for frame in depth_frames:
        depth = frame
        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        new_frames.append(depth)
    new_frames = np.array(new_frames)
    return new_frames


def get_pointcloud(image, depth, filename, fx, fy, save=False):
    width, height = image.shape[1], image.shape[0]
    color_image = Image.fromarray(image)
    depth = np.squeeze(depth)

    # Generate mesh grid and calculate point cloud coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / fx
    y = (y - height / 2) / fy
    z = np.array(depth)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

    colors = np.array(color_image).reshape(-1, 3) / 255.0

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if save:
        o3d.io.write_point_cloud(os.path.join(f"outputs/pointclouds/{TASK}", filename + ".ply"), pcd)

    return pcd


def transform_pointcloud(pointcloud, world_matrix, ego_matrix, offset=0, save=False, save_dir=None):
    """
    将点云从一个坐标系转换到另一个坐标系。
    :param pointcloud: Open3D 点云对象
    :param transform_matrix: 4x4 齐次变换矩阵
    :param offset: (x, y, z) 偏移量
    :return: 转换后的点云
    """
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # 将点云转换为齐次坐标
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # 计算变换矩阵
    # world_points = (np.linalg.inv(world_matrix) @ points_homogeneous.T).T
    world_points = (world_matrix @ points_homogeneous.T).T

    # 保存世界点云
    transformed_pointcloud = o3d.geometry.PointCloud()
    transformed_pointcloud.points = o3d.utility.Vector3dVector(world_points[:, :3])
    transformed_pointcloud.colors = o3d.utility.Vector3dVector(colors)

    if save:
        o3d.io.write_point_cloud(os.path.join(save_dir, "transformed_world_1" + ".ply"), transformed_pointcloud)

    # transformed_points = world_pointcloud[:, :3]
    transformed_points = (ego_matrix @ world_points.T).T[:, :3]

    # 添加 z_offset
    transformed_points += offset

    # 创建新的点云
    transformed_pointcloud = o3d.geometry.PointCloud()
    transformed_pointcloud.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_pointcloud.colors = o3d.utility.Vector3dVector(colors)

    if save:
        o3d.io.write_point_cloud(os.path.join(save_dir, "transformed_ego" + ".ply"), transformed_pointcloud)

    return transformed_pointcloud


def get_world_pcd(pointcloud, world_camera_matpos, save=False, save_dir=None):
    """
    将点云从一个坐标系转换到另一个坐标系。
    :param pointcloud: Open3D 点云对象
    :param transform_matrix: 4x4 齐次变换矩阵
    :param offset: (x, y, z) 偏移量
    :return: 转换后的点云
    """
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # 将点云转换为齐次坐标
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # 计算变换矩阵
    # world_points = (np.linalg.inv(world_matrix) @ points_homogeneous.T).T[:, :3]
    world_points = (world_camera_matpos @ points_homogeneous.T).T[:, :3]

    # 保存世界点云
    transformed_pointcloud = o3d.geometry.PointCloud()
    transformed_pointcloud.points = o3d.utility.Vector3dVector(world_points)
    transformed_pointcloud.colors = o3d.utility.Vector3dVector(colors)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(save_dir, "transformed_world" + ".ply"), transformed_pointcloud)

    return transformed_pointcloud


def project_to_image(pointcloud, intrinsic_matrix, image_width, image_height):
    """
    将点云投影到图像平面。
    :param pointcloud: Open3D 点云对象
    :param intrinsic_matrix: 3x3 内参矩阵
    :param image_width: 图像宽度
    :param image_height: 图像高度
    :return: 深度图和颜色图
    """
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # 投影到图像平面
    projected_points = (intrinsic_matrix @ points.T).T
    projected_points[:, :2] /= projected_points[:, 2:3]  # 归一化

    # 创建深度图和颜色图
    depth_image = np.zeros((image_height, image_width))
    color_image = np.zeros((image_height, image_width, 3))

    for i, point in enumerate(projected_points):
        x, y, z = point
        if 0 <= x < image_width and 0 <= y < image_height and z > 0:
            depth_image[int(y), int(x)] = z
            color_image[int(y), int(x)] = colors[i]

    return depth_image, (color_image * 255).astype(np.uint8)


def get_other_view(demo_id=0, filename=None):
    if filename == None:
        filename = f"/Disk2/zichen/mimicgen/mimicgen/core_datasets/{TASK}/demo_src_{TASK}_task_D1/demo.hdf5"
    file = h5py.File(filename, 'r')
    # file = h5py.File(f"/Disk2/zichen/mimicgen/datasets/core/stack_three_d1.hdf5", 'r')
    data = file['data']
    demo = data[f'demo_{demo_id}']

    # Get information
    actions = demo['actions'][()]
    states = demo['states'][()]
    info = demo['datagen_info']
    cam_info = demo['cam_info']
    # eef_poses = info['eef_pose'][()]
    # eef_poses[:, :3, :2] *= -1

    # # Get camera information
    # ego_extrinsic_matrixs = cam_info['robot0_eye_in_hand']['extrinsic_matrix'][:]
    # # ego_extrinsic_matrixs = eef_poses
    # new_data = ego_extrinsic_matrixs[-1]  # 示例数据

    # # 为了预测，将ego_extrinsic_matrixs延后一帧
    # # ego_extrinsic_matrixs = to_next(ego_extrinsic_matrixs)

    # world_extrinsic_matrixs = cam_info['robot0_agentview_right']['extrinsic_matrix'][:]

    # ego_intrinsic_matrixs = cam_info['robot0_eye_in_hand']['intrinsic_matrix'][:]
    # world_intrinsic_matrixs = cam_info['robot0_agentview_right']['intrinsic_matrix'][:]

    # ego_fx = ego_intrinsic_matrixs[0][0][0]
    # ego_fy = ego_intrinsic_matrixs[0][1][1]
    # world_fx = world_intrinsic_matrixs[0][0][0]
    # world_fy = world_intrinsic_matrixs[0][1][1]

    # Get image
    obs = demo['obs']

    world_image = obs['robot0_agentview_right_image'][()]
    ego_image = obs['robot0_eye_in_hand_image'][()]

    # world_depth = obs['robot0_agentview_right_depth'][()]
    # ego_depth = obs['robot0_eye_in_hand_depth'][()]

    index = list(range(0, world_image.shape[0], 10))

    # Export to video
    world_path = f'outputs/videos/{TASK}/{demo_id}_world.mp4'
    ego_path = f'outputs/videos/{TASK}/{demo_id}_ego.mp4'
    trans_path = f'outputs/videos/{TASK}/{demo_id}_trans.mp4'
    cat_path = f'outputs/videos/{TASK}/{demo_id}_cat.mp4'

    os.makedirs(f'outputs/videos/{TASK}', exist_ok=True)

    # Save the videos
    imageio.mimwrite(world_path, world_image, fps=6)
    imageio.mimwrite(ego_path, ego_image, fps=6)

    # trans_frames = []

    # for frame_id in tqdm(index, desc=f"Processing demo {demo_id}"):
    #     world_depth[frame_id] = cv2.flip(world_depth[frame_id], 0)[:, :, np.newaxis]
    #     world_pcd = get_pointcloud(world_image[frame_id], world_depth[frame_id], "world", world_fx, world_fy)
    #     # ego_pcd = get_pointcloud(ego_frames[frame_id], ego_depth[frame_id], "ego", ego_fx, ego_fy)

    #     # print("Pointcloud saved.")

    #     # 假设你已经有 world_pointcloud, world_to_ego_matrix, ego_intrinsic_matrix
    #     # camera_axis_correction = np.array(
    #     # [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    #     # )
    #     world_camera_matpos = world_extrinsic_matrixs[frame_id] # 并非外参矩阵，实则mat+pos
    #     ego_camera_matpos = ego_extrinsic_matrixs[frame_id]

    #     ego_extrinsic_matrix = matrix_c2w(ego_camera_matpos)

    #     world_intrinsic_matrix = world_intrinsic_matrixs[frame_id]
    #     ego_intrinsic_matrix = ego_intrinsic_matrixs[frame_id]

    #     # world_to_ego_matrix = ego_extrinsic_matrix @ np.linalg.inv(world_extrinsic_matrix)
    #     # ego_pointcloud = transform_pointcloud(world_pcd, world_extrinsic_matrix, ego_extrinsic_matrix, offset=offset)
    #     # trans_world_pointcloud = get_world_pcd(world_pcd, world_extrinsic_matrix)
    #     trans_world_pointcloud = get_world_pcd(world_pcd, world_camera_matpos)

    #     # Trans_image, Trans_image = project_to_image(ego_pointcloud, ego_intrinsic_matrix, image_width=256, image_height=256)
    #     Trans_image = render_pcd(trans_world_pointcloud, ego_extrinsic_matrix, ego_intrinsic_matrix, point_size=10)

    #     trans_frames.append(Trans_image)

    # trans_frames = np.array(trans_frames)
    # imageio.mimwrite(trans_path, trans_frames, fps=6)

    # cat_frames = np.concatenate((world_image[index], ego_image[index], trans_frames), axis=2)
    # imageio.mimwrite(cat_path, cat_frames, fps=6)

    # print(f"Complete {demo_id=}.")
    # return trans_frames



def save_pointcloud(demo_id=0, frame_id=0):
    file = h5py.File(f"/Disk2/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_im256.hdf5", 'r')
    # file = h5py.File(f"/Disk2/zichen/mimicgen/datasets/core/stack_three_d1.hdf5", 'r')
    data = file['data']
    demo = data[f'demo_{demo_id}']
    offset = (0, 0, 0)

    # Get information
    actions = demo['actions'][()]
    states = demo['states'][()]
    # info = demo['datagen_info']
    cam_info = demo['cam_info']
    # eef_poses = info['eef_pose'][()]
    # eef_poses[:, :3, :2] *= -1

    # Get camera information
    ego_extrinsic_matrixs = cam_info['ego_extrinsic_matrix'][()]
    # ego_extrinsic_matrixs = eef_poses
    world_extrinsic_matrixs = cam_info['exo_extrinsic_matrix'][()]

    ego_intrinsic_matrixs = cam_info['ego_intrinsic_matrix'][()]
    world_intrinsic_matrixs = cam_info['exo_intrinsic_matrix'][()]

    ego_fx = ego_intrinsic_matrixs[0][0][0]
    ego_fy = ego_intrinsic_matrixs[0][1][1]
    world_fx = world_intrinsic_matrixs[0][0][0]
    world_fy = world_intrinsic_matrixs[0][1][1]

    # Get image
    obs = demo['obs']

    world_frames = obs['robot0_agentview_right_image'][()]

    # flipped_video = np.zeros_like(world_frames)
    # for i in range(world_frames.shape[0]):
    #     flipped_video[i] = cv2.flip(world_frames[i], 0)
        # flipped_video[i] = cv2.flip(flipped_video[i], 1)

    # world_image = obs['birdview_image'][()]
    ego_frames = obs['robot0_eye_in_hand_image'][()]

    world_depth = obs['robot0_agentview_right_depth'][()]
    # world_depth = obs['birdview_depth'][()]
    ego_depth = obs['robot0_eye_in_hand_depth'][()]

    for i in range(world_depth.shape[0]):
        world_depth[i] = cv2.flip(world_depth[i], 0)[:, :, np.newaxis]
        ego_depth[i] = cv2.flip(ego_depth[i], 0)[:, :, np.newaxis]

    os.makedirs(f'outputs/pointclouds/{TASK}/frame_{frame_id}', exist_ok=True)

    world_pcd = get_pointcloud(world_frames[frame_id], world_depth[frame_id], f"frame_{frame_id}/world", world_fx, world_fy, save=True)
    ego_pcd = get_pointcloud(ego_frames[frame_id], ego_depth[frame_id], f"frame_{frame_id}/ego", ego_fx, ego_fy, save=True)

    world_extrinsic_matrix = world_extrinsic_matrixs[frame_id]

    ego_extrinsic_matrix = ego_extrinsic_matrixs[frame_id]
    # new_ego_extrinsic_matrix = recorrect_ego_matrix(ego_extrinsic_matrix)
    new_ego_extrinsic_matrix = matrix_c2w(ego_extrinsic_matrix)

    world_intrinsic_matrix = world_intrinsic_matrixs[frame_id]
    ego_intrinsic_matrix = ego_intrinsic_matrixs[frame_id]

    ego_pointcloud = transform_pointcloud(world_pcd, world_extrinsic_matrix, new_ego_extrinsic_matrix, offset=offset, save=True, save_dir=f'outputs/pointclouds/{TASK}/frame_{frame_id}')
    trans_world_pointcloud = get_world_pcd(world_pcd, world_extrinsic_matrix, save=True, save_dir=f'outputs/pointclouds/{TASK}/frame_{frame_id}')

    render_pcd(trans_world_pointcloud, new_ego_extrinsic_matrix, ego_intrinsic_matrix, point_size=10)

    print("Get pointcloud.")


def get_extrinsic_matrix(action):
    pos = action[:3]
    rot = action[3:6]
    x, y, z = pos
    roll, pitch, yaw = rot

    # 绕x轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 绕y轴的旋转矩阵
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # 绕z轴的旋转矩阵
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 总的旋转矩阵
    R = np.dot(Rz, np.dot(Ry, Rx))

    # 构建外参矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -np.dot(R, np.array([x, y, z]))

    return T


def get_pose_from_action(action):
    target_pos = action[:3]
    target_quat = T.axisangle2quat(action[3:6])
    target_rot = T.quat2mat(target_quat)
    target_pose = PoseUtils.make_pose(target_pos, target_rot)
    return target_pose


def matrix_c2w(camera_matrix):
    new_matrix = camera_matrix.copy()

    pos = new_matrix[:3, 3]

    rot = new_matrix[:3, :3]

    t = - rot.T @ pos

    new_matrix[:3, 3] = t
    new_matrix[:3, :3] = rot.T
    return new_matrix


def get_tmp_pred(demo_id=0, data_dir=f"/Disk2/zichen/mimicgen/mimicgen/core_datasets/{TASK}/demo_src_{TASK}_task_D1"):
    tmp_dir = os.path.join(data_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    filename = os.path.join(tmp_dir, f"demo_{demo_id}.hdf5")
    if os.path.exists(filename):
        print(f"File {filename} already exists.")
        return
    data_writer = h5py.File(filename, 'w')

    data_grp = data_writer.create_group("data")

    trans_frames = get_other_view(demo_id)
    data_grp.create_dataset("agentview_pred", data=trans_frames, compression="gzip")

    data_writer.close()
    return


def add_pred(demo_list, data_dir):
    dataset_path = os.path.join(data_dir, "pred.hdf5")
    data_writer = h5py.File(dataset_path, "w")
    data_grp = data_writer.create_group("data")

    for demo_id in tqdm(demo_list, desc="demo_process"):
        demo_grp = data_grp.create_group(f"demo_{demo_id}")
        trans_frames = get_other_view(demo_id)
        demo_grp.create_dataset("agentview_pred", data=trans_frames, compression="gzip")

    data_writer.close()

    return


def vis_pred(demo_id, data_dir):
    dataset_path = os.path.join(data_dir, "pred.hdf5")
    file = h5py.File(dataset_path, 'r')
    data = file['data']
    demo = data[f"demo_{demo_id}"]
    pred = demo["agentview_pred"][()]
    imageio.mimwrite("outputs/videos/pred.mp4", pred, fps=6)

    return


def reconstruct_pred(obs, cam_info, exo_cam="robot0_agentview_right", device=None):

    # Get camera information
    ego_extrinsic_matrixs = cam_info['robot0_eye_in_hand']['extrinsic_matrix']

    # 为了预测，将ego_extrinsic_matrixs延后一帧
    ego_extrinsic_matrixs = to_next(ego_extrinsic_matrixs)

    world_extrinsic_matrixs = cam_info[exo_cam]['extrinsic_matrix']

    ego_intrinsic_matrixs = cam_info['robot0_eye_in_hand']['intrinsic_matrix']
    world_intrinsic_matrixs = cam_info[exo_cam]['intrinsic_matrix']

    ego_fx = ego_intrinsic_matrixs[0][0][0]
    ego_fy = ego_intrinsic_matrixs[0][1][1]
    world_fx = world_intrinsic_matrixs[0][0][0]
    world_fy = world_intrinsic_matrixs[0][1][1]

    world_image = obs[f'{exo_cam}_image']
    ego_image = obs['robot0_eye_in_hand_image']

    world_depth = obs[f'{exo_cam}_depth']
    ego_depth = obs['robot0_eye_in_hand_depth']

    index = list(range(0, world_image.shape[0], 1))

    # Export to video
    # world_path = f'outputs/videos/{TASK}/{demo_id}_world.mp4'
    # ego_path = f'outputs/videos/{TASK}/{demo_id}_ego.mp4'
    # trans_path = f'outputs/videos/{exo_cam}_trans.mp4'
    # cat_path = f'outputs/videos/{exo_cam}_cat.mp4'

    # os.makedirs(f'outputs/videos/{TASK}', exist_ok=True)

    # Save the videos
    # imageio.mimwrite(world_path, world_image, fps=6)
    # imageio.mimwrite(ego_path, ego_image, fps=6)

    trans_frames = []

    for frame_id in index:
        world_depth[frame_id] = cv2.flip(world_depth[frame_id], 0)[:, :, np.newaxis]
        world_pcd = get_pointcloud(world_image[frame_id], world_depth[frame_id], "world", world_fx, world_fy)
        # ego_pcd = get_pointcloud(ego_frames[frame_id], ego_depth[frame_id], "ego", ego_fx, ego_fy)

        # print("Pointcloud saved.")

        # 假设你已经有 world_pointcloud, world_to_ego_matrix, ego_intrinsic_matrix
        # camera_axis_correction = np.array(
        # [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        # )
        world_camera_matpos = world_extrinsic_matrixs[frame_id] # 并非外参矩阵，实则mat+pos
        ego_camera_matpos = ego_extrinsic_matrixs[frame_id]

        ego_extrinsic_matrix = matrix_c2w(ego_camera_matpos)

        world_intrinsic_matrix = world_intrinsic_matrixs[frame_id]
        ego_intrinsic_matrix = ego_intrinsic_matrixs[frame_id]

        # world_to_ego_matrix = ego_extrinsic_matrix @ np.linalg.inv(world_extrinsic_matrix)
        ego_pointcloud = transform_pointcloud(world_pcd, world_camera_matpos, ego_extrinsic_matrix)
        # trans_world_pointcloud = get_world_pcd(world_pcd, world_extrinsic_matrix)
        # trans_world_pointcloud = get_world_pcd(world_pcd, world_camera_matpos, save=True, save_dir=f'outputs/pointclouds/{TASK}')
        trans_world_pointcloud = get_world_pcd(world_pcd, world_camera_matpos)
        # o3d.io.write_point_cloud(os.path.join("/Disk2/zichen/robocasa/outputs/pointclouds/PnPCounterToSink/", exo_cam + ".ply"), trans_world_pointcloud)
        

        # Trans_image, Trans_image = project_to_image(ego_pointcloud, ego_intrinsic_matrix, image_width=256, image_height=256)
        # Trans_image = render_pcd(trans_world_pointcloud, ego_extrinsic_matrix, ego_intrinsic_matrix, point_size=10)
        # Trans_image = render_pcd(ego_pointcloud,
        #                             np.array([[1, 0, 0, 0],
        #                                         [0, 1, 0, 0],
        #                                         [0, 0, 1, 0],
        #                                         [0, 0, 0, 1]]),
        #                             ego_intrinsic_matrix,
        #                             point_size=10)
        Trans_image = pytorch3d_render(pcd=ego_pointcloud,
                                        extrinsic_matrix=np.array([[-1, 0, 0, 0],
                                                                    [0, -1, 0, 0],
                                                                    [0, 0, 1, 0],
                                                                    [0, 0, 0, 1]]),
                                        intrinsic_matrix=ego_intrinsic_matrix,
                                        width=256,
                                        height=256,
                                        point_size=15,
                                        background_color=(1, 1, 1),  # 白色背景
                                        device=device
                                    )

        trans_frames.append((Trans_image['image'] * 255).astype(np.uint8))
        # trans_frames.append(Trans_image)

    trans_frames = np.array(trans_frames)
    # imageio.mimwrite(trans_path, trans_frames, fps=6)

    # cat_frames = np.concatenate((world_image[index], ego_image[index], trans_frames), axis=2)
    # imageio.mimwrite(cat_path, cat_frames, fps=6)

    # print(f"Complete {demo_id=}.")
    return trans_frames


def get_cam_world_pcd(obs, cam_info, exo_cam="robot0_agentview_right", device=None, img_frame_id=0, cam_frame_id=0):

    # Get camera information
    ego_extrinsic_matrixs = cam_info['robot0_eye_in_hand']['extrinsic_matrix']
    ego_extrinsic_matrixs = to_next(ego_extrinsic_matrixs)

    world_extrinsic_matrixs = cam_info[exo_cam]['extrinsic_matrix']

    ego_intrinsic_matrixs = cam_info['robot0_eye_in_hand']['intrinsic_matrix']
    world_intrinsic_matrixs = cam_info[exo_cam]['intrinsic_matrix']

    ego_fx = ego_intrinsic_matrixs[0][0][0]
    ego_fy = ego_intrinsic_matrixs[0][1][1]
    world_fx = world_intrinsic_matrixs[0][0][0]
    world_fy = world_intrinsic_matrixs[0][1][1]

    world_image = obs[f'{exo_cam}_image']
    ego_image = obs['robot0_eye_in_hand_image']

    world_depth = obs[f'{exo_cam}_depth']
    ego_depth = obs['robot0_eye_in_hand_depth']

    index = list(range(0, world_image.shape[0], 1))

    trans_frames = []

    # if exo_cam == "robot0_eye_in_hand":
    world_depth[img_frame_id] = cv2.flip(world_depth[img_frame_id], 0)[:, :, np.newaxis]
    
    world_pcd = get_pointcloud(world_image[img_frame_id], world_depth[img_frame_id], "world", world_fx, world_fy)
    # ego_pcd = get_pointcloud(ego_frames[frame_id], ego_depth[frame_id], "ego", ego_fx, ego_fy)

    # print("Pointcloud saved.")

    # 假设你已经有 world_pointcloud, world_to_ego_matrix, ego_intrinsic_matrix
    # camera_axis_correction = np.array(
    # [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    # )
    world_camera_matpos = world_extrinsic_matrixs[img_frame_id] # 并非外参矩阵，实则mat+pos
    ego_camera_matpos = ego_extrinsic_matrixs[cam_frame_id]

    ego_extrinsic_matrix = matrix_c2w(ego_camera_matpos)

    world_intrinsic_matrix = world_intrinsic_matrixs[img_frame_id]
    ego_intrinsic_matrix = ego_intrinsic_matrixs[cam_frame_id]

    # world_to_ego_matrix = ego_extrinsic_matrix @ np.linalg.inv(world_extrinsic_matrix)
    trans_world_pointcloud = transform_pointcloud(world_pcd, world_camera_matpos, ego_extrinsic_matrix)
    # trans_world_pointcloud = get_world_pcd(world_pcd, world_extrinsic_matrix)
    # trans_world_pointcloud = get_world_pcd(world_pcd, world_camera_matpos, save=True, save_dir=f'outputs/pointclouds/{TASK}')
    # trans_world_pointcloud = get_world_pcd(world_pcd, world_camera_matpos)
    # o3d.io.write_point_cloud(os.path.join("/Disk2/zichen/robocasa/outputs/pointclouds/multicam/", exo_cam + ".ply"), trans_world_pointcloud)

    return trans_world_pointcloud


def get_all_pred(obs, cam_info, device=None):
    frame_num = obs["robot0_agentview_right_image"].shape[0]

    video_list = []

    for frame_id in range(frame_num):

        # get two image
        # left_image = obs["robot0_agentview_left_image"][frame_id]
        # right_image = obs["robot0_agentview_right_image"][frame_id]
        # hand_image = obs["robot0_eye_in_hand_image"][frame_id]

        right_world = get_cam_world_pcd(obs, cam_info, exo_cam="robot0_agentview_right", img_frame_id=frame_id, cam_frame_id=frame_id)
        left_world = get_cam_world_pcd(obs, cam_info, exo_cam="robot0_agentview_left", img_frame_id=frame_id, cam_frame_id=frame_id)
        hand_world = get_cam_world_pcd(obs, cam_info, exo_cam="robot0_eye_in_hand", img_frame_id=frame_id, cam_frame_id=frame_id)

        all_world = right_world + hand_world + left_world
        # all_world = hand_world
        # o3d.io.write_point_cloud(os.path.join("/Disk2/zichen/robocasa/outputs/pointclouds/multicam/", "all_world.ply"), all_world)

        Trans_image = pytorch3d_render(pcd=all_world,
                                            extrinsic_matrix=np.array([[-1, 0, 0, 0],
                                                                        [0, -1, 0, 0],
                                                                        [0, 0, 1, 0],
                                                                        [0, 0, 0, 1]]),
                                            intrinsic_matrix=cam_info['robot0_eye_in_hand']['intrinsic_matrix'][frame_id],
                                            width=256,
                                            height=256,
                                            point_size=10,
                                            background_color=(1, 1, 1),  # 白色背景
                                            znear=0.1,
                                            device=device
                                        )
        
        render_image = (Trans_image['image'] * 255).astype(np.uint8)
        # all_image = cv2.cvtColor(all_image, cv2.COLOR_RGB2BGR)

        video_list.append(render_image)
    
    video_list = np.array(video_list)
    # cv2.imwrite("/Disk2/zichen/robocasa/outputs/pointclouds/multicam/all_image.png", all_image)
    # imageio.mimwrite('/Disk2/zichen/robocasa/outputs/pointclouds/multicam/render1.mp4', video_list, fps=6)
    
    return video_list


def get_pred_multi(obs, cam_info, device=None):
    frame_num = obs["robot0_agentview_right_image"].shape[0]

    video_list = []

    for frame_id in range(frame_num):

        # get two image
        # left_image = obs["robot0_agentview_left_image"][frame_id]
        # right_image = obs["robot0_agentview_right_image"][frame_id]
        # hand_image = obs["robot0_eye_in_hand_image"][frame_id]

        right_world = get_cam_world_pcd(obs, cam_info, exo_cam="robot0_agentview_right", img_frame_id=frame_id, cam_frame_id=frame_id)
        # left_world = get_cam_world_pcd(obs, cam_info, exo_cam="robot0_agentview_left", img_frame_id=frame_id, cam_frame_id=frame_id)
        hand_world = get_cam_world_pcd(obs, cam_info, exo_cam="robot0_eye_in_hand", img_frame_id=frame_id, cam_frame_id=frame_id)

        all_world = right_world + hand_world
        # all_world = hand_world
        # o3d.io.write_point_cloud(os.path.join("/Disk2/zichen/robocasa/outputs/pointclouds/multicam/", "all_world.ply"), all_world)

        Trans_image = pytorch3d_render(pcd=all_world,
                                        extrinsic_matrix=np.array([[-1, 0, 0, 0],
                                                                    [0, -1, 0, 0],
                                                                    [0, 0, 1, 0],
                                                                    [0, 0, 0, 1]]),
                                        intrinsic_matrix=cam_info['robot0_eye_in_hand']['intrinsic_matrix'][frame_id],
                                        width=256,
                                        height=256,
                                        point_size=10,
                                        background_color=(1, 1, 1),  # 白色背景
                                        znear=0.1,
                                        device=device
                                        )
        
        render_image = (Trans_image['image'] * 255).astype(np.uint8)
        # all_image = cv2.cvtColor(all_image, cv2.COLOR_RGB2BGR)

        video_list.append(render_image)
    
    video_list = np.array(video_list)
    # cv2.imwrite("/Disk2/zichen/robocasa/outputs/pointclouds/multicam/all_image.png", all_image)
    # imageio.mimwrite('/Disk2/zichen/robocasa/outputs/pointclouds/multicam/render1.mp4', video_list, fps=6)
    
    return video_list



if __name__ == "__main__":
    # get_other_view(demo_id=5, filename="/Disk2/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_gentex_im128_randcams.hdf5")
    # save_pointcloud(0, 0)
    f = h5py.File("/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/mg/2024-05-04-22-14-06_and_2024-05-07-07-40-17/demo_gentex_im128_randcams_multi.hdf5", 'r')
    demo = f["data"]["demo_1"]  # access demo 0
    obs = {}

    for key in demo["obs"].keys():
        obs[key] = demo["obs"][key][:]

    cam_info = {}
    for key in demo["cam_info"].keys():
        cam_info[key] = {}
        for kp in demo["cam_info"][key].keys():
            cam_info[key][kp] = demo["cam_info"][key][kp][:]

    # get two image
    # left_image = obs["robot0_agentview_left_image"][0]
    # right_image = obs["robot0_agentview_right_image"][0]
    # hand_image = obs["robot0_eye_in_hand_image"][0]

    # cv2.imwrite("outputs/left_image.png", left_image)
    # cv2.imwrite("outputs/right_image.png", right_image)
    # cv2.imwrite("outputs/hand_image.png", hand_image)


    # reconstruct_pred(obs, cam_info, exo_cam="robot0_agentview_right")
    # reconstruct_pred(obs, cam_info, exo_cam="robot0_agentview_left")
    video = get_all_pred(obs, cam_info, device=torch.device("cuda:0"))
    imageio.mimwrite('/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/output/test.mp4', video, fps=1)
    