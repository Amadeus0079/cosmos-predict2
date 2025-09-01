import torch
import kaolin as kal
import numpy as np
from PIL import Image
from kaolin.render.camera import Camera
import math

def render_pointcloud_and_save(points, colors, output_path, image_size=(512, 512)):
    """
    使用Kaolin库渲染点云并保存为图片（兼容新版本Kaolin）
    """
    # 确保使用GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = points.to(device)
    colors = colors.to(device)
    
    # 设置相机参数
    # 相机位置（世界坐标系）
    camera_position = torch.tensor([0.0, 0.0, 3.0], device=device)
    # 相机目标点（看向的位置）
    camera_look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
    # 相机上方向
    camera_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    # 计算视图矩阵（替代generate_view_matrix）
    # 方法：通过相机位置、目标点和上方向计算视图变换
    forward = camera_look_at - camera_position
    forward = forward / torch.norm(forward)
    right = torch.cross(forward, camera_up)
    right = right / torch.norm(right)
    up = torch.cross(right, forward)
    
    view_matrix = torch.eye(4, device=device)
    view_matrix[:3, :3] = torch.stack([right, up, -forward], dim=1)
    view_matrix[:3, 3] = -torch.matmul(view_matrix[:3, :3], camera_position)
    
    # 设置透视投影相机（替代generate_perspective_projection）
    fov = 30.0  # 垂直视场角（度）
    aspect_ratio = image_size[0] / image_size[1]
    near = 0.1
    far = 10.0
    
    cam = kal.render.camera.Camera.from_args(eye=torch.tensor([2., 0., 0.]),
                                         at=torch.tensor([0., 0., 0.]),
                                         up=torch.tensor([0., 1., 0.]),
                                         fov=math.pi * 45 / 180,
                                         width=512, height=512, device='cuda')
    projection_matrix = cam.projection_matrix()
    
    # 点云渲染器
    renderer = kal.render.PointCloudRenderer(
        image_width=image_size[0],
        image_height=image_size[1],
        device=device
    )
    
    # 渲染点云
    point_size = 2.0
    rendered_image, _ = renderer(
        points, 
        colors, 
        view_matrix, 
        projection_matrix, 
        point_size=point_size
    )
    
    # 保存图像
    rendered_image = rendered_image.cpu().detach().numpy()
    rendered_image = (rendered_image * 255).astype(np.uint8)
    rendered_image = np.transpose(rendered_image, (1, 2, 0))
    
    image = Image.fromarray(rendered_image)
    image.save(output_path)
    print(f"渲染完成，图片已保存至: {output_path}")

if __name__ == "__main__":
    # 创建示例点云（球体表面的点）
    num_points = 10000
    theta = torch.rand(num_points) * 2 * np.pi
    phi = torch.acos(1 - 2 * torch.rand(num_points))
    r = 1.0
    
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    points = torch.stack([x, y, z], dim=1)
    
    # 生成颜色
    colors = (points + 1.0) / 2.0  # 从[-1,1]映射到[0,1]
    
    # 渲染并保存
    render_pointcloud_and_save(
        points=points,
        colors=colors,
        output_path="pointcloud_rendering.png",
        image_size=(800, 600)
    )
    