import open3d as o3d
import numpy as np

# 生成一个随机点云
points = np.random.rand(1000, 3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 给点云加颜色
colors = np.random.rand(1000, 3)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# 创建渲染器 (offscreen)
w, h = 800, 600
renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)

# 设置材质
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultUnlit"

# 添加点云到场景
renderer.scene.add_geometry("pcd", point_cloud, material)

# 设置相机视角
center = point_cloud.get_center()
eye = center + np.array([1, 1, 1])  # 相机位置
up = [0, 0, 1]
renderer.setup_camera(60, center, eye, up)

# 渲染并保存
img = renderer.render_to_image()
o3d.io.write_image("/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/output/pointcloud.png", img)

print("渲染完成，已保存 pointcloud.png")
