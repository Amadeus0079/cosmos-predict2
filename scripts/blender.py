import bpy
import numpy as np
import cv2
from mathutils import Matrix

def video2numpy(video_path):
    cap = cv2.VideoCapture(video_path)  # 打开视频
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for ret, frame in iter(cap.read, (False, None)) if ret]
    cap.release()  # 释放资源
    return np.array(frames, dtype=np.uint8)

def clear_scene():
    """清除Blender场景中的所有默认对象"""
    # 确保处于对象模式
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # 选择并删除所有对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_point_cloud(rgb_image, depth_image, camera_intrinsics, 
                      depth_scale=1000.0, point_size=0.01, max_points=100000):
    """
    从RGB图像和深度图创建点云
    
    参数:
        rgb_path: RGB图像路径
        depth_path: 深度图路径
        camera_intrinsics: 相机内参矩阵
        depth_scale: 深度缩放因子，将深度值转换为米
        point_size: 点的大小
        max_points: 最大点数量，防止点云过大导致性能问题
    """

    h, w = depth_image.shape[:2]
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    # 创建网格坐标
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    
    # 获取深度值并过滤无效值
    z = depth_image.flatten() / depth_scale  # 转换为米
    valid_mask = z > 0
    
    # 像素坐标转3D坐标
    x = (u[valid_mask] - cx) * z[valid_mask] / fx
    y = (v[valid_mask] - cy) * z[valid_mask] / fy
    z = z[valid_mask]
    
    # 获取对应颜色
    colors = rgb_image[v[valid_mask], u[valid_mask]] / 255.0  # 归一化到[0,1]
    
    # 随机采样点以避免点云过大
    if len(x) > max_points:
        indices = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
        colors = colors[indices]
    
    # 创建点云数据
    vertices = np.column_stack((x, y, z)).tolist()
    vertex_colors = colors.tolist()
    
    # 在Blender中创建点云网格
    mesh = bpy.data.meshes.new("PointCloudMesh")
    obj = bpy.data.objects.new("PointCloud", mesh)
    bpy.context.collection.objects.link(obj)
    
    # 设置顶点
    mesh.from_pydata(vertices, [], [])
    
    # 添加顶点颜色
    mesh.vertex_colors.new(name="Col")
    for i, color in enumerate(vertex_colors):
        # 添加alpha通道并转换为Blender的颜色空间
        mesh.vertex_colors["Col"].data[i].color = (*color, 1.0)
    
    # 创建点云材质
    mat = bpy.data.materials.new(name="PointCloudMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # 清除默认节点
    for node in nodes:
        nodes.remove(node)
    
    # 创建必要的节点
    output = nodes.new(type='ShaderNodeOutputMaterial')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    attr = nodes.new(type='ShaderNodeAttribute')
    attr.attribute_name = "Col"  # 使用顶点颜色属性
    
    # 连接节点
    links.new(attr.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # 将材质分配给点云对象
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    # 设置点大小 - 使用实例化小球的方式
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 创建一个小球作为点的显示
    bpy.ops.mesh.primitive_uv_sphere_add(radius=point_size)
    sphere = bpy.context.active_object
    sphere.parent = obj
    obj.instance_type = 'VERTS'  # 每个顶点实例化一个小球
    
    return obj

def setup_camera(extrinsics_matrix):
    """
    根据外参矩阵设置相机位置和旋转
    
    参数:
        extrinsics_matrix: 相机外参矩阵 (4x4)
    """
    # 创建相机
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    
    # 转换外参矩阵到Blender坐标系
    # Blender使用Y轴向上，而常规相机坐标系使用Z轴向上，需要调整
    blender_matrix = Matrix(extrinsics_matrix)
    
    # 提取旋转和平移
    rotation = blender_matrix.to_3x3().to_quaternion()
    location = blender_matrix.translation
    
    # 设置相机位置和旋转
    camera.location = location
    camera.rotation_euler = rotation.to_euler()
    
    # 设置相机为活动相机
    bpy.context.scene.camera = camera
    
    return camera

def setup_lighting():
    """设置场景灯光以获得良好的渲染效果"""
    # 创建主光源
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.active_object
    sun.data.energy = 2.0
    
    # 创建补光灯
    bpy.ops.object.light_add(type='AREA', location=(-5, 5, 8))
    area = bpy.context.active_object
    area.data.shape = 'RECTANGLE'
    area.data.size = 4
    area.data.size_y = 4
    area.data.energy = 800.0
    
    # 设置环境光
    bpy.context.scene.world.light_settings.use_ambient_occlusion = True
    bpy.context.scene.world.light_settings.ao_factor = 0.2
    
    return sun, area

def render_scene(output_path, resolution_x=800, resolution_y=600, engine='CYCLES'):
    """
    渲染场景并保存结果
    
    参数:
        output_path: 渲染结果保存路径
        resolution_x: 渲染宽度
        resolution_y: 渲染高度
        engine: 渲染引擎 ('CYCLES' 或 'BLENDER_EEVEE')
    """
    # 设置渲染参数
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    
    # 设置输出路径和格式
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    
    # 如果使用Cycles引擎，设置高质量参数
    if engine == 'CYCLES':
        scene.cycles.samples = 128  # 采样数，越高质量越好但越慢
        scene.cycles.preview_samples = 16
        scene.cycles.use_denoising = True  # 启用降噪
    
    # 如果使用EEVEE引擎，设置实时渲染参数
    elif engine == 'BLENDER_EEVEE':
        scene.eevee.taa_samples = 16
        scene.eevee.taa_render_samples = 64
    
    # 执行渲染
    bpy.ops.render.render(write_still=True)

def main():
    # 相机内参 (替换为你的实际相机参数)
    camera_intrinsics = np.array([
        [525.0, 0.0, 319.5],   # fx, 0, cx
        [0.0, 525.0, 239.5],   # 0, fy, cy
        [0.0, 0.0, 1.0]        # 0, 0, 1
    ])
    
    # 读取图像
    video_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_multiview/videos/test/4/robot0_agentview_right.mp4"
    depth_path = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_multiview/videos/test/4/robot0_agentview_right.npy"
    video = video2numpy(video_path)
    depths = np.load(depth_path)
    rgb_image = video[0]
    depth_image = depths[0]
    
    # 清除现有场景
    clear_scene()
    
    # 创建点云
    print("正在从RGB和深度图创建点云...")
    point_cloud = create_point_cloud(
        rgb_image, 
        depth_image, 
        camera_intrinsics,
        depth_scale=1000.0,  # 根据你的深度图单位调整
        point_size=0.01,     # 点的大小，根据场景尺度调整
        max_points=100000    # 限制最大点数量，防止性能问题
    )
    
    # 定义新视角的外参 (示例: 旋转30度并向后平移)
    theta = np.radians(30)  # 30度旋转
    new_extrinsics = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 1.5],  # 沿Z轴平移1.5米
        [0, 0, 0, 1]
    ])
    
    # 设置相机
    print("正在设置新视角相机...")
    camera = setup_camera(new_extrinsics)
    
    # 设置灯光
    print("正在设置场景灯光...")
    setup_lighting()
    
    # 渲染场景
    print("正在渲染新视角图像...")
    render_scene(
        "cosmos-predict2/output/blender_rendered_view.png", 
        resolution_x=640, 
        resolution_y=480, 
        engine='CYCLES'  # 可选 'BLENDER_EEVEE' 进行快速渲染
    )
    
    print(f"渲染完成，结果已保存为 blender_rendered_view.png")

if __name__ == "__main__":
    main()
    