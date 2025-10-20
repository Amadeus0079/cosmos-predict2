import os
from collections import defaultdict

# -------------------------- 1. 配置目标路径 --------------------------
# 替换为你的根路径（即 annotation 文件夹的绝对路径）
root_dir = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_multiview/videos"
# 待检查的三个子文件夹
subdirs = ["val", "train", "test"]

# -------------------------- 2. 读取文件信息 --------------------------
# 存储结构：key=文件名, value=该文件所在的文件夹列表（如 {"a.txt": ["train", "val"]}）
file_location = defaultdict(list)
# 存储每个文件夹的文件数量
folder_file_count = {}

for subdir in subdirs:
    # 拼接子文件夹的完整路径
    subdir_path = os.path.join(root_dir, subdir)
    # 检查路径是否存在（避免路径错误）
    if not os.path.exists(subdir_path):
        print(f"警告：路径不存在 -> {subdir_path}")
        continue
    
    # 遍历子文件夹内的所有文件（仅文件，不包含子文件夹）
    file_names = []
    for name in os.listdir(subdir_path):
        # 拼接文件完整路径，判断是否为文件（排除子文件夹）
        file_path = os.path.join(subdir_path, name)
        # if os.path.isfile(file_path):
        if os.path.isdir(file_path):
            file_names.append(name)
            # 记录该文件所在的文件夹
            file_location[name].append(subdir)
    
    # 统计当前文件夹的文件数量
    folder_file_count[subdir] = len(file_names)

# -------------------------- 3. 输出结果 --------------------------
print("=" * 60)
print("1. 各文件夹文件数量统计：")
print("-" * 60)
for subdir, count in folder_file_count.items():
    print(f"   {subdir:6s} 文件夹：{count:4d} 个文件")

print("\n" + "=" * 60)
print("2. 跨文件夹重复文件名检测结果：")
print("-" * 60)
# 筛选出出现在多个文件夹中的文件名
duplicate_files = {name: dirs for name, dirs in file_location.items() if len(dirs) > 1}

if not duplicate_files:
    print("   ✅ 无跨文件夹重复文件名！")
else:
    print(f"   ❌ 共发现 {len(duplicate_files)} 个重复文件名，详情如下：")
    print("   " + "-" * 56)
    for idx, (name, dirs) in enumerate(duplicate_files.items(), 1):
        # 格式化输出：文件名 + 所在文件夹（如 "a.txt -> [train, val]"）
        dirs_str = "、".join(dirs)
        print(f"   {idx:2d}. 文件名：{name:<30s}  所在文件夹：{dirs_str}")

print("\n" + "=" * 60)