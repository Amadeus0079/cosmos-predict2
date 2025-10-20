import os
import json
import shutil
from pathlib import Path

def main():
    # 根目录
    root_dir = "/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichen/cosmos-predict2/datasets/robocasa_multiview"
    
    # 子数据集划分
    splits = ["val", "train", "test"]
    
    # 记录删除的文件夹数量
    total_deleted = 0
    
    for split in splits:
        # 构建路径
        annotation_dir = os.path.join(root_dir, "annotation", split)
        videos_dir = os.path.join(root_dir, "videos", split)
        
        # 检查路径是否存在
        if not os.path.exists(annotation_dir):
            print(f"警告: annotation路径不存在 - {annotation_dir}")
            continue
            
        if not os.path.exists(videos_dir):
            print(f"警告: videos路径不存在 - {videos_dir}")
            continue
        
        # 获取annotation中所有json文件的编号
        annotation_ids = set()
        for filename in os.listdir(annotation_dir):
            if filename.endswith(".json"):
                # 提取数字部分 (例如从"1.json"中提取"1")
                file_id = filename[:-5]  # 去除".json"后缀
                annotation_ids.add(file_id)
        
        # 获取videos中所有子文件夹的编号
        video_folder_ids = set()
        for foldername in os.listdir(videos_dir):
            folder_path = os.path.join(videos_dir, foldername)
            if os.path.isdir(folder_path):
                video_folder_ids.add(foldername)
        
        # 找出需要删除的文件夹 (存在于videos但不存在于annotation中的)
        folders_to_delete = video_folder_ids - annotation_ids
        
        # 显示信息
        print(f"\n处理 {split} 数据集:")
        print(f"  annotation中有 {len(annotation_ids)} 个json文件")
        print(f"  videos中有 {len(video_folder_ids)} 个子文件夹")
        print(f"  需要删除 {len(folders_to_delete)} 个多余的子文件夹")
        
        # 删除多余的文件夹
        for folder_id in folders_to_delete:
            folder_path = os.path.join(videos_dir, folder_id)
            try:
                # 使用shutil.rmtree删除文件夹及其内容
                shutil.rmtree(folder_path)
                total_deleted += 1
                # 每删除10个显示一次进度
                if total_deleted % 10 == 0:
                    print(f"  已删除 {total_deleted} 个文件夹...")
            except Exception as e:
                print(f"  错误: 删除文件夹 {folder_path} 失败 - {str(e)}")
    
    print(f"\n处理完成! 总共删除了 {total_deleted} 个多余的子文件夹")

if __name__ == "__main__":
    # 提示用户确认操作
    confirm = input("警告: 此操作将永久删除文件。请确认是否继续? (y/n): ")
    if confirm.lower() == 'y' or confirm.lower() == 'yes':
        main()
    else:
        print("操作已取消")
