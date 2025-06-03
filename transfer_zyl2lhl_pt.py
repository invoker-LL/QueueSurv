import os
import h5py
import torch
import numpy as np
from multiprocessing import Pool, cpu_count

# 读取 HDF5 文件
def read_h5_to_tensor(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        # 获取数据集
        features = f['features'][()]
        coords = f['coords'][()]
        # 将 NumPy 数组转换为 PyTorch 张量
        features = torch.tensor(features)
        coords = np.asarray(coords)  # 如果需要，可以将 coords 也转为张量
    features_all = {'features': [features], 'coords': [coords]}
    return features_all

# 处理单个文件的函数
def process_file(feat_name, original_root, save_root):
    if feat_name.find('DX')<0: return
    ori_path = os.path.join(original_root, feat_name)
    save_path = os.path.join(save_root, feat_name.replace('.h5', '.pt'))

    # 如果目标文件已存在，则跳过
    if os.path.exists(save_path):
        print(f"Skipping {feat_name}, already exists.")
        return

    # 读取并保存文件
    try:
        save_pt = read_h5_to_tensor(ori_path)
        torch.save(save_pt, save_path)
        print(f"Processed {feat_name}")
    except Exception as e:
        print(f"Error processing {feat_name}: {e}")

# 主函数
def main():
    original_root = '/mnt/Xsky/zyl/dataset/TCGA/instance_features_UNI2'
    save_root = '/mnt/Xsky/lhl/wsi_ft_git/data_feat/TCGA_UNI2_features/pt_files'
    os.makedirs(save_root, exist_ok=True)

    # 获取所有 HDF5 文件
    all_features = os.listdir(original_root)

    # 使用多进程加速
    with Pool(processes=8) as pool:  # 使用所有可用的 CPU 核心
        pool.starmap(process_file, [(feat_name, original_root, save_root) for feat_name in all_features])

if __name__ == "__main__":
    main()