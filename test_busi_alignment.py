#!/usr/bin/env python3
"""测试修正后的BUSI配置"""

import os
import sys
sys.path.append('/app/MixDSemi/SynFoCLIP/code')

# 导入修正后的处理逻辑
from preprocess import DATASET_MAPPING
from batchgenerators.utilities.file_and_folder_operations import *
from glob import glob
import random

def test_busi_alignment():
    """测试BUSI数据集对齐"""
    
    print("=== Testing BUSI dataset alignment ===")
    
    # 使用与preprocess.py相同的配置
    dataset_name = 'BUSI'
    actual_data_folder, text_file_name = DATASET_MAPPING[dataset_name]
    dataset_path = os.path.join('/app/MixDSemi/data', actual_data_folder)
    
    print(f"Dataset: {dataset_name}")
    print(f"Actual path: {dataset_path}")
    
    # 模拟preprocess.py中的BUSI处理逻辑
    domain_name_mapping = {1: 'benign', 2: 'malignant'}
    total_images_list = {}
    random.seed(1212)  # 与dataloader保持一致的随机种子
    
    for domain_id, domain_name in domain_name_mapping.items():
        domain_path = os.path.join(dataset_path, domain_name)
        if os.path.exists(domain_path):
            # 模拟dataloader的处理逻辑
            imagelist = glob(os.path.join(domain_path, '*.png'))
            imagelist.sort()
            
            # 组织图像-mask对
            domain_data_list = []
            for image_file in imagelist:
                if 'mask' not in image_file:
                    domain_data_list.append([image_file])
                else:
                    if domain_data_list:
                        domain_data_list[-1].append(image_file)
            
            # 80%作为训练集
            test_num = int(len(domain_data_list) * 0.2)
            train_num = len(domain_data_list) - test_num
            train_data = domain_data_list[:train_num]
            
            # 只提取训练集的图像路径（第一个元素是图像，其余是mask）
            train_images = [group[0] for group in train_data]
            total_images_list[domain_name] = train_images
            print(f"  Domain: {domain_name}, Train images: {len(train_images)} (Total groups: {len(domain_data_list)})")
    
    total_train_images = sum(len(images) for images in total_images_list.values())
    print(f"    Total train images in all domains: {total_train_images}")
    
    # 与dataloader逻辑对比
    print(f"\n=== Comparison with dataloader ===")
    print(f"Preprocess (updated): {total_train_images} train images")
    print(f"Expected from analyze_busi.py: 518 train images")
    
    if total_train_images == 518:
        print("✅ BUSI train images align with dataloader!")
    else:
        print("❌ BUSI train images don't align with dataloader!")
    
    # 显示每个域的具体数量对比
    print(f"\nDetailed comparison:")
    expected = {'benign': 350, 'malignant': 168}
    for domain_name, images in total_images_list.items():
        expected_count = expected.get(domain_name, 0)
        actual_count = len(images)
        status = "✅" if actual_count == expected_count else "❌"
        print(f"  {status} {domain_name}: {actual_count} (expected: {expected_count})")

if __name__ == "__main__":
    test_busi_alignment()