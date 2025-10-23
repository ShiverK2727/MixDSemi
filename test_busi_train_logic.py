#!/usr/bin/env python3
"""测试修正后的BUSI训练集逻辑"""

import os
import sys
sys.path.append('/app/MixDSemi/SynFoCLIP/code')

from batchgenerators.utilities.file_and_folder_operations import *
from glob import glob
import random

def test_busi_train_logic():
    """测试BUSI训练集处理逻辑"""
    
    print("=== Testing BUSI train set logic ===")
    
    dataset_path = '/app/MixDSemi/data/Dataset_BUSI_with_GT'
    domain_name_mapping = {1: 'benign', 2: 'malignant'}
    train_nums = [350, 168]  # SynFoC中定义的训练集数量
    
    random.seed(1212)  # 与SynFoC保持一致
    
    total_train_images = 0
    
    for domain_id, domain_name in domain_name_mapping.items():
        domain_path = os.path.join(dataset_path, domain_name)
        print(f"\nProcessing domain {domain_id}: {domain_name}")
        
        # 模拟SynFoC的处理逻辑
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
        
        # 80/20划分
        test_num = int(len(domain_data_list) * 0.2)
        train_num = len(domain_data_list) - test_num
        train_data = domain_data_list[:train_num]
        
        # 只取前train_nums[domain_id-1]张
        actual_train_num = train_nums[domain_id - 1]
        selected_train_data = train_data[:actual_train_num]
        
        # 提取图像路径
        train_images = [group[0] for group in selected_train_data]
        
        print(f"  Total image-mask groups: {len(domain_data_list)}")
        print(f"  Available for training (80%): {len(train_data)}")
        print(f"  Actually used for training: {len(train_images)}")
        print(f"  SynFoC domain_len: {actual_train_num}")
        
        if len(train_images) == actual_train_num:
            print(f"  ✅ Matches SynFoC training count")
        else:
            print(f"  ❌ Doesn't match SynFoC training count")
        
        total_train_images += len(train_images)
    
    print(f"\n=== Summary ===")
    print(f"Total training images: {total_train_images}")
    print(f"Expected total (350+168): {sum(train_nums)}")
    
    if total_train_images == sum(train_nums):
        print("✅ Total matches SynFoC expectation")
    else:
        print("❌ Total doesn't match SynFoC expectation")

def compare_with_synfoc_domain_len():
    """与SynFoC train.py中的domain_len对比"""
    print("\n=== Comparison with SynFoC domain_len ===")
    
    # SynFoC中的定义
    synfoc_busi_domain_len = [350, 168]
    
    print("SynFoC domain_len for BUSI:")
    print(f"  benign (domain 1): {synfoc_busi_domain_len[0]}")
    print(f"  malignant (domain 2): {synfoc_busi_domain_len[1]}")
    print(f"  Total: {sum(synfoc_busi_domain_len)}")
    
    print("\nThis means in SynFoC training:")
    print("  - benign域使用前350张训练图像")
    print("  - malignant域使用前168张训练图像")
    print("  - 总共518张图像用于半监督学习")

if __name__ == "__main__":
    test_busi_train_logic()
    compare_with_synfoc_domain_len()