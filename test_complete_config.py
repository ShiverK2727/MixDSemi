#!/usr/bin/env python3
"""测试完整的数据集配置"""

import os
import json

# 训练时使用的数据集名称 -> (实际数据文件夹名称, 文本描述文件名称)
DATASET_MAPPING = {
    'ProstateSlice': ('ProstateSlice', 'ProstateSlice'),
    'Fundus': ('Fundus', 'Fundus'), 
    'MNMS': ('mnms', 'MNMS'),
    'BUSI': ('Dataset_BUSI_with_GT', 'BUSI')
}

def test_complete_config(data_root='/app/MixDSemi/data', 
                        text_root='/app/MixDSemi/SynFoCLIP/code/text'):
    """测试完整的数据集配置"""
    
    for dataset_name in DATASET_MAPPING.keys():
        print(f"\n=== Testing {dataset_name} ===")
        
        # 获取实际的数据路径和文本文件名
        actual_data_folder, text_file_name = DATASET_MAPPING[dataset_name]
        dataset_path = os.path.join(data_root, actual_data_folder)
        text_file = os.path.join(text_root, f"{text_file_name}.json")
        
        # 检查数据路径
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path does not exist: {dataset_path}")
            continue
        print(f"✅ Dataset path exists: {dataset_path}")
        
        # 检查文本描述文件
        if os.path.exists(text_file):
            print(f"✅ Text file exists: {text_file}")
            try:
                with open(text_file, 'r') as f:
                    text_descriptions = json.load(f)
                print(f"  📄 Text file loaded successfully with {len(text_descriptions)} LLM types")
            except Exception as e:
                print(f"❌ Error loading text file: {e}")
        else:
            print(f"❌ Text file missing: {text_file}")
        
        # 检查图像路径配置
        print(f"📁 Checking image paths for {dataset_name}:")
        
        if dataset_name == 'ProstateSlice':
            domain_folders = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
            for domain in domain_folders:
                image_path = os.path.join(dataset_path, domain, 'train', 'image')
                if os.path.exists(image_path):
                    png_files = [f for f in os.listdir(image_path) if f.endswith('.png')]
                    print(f"  ✅ {domain}: {len(png_files)} images")
                else:
                    print(f"  ❌ {domain}: path not found - {image_path}")
                    
        elif dataset_name == 'Fundus':
            for domain_id in [1, 2, 3, 4]:
                domain = f'Domain{domain_id}'
                image_path = os.path.join(dataset_path, domain, 'train', 'ROIs', 'image')
                if os.path.exists(image_path):
                    png_files = [f for f in os.listdir(image_path) if f.endswith('.png')]
                    print(f"  ✅ {domain}: {len(png_files)} images")
                else:
                    print(f"  ❌ {domain}: path not found - {image_path}")
                    
        elif dataset_name == 'MNMS':
            vendors = ['vendorA', 'vendorB', 'vendorC', 'vendorD']
            for vendor in vendors:
                image_path = os.path.join(dataset_path, vendor, 'train', 'image')
                if os.path.exists(image_path):
                    png_files = [f for f in os.listdir(image_path) if f.endswith('.png')]
                    print(f"  ✅ {vendor}: {len(png_files)} images")
                else:
                    print(f"  ❌ {vendor}: path not found - {image_path}")
                    
        elif dataset_name == 'BUSI':
            # 使用与preprocess.py相同的逻辑
            domain_name_mapping = {1: 'benign', 2: 'malignant'}
            train_nums = [350, 168]  # SynFoC中定义的训练集数量
            import random
            random.seed(1212)
            
            for domain_id, domain_name in domain_name_mapping.items():
                domain_path = os.path.join(dataset_path, domain_name)
                if os.path.exists(domain_path):
                    # 模拟SynFoC的处理逻辑
                    from glob import glob
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
                    
                    # 实际使用的训练数量
                    actual_train_num = train_nums[domain_id - 1]
                    selected_train_data = train_data[:actual_train_num]
                    train_images = [group[0] for group in selected_train_data]
                    
                    print(f"  ✅ {domain_name}: {len(train_images)} train images used (available: {len(train_data)}, total: {len(domain_data_list)})")
                else:
                    print(f"  ❌ {domain_name}: path not found - {domain_path}")

if __name__ == "__main__":
    test_complete_config()