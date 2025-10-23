#!/usr/bin/env python3
"""模拟BUSI数据集的处理逻辑，检查训练集图像数量"""

import os
from glob import glob
import random

def analyze_busi_dataset(base_dir='/app/MixDSemi/data/Dataset_BUSI_with_GT', phase='train'):
    """分析BUSI数据集的处理逻辑"""
    
    domain_name = {1:'benign', 2:'malignant'}
    domain = [1, 2]  # 两个域：benign 和 malignant
    
    SEED = 1212
    random.seed(SEED)
    
    total_samples = 0
    
    for i in domain:
        print(f"\n=== Processing domain {i}: {domain_name[i]} ===")
        image_dir = os.path.join(base_dir, domain_name[i] + '/')
        print(f'Loading data from: {image_dir}')
        
        # 获取所有png文件并排序
        imagelist = glob(image_dir + '*.png')
        imagelist.sort()
        
        print(f"Total png files found: {len(imagelist)}")
        
        # 按照dataloader逻辑组织数据：先找图像，再找对应的mask
        domain_data_list = []
        for image_file in imagelist:
            if 'mask' not in image_file:
                domain_data_list.append([image_file])  # 开始一个新的图像组
            else:
                if domain_data_list:  # 如果有正在处理的图像组
                    domain_data_list[-1].append(image_file)  # 将mask添加到最后一个图像组
        
        print(f"Image-mask groups found: {len(domain_data_list)}")
        
        # 显示前几个组的结构
        for j, group in enumerate(domain_data_list[:3]):
            image_name = os.path.basename(group[0])
            mask_count = len(group) - 1
            print(f"  Group {j+1}: {image_name} with {mask_count} masks")
        
        # 80%作为训练集，20%作为测试集
        test_num = int(len(domain_data_list) * 0.2)
        train_num = len(domain_data_list) - test_num
        
        if phase == 'test':
            selected_data = domain_data_list[-test_num:]
            print(f"Test data: {len(selected_data)} images")
        elif phase == 'train':
            selected_data = domain_data_list[:train_num]
            print(f"Train data: {len(selected_data)} images")
        else:
            raise Exception('Unknown split...')
        
        total_samples += len(selected_data)
        
        # 详细显示数据分割
        print(f"Domain {domain_name[i]}:")
        print(f"  Total image groups: {len(domain_data_list)}")
        print(f"  Train images: {train_num}")
        print(f"  Test images: {test_num}")
        print(f"  Selected for {phase}: {len(selected_data)}")
    
    print(f"\n=== Summary ===")
    print(f"Total {phase} images across all domains: {total_samples}")
    
    return total_samples

def compare_with_preprocess_logic():
    """比较preprocess.py中的逻辑"""
    print("=== Comparing with preprocess.py logic ===")
    
    base_dir = '/app/MixDSemi/data/Dataset_BUSI_with_GT'
    domains = ['benign', 'malignant']
    
    total_images = 0
    for domain in domains:
        domain_path = os.path.join(base_dir, domain)
        if os.path.exists(domain_path):
            # 获取所有png文件，排除mask文件 (preprocess.py的逻辑)
            all_files = [f for f in os.listdir(domain_path) if f.endswith('.png')]
            image_files = [f for f in all_files if 'mask' not in f]
            print(f"Preprocess logic - {domain}: {len(image_files)} images (total png: {len(all_files)})")
            total_images += len(image_files)
    
    print(f"Preprocess total images: {total_images}")
    
    # 分析训练集数量
    train_total = analyze_busi_dataset(phase='train')
    test_total = analyze_busi_dataset(phase='test')
    
    print(f"\nDataloader train images: {train_total}")
    print(f"Dataloader test images: {test_total}")
    print(f"Dataloader total images: {train_total + test_total}")
    print(f"Preprocess total images: {total_images}")
    
    if train_total + test_total == total_images:
        print("✅ Image counts match!")
    else:
        print("❌ Image counts don't match!")

if __name__ == "__main__":
    compare_with_preprocess_logic()