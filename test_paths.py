#!/usr/bin/env python3
"""测试各个数据集的路径配置是否正确"""

import os
from batchgenerators.utilities.file_and_folder_operations import *

def test_dataset_paths(data_root='/app/MixDSemi/data'):
    """测试四个数据集的路径配置"""
    datasets = ['ProstateSlice', 'Fundus', 'mnms', 'Dataset_BUSI_with_GT']
    
    for dataset_name in datasets:
        print(f"\n=== Testing {dataset_name} ===")
        dataset_path = os.path.join(data_root, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path does not exist: {dataset_path}")
            continue
        
        print(f"✅ Dataset path exists: {dataset_path}")
        
        if dataset_name == 'ProstateSlice':
            # ProstateSlice: base_dir/DOMAIN_NAME/train/image/
            domain_name_mapping = {'BIDMC': 'BIDMC', 'BMC': 'BMC', 'HK': 'HK', 
                                  'I2CVB': 'I2CVB', 'RUNMC': 'RUNMC', 'UCL': 'UCL'}
            total_domains = subdirs(dataset_path, join=False)
            print(f"Available domains: {total_domains}")
            
            for domain in total_domains:
                if domain in domain_name_mapping:
                    image_path = os.path.join(dataset_path, domain, 'train', 'image')
                    if os.path.exists(image_path):
                        domain_images = subfiles(image_path, suffix='.png', join=True)
                        print(f"  ✅ {domain}: {len(domain_images)} images")
                    else:
                        print(f"  ❌ {domain}: path not found - {image_path}")
                        
        elif dataset_name == 'Fundus':
            # Fundus: base_dir/Domain{i}/train/ROIs/image/
            for domain_id in [1, 2, 3, 4]:
                domain = f'Domain{domain_id}'
                domain_path = os.path.join(dataset_path, domain, 'train', 'ROIs', 'image')
                if os.path.exists(domain_path):
                    domain_images = subfiles(domain_path, suffix='.png', join=True)
                    print(f"  ✅ {domain}: {len(domain_images)} images")
                else:
                    print(f"  ❌ {domain}: path not found - {domain_path}")
                    
        elif dataset_name == 'mnms':
            # MNMS: base_dir/VENDOR_NAME/train/image/
            domain_name_mapping = {'vendorA': 'vendorA', 'vendorB': 'vendorB', 
                                  'vendorC': 'vendorC', 'vendorD': 'vendorD'}
            for domain in domain_name_mapping.keys():
                domain_path = os.path.join(dataset_path, domain, 'train', 'image')
                if os.path.exists(domain_path):
                    domain_images = subfiles(domain_path, suffix='.png', join=True)
                    print(f"  ✅ {domain}: {len(domain_images)} images")
                else:
                    print(f"  ❌ {domain}: path not found - {domain_path}")
                    
        elif dataset_name == 'Dataset_BUSI_with_GT':
            # BUSI: base_dir/DOMAIN_NAME/ (需要排除mask文件)
            domain_name_mapping = {'benign': 'benign', 'malignant': 'malignant'}
            for domain in domain_name_mapping.keys():
                domain_path = os.path.join(dataset_path, domain)
                if os.path.exists(domain_path):
                    # 获取所有png文件，排除mask文件
                    all_files = subfiles(domain_path, suffix='.png', join=True)
                    domain_images = [f for f in all_files if 'mask' not in os.path.basename(f)]
                    print(f"  ✅ {domain}: {len(domain_images)} images (total png: {len(all_files)})")
                else:
                    print(f"  ❌ {domain}: path not found - {domain_path}")

if __name__ == "__main__":
    test_dataset_paths()