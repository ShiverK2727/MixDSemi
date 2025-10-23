import json
import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from open_clip import get_tokenizer, create_model_and_transforms
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from utils.clip import create_biomedclip_model_and_preprocess_local
from batchgenerators.utilities.file_and_folder_operations import *

# 训练时使用的数据集名称 -> (实际数据文件夹名称, 文本描述文件名称)
DATASET_MAPPING = {
    'ProstateSlice': ('ProstateSlice', 'ProstateSlice'),
    'Fundus': ('Fundus', 'Fundus'), 
    'MNMS': ('mnms', 'MNMS'),
    'BUSI': ('Dataset_BUSI_with_GT', 'BUSI')
}

DATASETS_NAMES = ['ProstateSlice', 'Fundus', 'MNMS', 'BUSI']
# DATASETS_NAMES = ['ProstateSlice']
LLMs = ['gemini', 'GPT5', 'DeepSeek']
DESCRIBE_NUMS = [20, 40, 60, 80]
BATCH_SIZE = 128

def main(data_root, text_root, output_root, model_path, device='cuda'):
    for dataset_name in DATASETS_NAMES:
        print(f"Processing dataset: {dataset_name}")
        
        # 获取实际的数据路径和文本文件名
        actual_data_folder, text_file_name = DATASET_MAPPING[dataset_name]
        dataset_path = os.path.join(data_root, actual_data_folder)
        output_base_path = os.path.join(output_root, dataset_name)  # 输出还是用训练时的名称
        maybe_mkdir_p(output_base_path)

        # Load text descriptions
        text_file = os.path.join(text_root, f"{text_file_name}.json")
        with open(text_file, 'r') as f:
            text_descriptions = json.load(f)

        # 根据数据集配置不同的路径结构
        if dataset_name == 'ProstateSlice':
            # ProstateSlice: base_dir/DOMAIN_NAME/train/image/
            domain_name_mapping = {'BIDMC': 'BIDMC', 'BMC': 'BMC', 'HK': 'HK', 
                                  'I2CVB': 'I2CVB', 'RUNMC': 'RUNMC', 'UCL': 'UCL'}
            total_domains = subdirs(dataset_path, join=False)
            total_images_list = {}
            for domain in total_domains:
                if domain in domain_name_mapping:
                    domain_images = subfiles(os.path.join(dataset_path, domain, 'train', 'image'), suffix='.png', join=True)
                    total_images_list[domain] = domain_images
                    print(f"  Domain: {domain}, Total images: {len(domain_images)}")
            print(f"    Total images in all domains: {sum(len(images) for images in total_images_list.values())}")
            
        elif dataset_name == 'Fundus':
            # Fundus: base_dir/Domain{i}/train/ROIs/image/
            total_images_list = {}
            for domain_id in [1, 2, 3, 4]:
                domain = f'Domain{domain_id}'
                domain_path = os.path.join(dataset_path, domain, 'train', 'ROIs', 'image')
                if os.path.exists(domain_path):
                    domain_images = subfiles(domain_path, suffix='.png', join=True)
                    total_images_list[domain] = domain_images
                    print(f"  Domain: {domain}, Total images: {len(domain_images)}")
            print(f"    Total images in all domains: {sum(len(images) for images in total_images_list.values())}")
            
        elif dataset_name == 'MNMS':
            # MNMS(实际路径mnms): base_dir/VENDOR_NAME/train/image/
            domain_name_mapping = {'vendorA': 'vendorA', 'vendorB': 'vendorB', 
                                  'vendorC': 'vendorC', 'vendorD': 'vendorD'}
            total_images_list = {}
            for domain in domain_name_mapping.keys():
                domain_path = os.path.join(dataset_path, domain, 'train', 'image')
                if os.path.exists(domain_path):
                    domain_images = subfiles(domain_path, suffix='.png', join=True)
                    total_images_list[domain] = domain_images
                    print(f"  Domain: {domain}, Total images: {len(domain_images)}")
            print(f"    Total images in all domains: {sum(len(images) for images in total_images_list.values())}")
            
        elif dataset_name == 'BUSI':
            # BUSI: 只处理训练集数据，按照SynFoC的逻辑
            domain_name_mapping = {1: 'benign', 2: 'malignant'}
            # SynFoC中定义的训练集数量
            train_nums = [350, 168]  # benign: 350, malignant: 168
            total_images_list = {}
            import random
            random.seed(1212)  # 与SynFoC保持一致的随机种子
            
            for domain_id, domain_name in domain_name_mapping.items():
                domain_path = os.path.join(dataset_path, domain_name)
                if os.path.exists(domain_path):
                    # 模拟SynFoC的BUSI数据处理逻辑
                    from glob import glob
                    imagelist = glob(os.path.join(domain_path, '*.png'))
                    imagelist.sort()
                    
                    # 按照SynFoC逻辑组织图像-mask对
                    domain_data_list = []
                    for image_file in imagelist:
                        if 'mask' not in image_file:
                            domain_data_list.append([image_file])
                        else:
                            if domain_data_list:
                                domain_data_list[-1].append(image_file)
                    
                    # 80%作为训练集 (与SynFoC逻辑一致)
                    test_num = int(len(domain_data_list) * 0.2)
                    train_num = len(domain_data_list) - test_num
                    train_data = domain_data_list[:train_num]
                    
                    # 只取前train_nums[domain_id-1]张训练图像 (SynFoC中实际使用的数量)
                    actual_train_num = train_nums[domain_id - 1]
                    selected_train_data = train_data[:actual_train_num]
                    
                    # 提取训练图像路径
                    train_images = [group[0] for group in selected_train_data]
                    total_images_list[domain_name] = train_images
                    print(f"  Domain: {domain_name}, Train images used: {len(train_images)} (Available: {len(train_data)}, Total groups: {len(domain_data_list)})")
            print(f"    Total train images used: {sum(len(images) for images in total_images_list.values())}")
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(model_path, device)
        model.eval()
        for describe_nums in DESCRIBE_NUMS:
            print(f"  Describe nums: {describe_nums}")
            for llm in LLMs:
                total_types = len(text_descriptions[llm])
                selected_texts = []
                total_types = len(text_descriptions[llm])
                for texts in text_descriptions[llm].values():
                    print(f"Texts for current type: {texts[:1]}...")  # 打印前1个文本以供调试
                    print(f"length of texts: {len(texts)}")
                    selected_texts.extend(texts[:describe_nums])
                print(f"  LLM: {llm}, Total types: {total_types}, Selected texts per type: {describe_nums}")
                print(f"    Total selected texts: {len(selected_texts)}")
                text_tokens = tokenizer(selected_texts).to(device)
                # 按批处理每个域的图像
                for domain, image_paths in total_images_list.items():
                    print(f"    Processing domain: {domain}, Total images: {len(image_paths)}")
                    domain_text_image_match = {}
                    
                    # 为当前域创建输出文件夹
                    domain_output_path = os.path.join(output_base_path, domain)
                    maybe_mkdir_p(domain_output_path)
                    
                    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
                        batch_paths = image_paths[i:i+BATCH_SIZE]
                        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
                        images = torch.stack(images).to(device)
                        print(f"      Processing batch {i//BATCH_SIZE + 1}/{(len(image_paths) + BATCH_SIZE - 1)//BATCH_SIZE}, Batch size: {images.size(0)}")
                        print(f"        Image tensor shape: {images.shape}")
                        
                        with torch.no_grad():
                            image_features, text_features, logit_scale = model(images, text_tokens)
                            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                        
                        # 为当前批次的每个图像保存logits
                        for j, image_path in enumerate(batch_paths):
                            image_name = os.path.basename(image_path).replace('.png', '')  # 移除扩展名
                            domain_text_image_match[image_name] = logits[j].cpu()  # 移到CPU以节省GPU内存
                    
                    # 保存当前域、LLM和describe_nums配置的结果
                    output_filename = f"{llm}_{describe_nums}.pt"
                    output_file_path = os.path.join(domain_output_path, output_filename)
                    torch.save(domain_text_image_match, output_file_path)
                    print(f"    Saved {len(domain_text_image_match)} image-text matches to: {output_file_path}")



if __name__ == "__main__":
    data_root = '/app/MixDSemi/data'
    text_root = '/app/MixDSemi/SynFoCLIP/code/text'
    output_root = '/app/MixDSemi/SynFoCLIP/preprocess'
    model_path = '/root/models/BiomedCLIP'
    device = 'cuda:1'
    main(data_root, text_root, output_root, model_path, device)






