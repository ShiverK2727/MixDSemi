import json
import torch

def analyze_text_processing():
    """详细分析文本处理过程中的索引变化"""
    
    # 测试所有数据集
    datasets = {
        'ProstateSlice': '/app/MixDSemi/SynFoCLIP/code/text/ProstateSlice.json',
        'Fundus': '/app/MixDSemi/SynFoCLIP/code/text/Fundus.json', 
        'MNMS': '/app/MixDSemi/SynFoCLIP/code/text/MNMS.json',
        'BUSI': '/app/MixDSemi/SynFoCLIP/code/text/BUSI.json'
    }
    
    for dataset_name, json_path in datasets.items():
        print(f"\n{'='*50}")
        print(f"数据集: {dataset_name}")
        print('='*50)
        
        with open(json_path, 'r') as f:
            text_descriptions = json.load(f)
        
        # 分析第一个LLM
        first_llm = list(text_descriptions.keys())[0]
        llm_data = text_descriptions[first_llm]
        
        print(f"LLM: {first_llm}")
        print(f"Keys: {list(llm_data.keys())}")
        
        # 模拟preprocess.py中的处理过程
        total_types = len(llm_data)
        print(f"total_types = {total_types}")
        
        # 检查键的顺序和对应关系
        selected_texts = []
        describe_nums = 20  # 使用较小的数值测试
        
        print(f"\n文本选择过程:")
        type_names = []
        for i, (type_name, texts) in enumerate(llm_data.items()):
            print(f"  索引 {i}: '{type_name}' -> 选择前{describe_nums}个文本")
            type_names.append(type_name)
            selected_texts.extend(texts[:describe_nums])
        
        print(f"\n总共选择文本: {len(selected_texts)}")
        print(f"预期tensor形状: [{total_types}, {describe_nums}]")
        
        # 模拟reshape操作
        dummy_logits = torch.randn(total_types * describe_nums)
        reshaped_logits = dummy_logits.reshape(total_types, describe_nums)
        
        print(f"实际tensor形状: {reshaped_logits.shape}")
        
        # 分析索引对应关系
        print(f"\n索引对应关系:")
        for idx, type_name in enumerate(type_names):
            print(f"  reshaped_logits[{idx}, :] -> '{type_name}' scores")
        
        # 检查style的位置
        if 'style' in type_names:
            style_idx = type_names.index('style')
            print(f"\nstyle分析:")
            print(f"  'style'在索引 {style_idx}")
            print(f"  是否在最后位置: {style_idx == len(type_names) - 1}")
            
            if style_idx == len(type_names) - 1:
                print(f"  访问style分数: reshaped_logits[-1, :] 或 reshaped_logits[{style_idx}, :]")
                print(f"  访问content分数: reshaped_logits[:-1, :] 或 reshaped_logits[:{style_idx}, :]")
                
                # 验证分离操作
                style_scores = reshaped_logits[-1, :]  # 最后一行
                content_scores = reshaped_logits[:-1, :]  # 除了最后一行的所有行
                
                print(f"\n验证分离操作:")
                print(f"  style_scores shape: {style_scores.shape}")  # 应该是[describe_nums]
                print(f"  content_scores shape: {content_scores.shape}")  # 应该是[total_types-1, describe_nums]
                
                print(f"  预期style shape: [{describe_nums}]")
                print(f"  预期content shape: [{total_types-1}, {describe_nums}]")
                
                # 检查是否匹配
                style_match = list(style_scores.shape) == [describe_nums]
                content_match = list(content_scores.shape) == [total_types-1, describe_nums]
                
                print(f"  style形状匹配: {style_match}")
                print(f"  content形状匹配: {content_match}")
            else:
                print(f"  WARNING: 'style'不在最后位置！")
        else:
            print(f"\n注意: 该数据集没有'style'类型")

if __name__ == "__main__":
    analyze_text_processing()