import json
import os

def check_json_order():
    """检查JSON文件中键值对的顺序和索引"""
    
    json_path = '/app/MixDSemi/SynFoCLIP/code/text/BUSI.json'
    
    with open(json_path, 'r') as f:
        text_descriptions = json.load(f)
    
    print("=== JSON结构分析 ===")
    
    # 遍历每个LLM
    for llm_idx, (llm, llm_data) in enumerate(text_descriptions.items()):
        print(f"\nLLM {llm_idx}: {llm}")
        print(f"Keys in {llm}: {list(llm_data.keys())}")
        
        # 检查键的顺序
        keys_list = list(llm_data.keys())
        print(f"Keys order: {keys_list}")
        
        # 分析索引
        for key_idx, (key, texts) in enumerate(llm_data.items()):
            print(f"  Index {key_idx}: '{key}' -> {len(texts)} texts")
            if isinstance(texts, list) and len(texts) > 0:
                print(f"    First text: {texts[0][:50]}...")
        
        # 检查style是否在最后
        if 'style' in keys_list:
            style_index = keys_list.index('style')
            print(f"  'style' is at index {style_index} (0-based)")
            print(f"  Total keys: {len(keys_list)}")
            print(f"  Is 'style' last? {style_index == len(keys_list) - 1}")
        else:
            print(f"  No 'style' key found in {llm}")

def simulate_text_processing():
    """模拟preprocess.py中的文本处理过程"""
    
    print("\n\n=== 模拟文本处理过程 ===")
    
    json_path = '/app/MixDSemi/SynFoCLIP/code/text/BUSI.json'
    
    with open(json_path, 'r') as f:
        text_descriptions = json.load(f)
    
    DESCRIBE_NUMS = [80]  # 使用一个数值进行测试
    LLMs = ['gemini']     # 使用一个LLM进行测试
    
    for llm in LLMs:
        print(f"\n处理LLM: {llm}")
        
        # 模拟preprocess.py中的处理逻辑
        total_types = len(text_descriptions[llm])
        print(f"total_types = len(text_descriptions['{llm}']) = {total_types}")
        
        # 检查keys的顺序
        keys_order = list(text_descriptions[llm].keys())
        print(f"Keys order: {keys_order}")
        
        selected_texts = []
        type_idx = 0
        
        # 按照JSON中的顺序遍历
        for type_name, texts in text_descriptions[llm].items():
            print(f"  Type {type_idx}: '{type_name}' -> {len(texts)} texts")
            selected_texts.extend(texts[:DESCRIBE_NUMS[0]])  # 选择前DESCRIBE_NUMS个文本
            type_idx += 1
        
        print(f"Total selected texts: {len(selected_texts)}")
        print(f"Expected tensor shape after reshape: [{total_types}, {DESCRIBE_NUMS[0]}]")
        
        # 模拟reshape操作
        import torch
        dummy_logits = torch.randn(total_types * DESCRIBE_NUMS[0])  # 模拟logits
        reshaped_logits = dummy_logits.reshape(total_types, DESCRIBE_NUMS[0])
        
        print(f"Reshaped tensor shape: {reshaped_logits.shape}")
        
        # 分析索引对应关系
        print("\n索引对应关系:")
        for idx, (type_name, _) in enumerate(text_descriptions[llm].items()):
            print(f"  reshaped_logits[{idx}, :] -> '{type_name}' scores")
        
        # 检查style的位置
        if 'style' in keys_order:
            style_idx = keys_order.index('style')
            print(f"\n'style' 在索引 {style_idx}")
            print(f"访问style分数: reshaped_logits[{style_idx}, :]")
            
            # 分离content和style
            if style_idx == len(keys_order) - 1:  # style在最后
                print(f"Content types (前{style_idx}个): reshaped_logits[:{style_idx}, :] 或 reshaped_logits[:-1, :]")
                print(f"Style type (最后1个): reshaped_logits[{style_idx}:, :] 或 reshaped_logits[-1:, :] 或 reshaped_logits[-1, :]")
            else:
                print(f"Warning: 'style'不在最后位置！当前位置: {style_idx}")

if __name__ == "__main__":
    check_json_order()
    simulate_text_processing()