#!/usr/bin/env python3
"""
测试新的二维张量保存格式和风格提示分离逻辑
"""

import torch
import os

def test_new_format_and_separation():
    """测试新格式的数据结构和分离逻辑"""
    
    print("🧪 Testing New 2D Tensor Format and Style Separation")
    print("=" * 60)
    
    # 模拟数据
    total_types = 4
    describe_nums = 80
    
    print(f"📋 Configuration:")
    print(f"   total_types: {total_types}")
    print(f"   describe_nums: {describe_nums}")
    print(f"   假设最后一个type是图像风格提示")
    
    # 创建模拟的二维张量数据
    reshaped_logits = torch.randn(total_types, describe_nums)
    print(f"\n📊 Generated test data shape: {reshaped_logits.shape}")
    
    # 测试分离逻辑
    print(f"\n🔀 Testing Separation Logic:")
    
    # 1. 提取风格提示分数 (最后一个type)
    style_scores = reshaped_logits[-1, :]
    print(f"   风格提示分数 [-1, :]: {style_scores.shape}")
    
    # 2. 提取内容类型分数 (除了最后一个)
    content_scores = reshaped_logits[:-1, :]
    print(f"   内容类型分数 [:-1, :]: {content_scores.shape}")
    
    # 3. 错误方式对比
    wrong_content_scores = reshaped_logits[1:, :]
    print(f"   ❌ 错误方式 [1:, :]: {wrong_content_scores.shape}")
    
    # 验证分离的正确性
    print(f"\n✅ Validation:")
    print(f"   原始总类型数: {total_types}")
    print(f"   内容类型数: {content_scores.shape[0]}")
    print(f"   风格类型数: 1")
    print(f"   验证: {content_scores.shape[0] + 1} == {total_types} -> {content_scores.shape[0] + 1 == total_types}")
    
    # 展示每个type的索引
    print(f"\n📝 Type索引说明:")
    for i in range(total_types):
        type_desc = "图像风格提示" if i == total_types - 1 else f"内容类型{i+1}"
        print(f"   Type {i}: {type_desc}")
    
    print(f"\n🎯 分离结果:")
    print(f"   风格提示 (type {total_types-1}): reshaped_logits[-1, :]")
    print(f"   内容类型 (type 0-{total_types-2}): reshaped_logits[:-1, :]")
    
    # 实际应用示例
    print(f"\n🚀 实际应用示例:")
    
    # 计算各种统计量
    content_mean = content_scores.mean(dim=0)  # 内容类型平均分数
    content_std = content_scores.std(dim=0)    # 内容类型标准差
    style_mean = style_scores.mean()           # 风格提示平均分数
    style_max = style_scores.max()             # 风格提示最高分数
    
    print(f"   内容类型平均分数形状: {content_mean.shape}")
    print(f"   内容类型标准差形状: {content_std.shape}")
    print(f"   风格提示平均分数: {style_mean.item():.4f}")
    print(f"   风格提示最高分数: {style_max.item():.4f}")
    
    # 模拟保存和加载测试
    print(f"\n💾 保存和加载测试:")
    
    # 保存测试数据
    test_file = "test_2d_format.pt"
    test_data = {
        'image_001': reshaped_logits,
        'image_002': torch.randn(total_types, describe_nums)
    }
    torch.save(test_data, test_file)
    print(f"   ✅ 测试数据已保存到: {test_file}")
    
    # 加载并验证
    loaded_data = torch.load(test_file, map_location='cpu')
    for img_name, logits in loaded_data.items():
        print(f"   📸 {img_name}: {logits.shape}")
        
        # 分离测试
        img_style = logits[-1, :]
        img_content = logits[:-1, :]
        print(f"      风格: {img_style.shape}, 内容: {img_content.shape}")
    
    # 清理测试文件
    os.remove(test_file)
    print(f"   🗑️ 清理测试文件")
    
    return True

def compare_old_vs_new_format():
    """对比旧格式vs新格式的数据结构"""
    
    print(f"\n" + "=" * 60)
    print("📊 Old Format vs New Format Comparison")
    print("=" * 60)
    
    total_types = 4
    describe_nums = 80
    
    # 旧格式 (一维)
    old_format = torch.randn(total_types * describe_nums)
    print(f"🔴 旧格式 (1D): {old_format.shape}")
    print(f"   数据结构: [type0_desc0, type0_desc1, ..., type0_desc79,")
    print(f"            type1_desc0, type1_desc1, ..., type1_desc79,")
    print(f"            ..., type3_desc79]")
    print(f"   风格提示分离: old_format[{(total_types-1)*describe_nums}:{total_types*describe_nums}]")
    print(f"   内容类型分离: 需要复杂的索引操作")
    
    # 新格式 (二维)
    new_format = old_format.reshape(total_types, describe_nums)
    print(f"\n🟢 新格式 (2D): {new_format.shape}")
    print(f"   数据结构: [[type0_desc0, type0_desc1, ..., type0_desc79],")
    print(f"            [type1_desc0, type1_desc1, ..., type1_desc79],")
    print(f"            [type2_desc0, type2_desc1, ..., type2_desc79],")
    print(f"            [type3_desc0, type3_desc1, ..., type3_desc79]]")
    print(f"   风格提示分离: new_format[-1, :] -> 简洁直观")
    print(f"   内容类型分离: new_format[:-1, :] -> 简洁直观")
    
    print(f"\n✅ 新格式优势:")
    print(f"   1. 🎯 直观的二维结构，类型和描述维度分离")
    print(f"   2. 🔧 简化的索引操作")
    print(f"   3. 📊 便于统计分析 (按类型或按描述)")
    print(f"   4. 🧠 更好的语义表达")

if __name__ == "__main__":
    test_new_format_and_separation()
    compare_old_vs_new_format()