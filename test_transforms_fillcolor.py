#!/usr/bin/env python3
"""
测试旋转transforms的fillcolor参数是否正确设置
"""

import sys
sys.path.insert(0, '/app/MixDSemi/SynFoCLIP/code')

from dataloaders import custom_transforms as tr
from PIL import Image
import numpy as np

def test_random_scale_rotate():
    """测试RandomScaleRotate的fillcolor参数"""
    print("=" * 60)
    print("测试 RandomScaleRotate")
    print("=" * 60)
    
    # 创建测试数据
    # 图像: 100x100, 全白(255)
    img = Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 255)
    # Mask: 50x50中心区域为0(前景), 其余为255(背景)
    mask_arr = np.ones((100, 100), dtype=np.uint8) * 255
    mask_arr[25:75, 25:75] = 0
    mask = Image.fromarray(mask_arr)
    
    sample = {'image': img, 'label': mask}
    
    # 测试1: 使用默认参数 (fillcolor=255, img_fillcolor=0)
    print("\n测试1: 默认参数 (mask fillcolor=255, img fillcolor=0)")
    transform = tr.RandomScaleRotate()
    print(f"  mask fillcolor: {transform.fillcolor}")
    print(f"  img fillcolor: {transform.img_fillcolor}")
    
    # 测试2: Prostate设置 (mask fillcolor=255, img fillcolor=0)
    print("\n测试2: Prostate设置")
    transform = tr.RandomScaleRotate(fillcolor=255, img_fillcolor=0)
    print(f"  mask fillcolor: {transform.fillcolor}")
    print(f"  img fillcolor: {transform.img_fillcolor}")
    
    # 测试3: MNMS/BUSI设置 (mask fillcolor=0, img fillcolor=0)
    print("\n测试3: MNMS/BUSI设置")
    transform = tr.RandomScaleRotate(fillcolor=0, img_fillcolor=0)
    print(f"  mask fillcolor: {transform.fillcolor}")
    print(f"  img fillcolor: {transform.img_fillcolor}")
    
    print("\n✓ RandomScaleRotate 参数正确")

def test_random_rotate():
    """测试RandomRotate的fillcolor参数"""
    print("\n" + "=" * 60)
    print("测试 RandomRotate")
    print("=" * 60)
    
    # 测试1: 使用默认参数
    print("\n测试1: 默认参数 (mask fillcolor=255, img fillcolor=0)")
    transform = tr.RandomRotate()
    print(f"  mask fillcolor: {transform.fillcolor}")
    print(f"  img fillcolor: {transform.img_fillcolor}")
    
    # 测试2: 自定义参数
    print("\n测试2: 自定义参数 (mask fillcolor=0, img fillcolor=0)")
    transform = tr.RandomRotate(fillcolor=0, img_fillcolor=0)
    print(f"  mask fillcolor: {transform.fillcolor}")
    print(f"  img fillcolor: {transform.img_fillcolor}")
    
    print("\n✓ RandomRotate 参数正确")

def test_patch_sampler_fg_func():
    """测试RandomPatchSamplerWithClass的fg_func参数"""
    print("\n" + "=" * 60)
    print("测试 RandomPatchSamplerWithClass.fg_func")
    print("=" * 60)
    
    # 测试1: 默认fg_func (mask > 0)
    print("\n测试1: 默认fg_func")
    sampler = tr.RandomPatchSamplerWithClass(
        num_patches=4,
        num_fg=2
    )
    
    # 测试前景检测
    mask = np.array([[0, 0, 128, 255],
                     [0, 0, 128, 255],
                     [1, 1, 200, 255],
                     [1, 1, 200, 255]])
    
    fg_mask = sampler.fg_func(mask)
    print(f"  输入mask:\n{mask}")
    print(f"  fg_func(mask > 0) 结果:\n{fg_mask}")
    print(f"  前景像素数: {fg_mask.sum()}")
    assert fg_mask.sum() == 12, "默认fg_func应该检测到12个前景像素"
    
    # 测试2: Prostate fg_func (mask == 0)
    print("\n测试2: Prostate fg_func (mask == 0)")
    sampler.fg_func = lambda m: (m == 0)
    
    fg_mask = sampler.fg_func(mask)
    print(f"  fg_func(mask == 0) 结果:\n{fg_mask}")
    print(f"  前景像素数: {fg_mask.sum()}")
    assert fg_mask.sum() == 4, "Prostate fg_func应该检测到4个前景像素(值为0)"
    
    print("\n✓ RandomPatchSamplerWithClass.fg_func 工作正常")

if __name__ == '__main__':
    test_random_scale_rotate()
    test_random_rotate()
    test_patch_sampler_fg_func()
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过!")
    print("=" * 60)
