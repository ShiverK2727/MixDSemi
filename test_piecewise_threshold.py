#!/usr/bin/env python3
"""
测试分段动态阈值计算逻辑
"""

def get_piecewise_threshold(current_stage, num_stages, tau_min, tau_max):
    """
    Calculate piecewise threshold based on curriculum stage.
    
    Args:
        current_stage: Current curriculum stage (0-indexed)
        num_stages: Total number of curriculum stages
        tau_min: Minimum threshold at stage 0
        tau_max: Maximum threshold at final stage
    
    Returns:
        threshold: Linearly interpolated threshold for current stage
    """
    if num_stages <= 1:
        return tau_max
    
    stage = max(0, min(current_stage, num_stages - 1))
    threshold = tau_min + (stage / (num_stages - 1)) * (tau_max - tau_min)
    
    return threshold


def main():
    print("=" * 80)
    print("分段动态阈值测试")
    print("=" * 80)
    
    # 测试配置
    configs = [
        {"name": "Prostate (推荐)", "num_stages": 5, "tau_min": 0.75, "tau_max": 0.95},
        {"name": "Fundus (标准)", "num_stages": 5, "tau_min": 0.80, "tau_max": 0.95},
        {"name": "MNMS (困难)", "num_stages": 5, "tau_min": 0.85, "tau_max": 0.97},
        {"name": "7阶段示例", "num_stages": 7, "tau_min": 0.75, "tau_max": 0.95},
    ]
    
    for config in configs:
        print(f"\n【{config['name']}】")
        print(f"配置: num_stages={config['num_stages']}, tau_min={config['tau_min']}, tau_max={config['tau_max']}")
        print(f"{'Stage':<8} {'公式':<30} {'阈值':<10} {'样本特征'}")
        print("-" * 80)
        
        for stage in range(config['num_stages']):
            threshold = get_piecewise_threshold(
                current_stage=stage,
                num_stages=config['num_stages'],
                tau_min=config['tau_min'],
                tau_max=config['tau_max']
            )
            
            # 计算公式显示
            formula = f"{config['tau_min']:.2f} + ({stage}/{config['num_stages']-1})×{config['tau_max']-config['tau_min']:.2f}"
            
            # 样本特征描述
            if stage == 0:
                desc = "最简单（接近标注域）"
            elif stage == config['num_stages'] - 1:
                desc = "最困难（远离标注域）"
            else:
                progress = stage / (config['num_stages'] - 1)
                if progress < 0.33:
                    desc = "中等简单"
                elif progress < 0.67:
                    desc = "中等"
                else:
                    desc = "中等困难"
            
            print(f"{stage:<8} {formula:<30} {threshold:<10.4f} {desc}")
    
    # 边界测试
    print("\n" + "=" * 80)
    print("边界测试")
    print("=" * 80)
    
    test_cases = [
        {"stage": -1, "expected_clamped_stage": 0},
        {"stage": 0, "expected": "tau_min"},
        {"stage": 2, "expected": "interpolated"},
        {"stage": 4, "expected": "tau_max"},
        {"stage": 10, "expected_clamped_stage": 4},
    ]
    
    num_stages = 5
    tau_min = 0.80
    tau_max = 0.95
    
    print(f"配置: num_stages={num_stages}, tau_min={tau_min}, tau_max={tau_max}")
    print(f"{'输入Stage':<12} {'有效Stage':<12} {'计算阈值':<12} {'预期行为'}")
    print("-" * 80)
    
    for case in test_cases:
        stage = case['stage']
        threshold = get_piecewise_threshold(stage, num_stages, tau_min, tau_max)
        
        # 确定有效stage
        effective_stage = max(0, min(stage, num_stages - 1))
        
        # 预期描述
        if 'expected_clamped_stage' in case:
            expected = f"Clamped to stage {case['expected_clamped_stage']}"
        else:
            expected = case['expected']
        
        print(f"{stage:<12} {effective_stage:<12} {threshold:<12.4f} {expected}")
    
    # 特殊情况测试
    print("\n" + "=" * 80)
    print("特殊情况测试")
    print("=" * 80)
    
    # 单阶段
    threshold_single = get_piecewise_threshold(0, 1, 0.80, 0.95)
    print(f"单阶段 (num_stages=1): threshold = {threshold_single:.4f} (应等于 tau_max={0.95})")
    
    # 零阶段（非法，但代码应该处理）
    threshold_zero = get_piecewise_threshold(0, 0, 0.80, 0.95)
    print(f"零阶段 (num_stages=0): threshold = {threshold_zero:.4f} (应等于 tau_max={0.95})")
    
    print("\n" + "=" * 80)
    print("✓ 所有测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
