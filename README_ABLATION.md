#!/bin/bash
# ============================================================
# 域课程学习消融实验 - 文档和命令快速导航
# ============================================================

cat << 'EOF'

╔════════════════════════════════════════════════════════════════╗
║       域课程学习在现有方法中的消融实验 - 快速导航            ║
╚════════════════════════════════════════════════════════════════╝

## 📚 文档导航

### 1️⃣  快速开始
   📄 ABLATION_SUMMARY.md
   → 核心问题和解决方案
   → 7个推荐的消融实验
   → 预期性能指标

### 2️⃣  命令参考
   📄 COMMANDS_QUICK_REFERENCE.md
   → 所有实验的完整命令
   → 参数对比表
   → 性能分析指标

### 3️⃣  实现指南
   📄 IMPLEMENTATION_GUIDE.md
   → MiDSS修改步骤
   → SynFoC修改步骤
   → 代码集成示例

### 4️⃣  详细实验设计
   📄 ABLATION_EXPERIMENT_GUIDE.md
   → 消融实验详细设计
   → 参数优化建议
   → 故障排除

═══════════════════════════════════════════════════════════════

## 🎯 实验方案总结

┌─────────────────────────────────────────────────────────────┐
│ 核心问题: 如何在MiDSS和SynFoC中集成域课程学习?            │
└─────────────────────────────────────────────────────────────┘

### 方案1: MiDSS + 域课程学习

✅ 保留: 所有MiDSS原有特性
❌ 取消: 无需取消
✨ 添加: 域课程学习、SymGD、LLM、频域增强

命令示例:
python /app/MixDSemi/MiDSS/code/train.py \
  --dataset prostate --lb_domain 1 --lb_num 20 \
  --save_name MiDSS_DC_Full_v1 --gpu 3 --save_model --overwrite \
  --max_iterations 30000 \
  --dc_parts 5 --enable_piecewise_tau --use_symgd \
  --use_freq_aug --use_curr_conf --use_next_conf \
  --llm_model GPT5 --describe_nums 80

预期改进: 70-75% → 76-81%

───────────────────────────────────────────────────────────────

### 方案2: SynFoC + 域课程学习 (UNet Only)

✅ 保留: EMA、CutMix、频域参数
❌ 取消: SAM相关参数(rank、AdamW、ckpt等)
✨ 添加: 域课程学习、SymGD、LLM、频域增强

命令示例:
python /app/MixDSemi/SynFoC/code/train.py \
  --dataset prostate --lb_domain 1 --lb_num 20 \
  --save_name SynFoC_DC_Full_v1 --gpu 3 --save_model --overwrite \
  --max_iterations 30000 \
  --dc_parts 5 --enable_piecewise_tau --use_symgd \
  --use_freq_aug --use_curr_conf --use_next_conf \
  --llm_model GPT5 --describe_nums 80

预期改进: 70-76% → 77-82%

═══════════════════════════════════════════════════════════════

## 🚀 快速执行命令

### 运行方式1: 单个实验 (推荐新手)
cd /app/MixDSemi/SynFoCLIP/code
source COMMANDS_QUICK_REFERENCE.md  # 查看所有命令

### 运行方式2: 批量实验 (推荐高级用户)
bash /app/MixDSemi/SynFoCLIP/code/run_ablation_experiments.sh

### 运行方式3: 自定义 (推荐研究者)
# 从COMMANDS_QUICK_REFERENCE.md复制命令并修改参数

═══════════════════════════════════════════════════════════════

## 📊 7个推荐的消融实验

Exp 1: MiDSS 基线 (70-75%)
       └─ 纯MiDSS，无任何改进

Exp 2: MiDSS + 频域增强 (72-77%)
       └─ 验证频域增强的单独贡献

Exp 3: MiDSS + DC (无LLM) (75-80%)
       └─ 验证DC的核心有效性

Exp 4: MiDSS + DC (完整) (76-81%)
       └─ 完整方法，最好性能

Exp 5: SynFoC 基线 (70-76%)
       └─ SynFoC (UNet only)

Exp 6: SynFoC + DC (无LLM) (76-82%)
       └─ DC在SynFoC上的有效性

Exp 7: SynFoC + DC (完整) (77-82%)
       └─ 完整方法

═══════════════════════════════════════════════════════════════

## 🔑 关键参数速查

### 域课程学习参数
--dc_parts 5                      # 5个课程分区
--dc_distance_mode sqrt_prod      # √(δ_L × δ_U)距离
--enable_piecewise_tau            # 自适应阈值
--tau_min 0.80 --tau_max 0.95     # 阈值从0.80增长到0.95

### 置信度检查参数
--use_curr_conf                   # 检查当前分区置信度
--use_next_conf                   # 检查下一分区置信度
--expand_conf_threshold 0.75      # 置信度≥75%才扩展

### 对称梯度引导参数
--use_symgd --symgd_mode full     # 完整SymGD(UL+LU)
--ul_weight 1.0 --lu_weight 1.0   # CutMix权重
--cons_weight 1.0                 # 一致性权重

### LLM置信度参数
--llm_model GPT5                  # GPT5文本理解
--describe_nums 80                # 80个文本描述
--conf_strategy robust            # 鲁棒置信度策略

### 频域增强参数
--use_freq_aug                    # 启用频域增强
--LB 0.01                         # 低频带比例

═══════════════════════════════════════════════════════════════

## 📁 输出位置

MiDSS实验结果:
/app/MixDSemi/MiDSS/model/prostate/train/{save_name}/

SynFoC实验结果:
/app/MixDSemi/SynFoC/model/prostate/train/{save_name}/

每个实验包含:
├── log.txt                    # 训练日志
├── training_config.json       # 参数记录
├── unet_avg_dice_best_model.pth  # 最佳模型
└── log/                       # TensorBoard日志

═══════════════════════════════════════════════════════════════

## 📈 性能对比矩阵

┌─────────────────────────────────────────────────────────────┐
│ 方法      │ 域课程 │ 频域增强 │ SymGD │ LLM │ 预期Dice    │
├─────────────────────────────────────────────────────────────┤
│ MiDSS基线 │  ❌   │   ❌    │  ❌  │ ❌  │ 70-75%    │
│ +频域     │  ❌   │   ✅    │  ❌  │ ❌  │ 72-77%    │
│ +DC简版   │  ✅   │   ✅    │  ✅  │ ❌  │ 75-80%    │
│ +DC完整   │  ✅   │   ✅    │  ✅  │ ✅  │ 76-81%    │
├─────────────────────────────────────────────────────────────┤
│ SynFoC基线│  ❌   │   ❌    │  ❌  │ ❌  │ 70-76%    │
│ +DC简版   │  ✅   │   ✅    │  ✅  │ ❌  │ 76-82%    │
│ +DC完整   │  ✅   │   ✅    │  ✅  │ ✅  │ 77-82%    │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════

## ⚙️ 代码修改清单

### MiDSS修改 (3个步骤)
☐ 1. 添加域课程学习参数定义
☐ 2. 导入必要模块 (curriculum, conf, tp_ram等)
☐ 3. 集成课程采样器和置信度检查逻辑

### SynFoC修改 (4个步骤)
☐ 1. 添加域课程学习参数定义
☐ 2. 导入必要模块
☐ 3. 注释SAM初始化代码
☐ 4. 集成课程采样器和置信度检查逻辑

详见: IMPLEMENTATION_GUIDE.md

═══════════════════════════════════════════════════════════════

## 🔍 验证修改成功

1. 参数被正确定义
   grep "dc_parts" train.py

2. 模块被正确导入
   grep "DomainDistanceCurriculumSampler" train.py

3. 课程采样器被初始化
   grep "curriculum_sampler = " train.py

4. 快速测试运行 (1000迭代)
   python train.py --dataset prostate --lb_num 20 \
     --max_iterations 1000 --dc_parts 3 --use_curr_conf

═══════════════════════════════════════════════════════════════

## 📞 需要帮助?

查看对应的文档:

Q: 如何快速了解实验设计?
A: 查看 ABLATION_SUMMARY.md

Q: 如何找到具体的命令?
A: 查看 COMMANDS_QUICK_REFERENCE.md

Q: 如何修改代码?
A: 查看 IMPLEMENTATION_GUIDE.md

Q: 如何解决问题?
A: 查看 ABLATION_EXPERIMENT_GUIDE.md 的故障排除部分

═══════════════════════════════════════════════════════════════

最后更新: 2025-10-28
作者: AI Assistant
用途: 消融实验和方法验证

EOF
