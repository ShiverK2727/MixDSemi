# 消融实验命令速查表

## 🎯 快速导航

### 实验目标
验证域课程学习（Domain Curriculum）在现有半监督学习方法（MiDSS、SynFoC）中的有效性

### 数据集
- **前列腺** (Prostate): 6个域，最少标注20个样本
- **训练迭代**: 30,000 iterations
- **评估间隔**: 500 iterations

---

## 📋 实验命令速查

### MiDSS 系列实验

#### Exp 1.1: MiDSS 基线 (无任何改进)
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_Baseline_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --deterministic 1
```

**特点**: 纯MiDSS方法，无域课程，无频域增强
**预期Dice**: ~70-75%

---

#### Exp 1.2: MiDSS + 频域增强
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_FreqAug_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --use_freq_aug --deterministic 1
```

**特点**: 仅添加频域增强
**参数变化**: `+--use_freq_aug`
**预期Dice**: ~72-77%

---

#### Exp 1.3: MiDSS + 域课程学习 (简化版，无LLM)
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_DC_Simple_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --use_symgd --symgd_mode full --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

**特点**: DC核心特性，无LLM辅助
**关键参数**:
- `--dc_parts 5` - 5个课程分区
- `--enable_piecewise_tau` - 自适应阈值
- `--use_symgd` - 对称梯度引导
- `--use_freq_aug` - 频域增强

**预期Dice**: ~75-80%

---

#### Exp 1.4: MiDSS + 完整域课程学习 (含LLM)
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_DC_Full_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

**特点**: 完整DC+LLM
**额外参数**:
- `--llm_model GPT5` - 使用GPT5生成置信度
- `--describe_nums 80` - 80个文本描述
- `--conf_strategy robust` - 鲁棒置信度策略

**预期Dice**: ~76-81%

---

### SynFoC 系列实验

#### Exp 2.1: SynFoC UNet Only 基线 (不含SAM)
```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_UNet_Only_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --deterministic 1
```

**特点**: SynFoC但禁用SAM，仅UNet
**预期Dice**: ~70-76%

**注意**: 需要在train.py中注释掉SAM相关初始化

---

#### Exp 2.2: SynFoC + 域课程学习 (简化版，无LLM)
```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_DC_Simple_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --use_symgd --symgd_mode full --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

**特点**: SynFoC + DC (UNet only)
**预期Dice**: ~76-82%

---

#### Exp 2.3: SynFoC + 完整域课程学习 (含LLM)
```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_DC_Full_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

**特点**: SynFoC + 完整DC + LLM
**预期Dice**: ~77-82%

---

## 📊 实验对比矩阵

| 实验ID | 方法 | 域课程 | 频域增强 | SymGD | LLM | 预期Dice | 备注 |
|-------|------|-------|--------|-------|-----|---------|------|
| 1.1 | MiDSS | ❌ | ❌ | ❌ | ❌ | 70-75% | 基线 |
| 1.2 | MiDSS | ❌ | ✅ | ❌ | ❌ | 72-77% | +频域 |
| 1.3 | MiDSS | ✅ | ✅ | ✅ | ❌ | 75-80% | +DC简版 |
| 1.4 | MiDSS | ✅ | ✅ | ✅ | ✅ | 76-81% | +DC完整 |
| 2.1 | SynFoC | ❌ | ❌ | ❌ | ❌ | 70-76% | UNet基线 |
| 2.2 | SynFoC | ✅ | ✅ | ✅ | ❌ | 76-82% | +DC简版 |
| 2.3 | SynFoC | ✅ | ✅ | ✅ | ✅ | 77-82% | +DC完整 |

---

## 🔑 关键参数说明

### 域课程学习参数
- `--dc_parts 5` - 将未标注数据分为5个难度递增的分区
- `--dc_distance_mode sqrt_prod` - 使用√(δ_L × δ_U)计算域距离
- `--enable_piecewise_tau` - 阈值随课程阶段递增（0.80→0.95）
- `--expend_test_steps_interval 300` - 每300步检查是否扩展课程
- `--expend_max_steps 5000` - 最多等待5000步后强制扩展

### 置信度检查参数
- `--use_curr_conf` - 检查当前分区置信度
- `--use_next_conf` - 检查下一分区置信度
- `--curr_conf_threshold 0.75` - 当前分区需≥75%置信度
- `--expand_conf_threshold 0.75` - 下一分区需≥75%置信度

### 对称梯度引导参数
- `--use_symgd` - 启用对称梯度引导
- `--symgd_mode full` - UL和LU混合都参与
- `--ul_weight 1.0` - UL(U背景+L前景)损失权重
- `--lu_weight 1.0` - LU(L背景+U前景)损失权重
- `--cons_weight 1.0` - 一致性损失权重

### 频域增强参数
- `--use_freq_aug` - 启用频域增强
- `--LB 0.01` - 低频带比例

### LLM置信度参数
- `--llm_model GPT5` - 使用GPT5生成文本描述
- `--describe_nums 80` - 80个文本描述用于置信度评估
- `--conf_strategy robust` - 鲁棒置信度策略
- `--conf_teacher_temp 1.0` - 教师模型软化温度

---

## 📁 结果输出位置

所有实验结果保存在:
```
/app/MixDSemi/MiDSS/model/prostate/train/{save_name}/
/app/MixDSemi/SynFoC/model/prostate/train/{save_name}/
```

每个实验会生成:
- `log.txt` - 训练日志
- `training_config.json` - 参数配置记录
- `unet_avg_dice_best_model.pth` - 最佳模型权重
- `log/` - TensorBoard日志

---

## 🚀 运行建议

### 单个实验运行
```bash
# 运行一个实验
bash run_ablation_experiments.sh | head -n 50
```

### 批量运行所有实验
```bash
# 修改脚本中的GPU编号，然后:
bash run_ablation_experiments.sh
```

### 顺序运行（推荐）
```bash
# 先运行基线
python /app/MixDSemi/MiDSS/code/train.py ... # Exp 1.1
# 检查基线性能后，再运行改进版本
```

---

## ⚠️ 注意事项

1. **LLM预处理**: 确保LLM生成的分数张量已保存到指定目录
2. **内存**: 某些配置可能需要调整批大小
3. **时间**: 每个实验约2-4小时（GPU时间）
4. **重现性**: 所有实验使用 `--deterministic 1 --seed 1337`
5. **日志**: 查看 `log.txt` 了解训练进度和课程扩展时机

---

## 📈 性能分析指标

### 每个实验应记录:
- 初始Dice: 第一次评估时的性能
- 最终Dice: 训练完成时的性能
- 改进幅度: 最终 - 初始
- 课程扩展时机: 在哪些迭代点扩展
- 收敛速度: 达到稳定性能需要多少迭代

### 对比分析:
```
改进幅度 = (DC性能 - 基线性能) / 基线性能 × 100%
```

---

## 🔍 快速诊断

如果性能不如预期，检查:

1. **是否加载了LLM分数**: `--preprocess_dir` 是否正确
2. **课程是否在扩展**: 查看log中的"Curriculum expansion"信息
3. **置信度检查是否启用**: 确认 `--use_curr_conf --use_next_conf` 都指定
4. **阈值是否过高**: 尝试降低 `--expand_conf_threshold` 到0.65

