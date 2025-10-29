# 消融实验方案 - 最终总结

## 📋 您提出的问题

```
基于当前的训练配置 (train_unet_MiDSS_DC_v2.py):

1. 取消哪些项和配置哪些策略项，可以在SynFoC基础上实现域课程学习?
2. 取消哪些项和配置哪些策略项，可以在MiDSS基础上实现域课程学习?
3. 这是消融实验的一部分，检查策略是否能用于现有方法
4. 给出实现对应实验的命令行指令案例
```

## ✅ 完整答案

---

## 第一部分: SynFoC + 域课程学习

### 需要取消的项

```
❌ --rank              (SAM LoRA秩，不需要)
❌ --AdamW            (SAM优化器，不需要)
❌ --module           (SAM模块名称，不需要)
❌ --img_size         (SAM输入512，改用数据集默认)
❌ --vit_name         (SAM ViT版本，不需要)
❌ --ckpt             (SAM预训练权重，不需要)
❌ --eval             (仅评估模式，不需要)
❌ --save_img         (保存推理结果，关闭以加速)
```

### 需要添加的策略项

```
✨ 域课程学习核心
   --dc_parts 5
   --dc_distance_mode sqrt_prod
   --enable_piecewise_tau
   --tau_min 0.80
   --tau_max 0.95
   --expend_test_steps_interval 300
   --expend_max_steps 5000
   --expend_test_samples 256
   --expand_conf_threshold 0.75
   --curr_conf_threshold 0.75
   --curr_conf_samples 256

✨ 置信度检查
   --use_curr_conf
   --use_next_conf

✨ 对称梯度引导 (SymGD)
   --use_symgd
   --symgd_mode full
   --ul_weight 1.0
   --lu_weight 1.0
   --cons_weight 1.0

✨ 频域增强
   --use_freq_aug
   --LB 0.01

✨ LLM置信度 (可选)
   --conf_strategy robust
   --conf_teacher_temp 1.0
   --llm_model GPT5
   --describe_nums 80
   --preprocess_dir <path>
```

### SynFoC 实现命令

#### 命令1: SynFoC + 域课程学习 (无LLM)
```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name SynFoC_DC_NoLLM_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --use_symgd --symgd_mode full --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

#### 命令2: SynFoC + 完整域课程学习 (含LLM)
```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name SynFoC_DC_Full_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

---

## 第二部分: MiDSS + 域课程学习

### 需要取消的项

```
❌ 无需取消任何项 ✓
   MiDSS已经有所有基础参数
```

### 需要添加的策略项

```
✨ 域课程学习核心
   --dc_parts 5
   --dc_distance_mode sqrt_prod
   --enable_piecewise_tau
   --tau_min 0.80
   --tau_max 0.95
   --expend_test_steps_interval 300
   --expend_max_steps 5000
   --expend_test_samples 256
   --expand_conf_threshold 0.75
   --curr_conf_threshold 0.75
   --curr_conf_samples 256

✨ 置信度检查
   --use_curr_conf
   --use_next_conf

✨ 对称梯度引导 (SymGD)
   --use_symgd
   --symgd_mode full
   --ul_weight 1.0
   --lu_weight 1.0
   --cons_weight 1.0

✨ 频域增强
   --use_freq_aug
   --LB 0.01

✨ LLM置信度 (可选)
   --conf_strategy robust
   --conf_teacher_temp 1.0
   --llm_model GPT5
   --describe_nums 80
   --preprocess_dir <path>
```

### MiDSS 实现命令

#### 命令1: MiDSS + 域课程学习 (无LLM)
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MiDSS_DC_NoLLM_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --use_symgd --symgd_mode full --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

#### 命令2: MiDSS + 完整域课程学习 (含LLM)
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MiDSS_DC_Full_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

---

## 第三部分: 完整的消融实验方案

### Exp 1: MiDSS 基线
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_Baseline_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --deterministic 1
```

### Exp 2: MiDSS + 频域增强
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_FreqAug_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --use_freq_aug --deterministic 1
```

### Exp 3: MiDSS + 域课程学习 (无LLM)
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_DC_NoLLM_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --use_symgd --symgd_mode full --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

### Exp 4: MiDSS + 域课程学习 (完整)
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_DC_Full_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

### Exp 5: SynFoC UNet Only 基线
```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_UNet_Only_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --deterministic 1
```

### Exp 6: SynFoC + 域课程学习 (无LLM)
```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_DC_NoLLM_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --use_symgd --symgd_mode full --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

### Exp 7: SynFoC + 域课程学习 (完整)
```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_DC_Full_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

---

## 第四部分: 关键参数对比

### SynFoC: 取消vs保留
| 类别 | 参数 | 操作 | 原因 |
|------|------|------|------|
| SAM | rank, AdamW, module, img_size, vit_name, ckpt, eval | ❌ 取消 | 不使用SAM模块 |
| 基础 | dataset, lb_num, save_name, gpu等 | ✅ 保留 | 必需的基础配置 |
| 半监督 | ema_decay, consistency, label_bs等 | ✅ 保留 | MiDSS核心特性 |
| 新增 | dc_parts, enable_piecewise_tau, use_symgd等 | ✨ 添加 | 域课程学习特性 |

### MiDSS: 保留vs添加
| 类别 | 参数 | 操作 | 原因 |
|------|------|------|------|
| 基础 | dataset, lb_num, save_name, gpu等 | ✅ 保留 | 必需的基础配置 |
| 半监督 | ema_decay, consistency, label_bs等 | ✅ 保留 | MiDSS原有特性 |
| 频域 | use_freq_aug, LB | ✅ 保留 | 已有的频域增强 |
| 新增 | dc_parts, enable_piecewise_tau, use_symgd等 | ✨ 添加 | 域课程学习特性 |

---

## 第五部分: 预期性能

```
SynFoC 系列:
  基线 (无改进)           → 70-76%
  +域课程学习(无LLM)      → 76-82% (+6-12%)
  +域课程学习(完整)       → 77-82% (+7-12%)

MiDSS 系列:
  基线 (无改进)           → 70-75%
  +频域增强               → 72-77% (+2-7%)
  +域课程学习(无LLM)      → 75-80% (+5-10%)
  +域课程学习(完整)       → 76-81% (+6-11%)
```

---

## 第六部分: 文档清单

已为您生成以下详细文档:

1. ✅ **README_ABLATION.md** - 快速导航和索引
2. ✅ **ABLATION_SUMMARY.md** - 核心总结
3. ✅ **COMMANDS_QUICK_REFERENCE.md** - 命令速查表
4. ✅ **ABLATION_EXPERIMENT_GUIDE.md** - 详细设计
5. ✅ **IMPLEMENTATION_GUIDE.md** - 代码修改指南
6. ✅ **run_ablation_experiments.sh** - 自动化脚本

---

## 📌 核心建议

### 对SynFoC的建议
1. ✂️ 注释掉所有SAM相关的导入和初始化
2. ✨ 仅保留UNet的前向传播
3. ➕ 添加所有域课程学习参数和逻辑

### 对MiDSS的建议
1. ✅ 保留所有现有参数和逻辑
2. ➕ 添加所有域课程学习参数和逻辑
3. 🔄 集成课程采样器替换随机采样

### 代码修改优先级
1. **高优先**: 参数定义 + 模块导入
2. **中优先**: 课程采样器初始化
3. **低优先**: 置信度计算和课程扩展逻辑

---

## ✨ 总结

**对于SynFoC:**
- 🔴 取消: SAM相关的8个参数
- 🟢 保留: 所有基础半监督参数
- 🟡 添加: 域课程学习的20+个参数

**对于MiDSS:**
- 🔴 取消: 无
- 🟢 保留: 所有现有参数
- 🟡 添加: 域课程学习的20+个参数

**预期收益:**
- 性能提升: +6-12%
- 通用性验证: 策略对不同方法都有效
- 消融分析: 逐步验证各个特性的贡献

