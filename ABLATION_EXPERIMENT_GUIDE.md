# 域课程学习消融实验指南

## 概述
基于当前的 `train_unet_MiDSS_DC_v2.py` 配置，本文档提供如何在以下两个基线方法上集成域课程学习的消融实验方案：
1. **SynFoC**: `/app/MixDSemi/SynFoC/code/train.py` (包含SAM模块)
2. **MiDSS**: `/app/MixDSemi/MiDSS/code/train.py` (不含SAM模块)

---

## 第一部分：SynFoC + 域课程学习（不涉及SAM）

### 消融实验方案：SynFoC baseline

**目标**: 在SynFoC基础上，关闭SAM模块，仅保留UNet+EMA+半监督学习，验证基础效果

**需要取消的项**:
```
--rank (SAM LoRA相关，关闭SAM模块)
--AdamW (SAM优化器相关，关闭SAM模块)
--module (SAM模块名称，关闭SAM模块)
--img_size (SAM输入尺寸512，改用数据集默认patch_size)
--vit_name (SAM ViT版本，关闭SAM模块)
--ckpt (SAM预训练权重路径，关闭SAM模块)
--eval (仅评估模式，关闭)
--save_img (保存推理结果图像，关闭以加速)
```

**需要添加的域课程学习参数**:
```
--dc_parts 5                           # 课程分区数
--dc_distance_mode sqrt_prod          # 距离度量方式
--enable_piecewise_tau               # 启用分段自适应阈值
--tau_min 0.80                        # 最小阈值
--tau_max 0.95                        # 最大阈值
--expend_test_steps_interval 300      # 测试间隔
--expend_max_steps 5000               # 最大扩展步数
--expend_test_samples 256             # 测试样本数
--expand_conf_threshold 0.75          # 扩展置信度阈值
--curr_conf_threshold 0.75            # 当前分区置信度阈值
--curr_conf_samples 256               # 当前分区样本数
--use_symgd                           # 启用对称梯度引导
--symgd_mode full                     # 完整SymGD模式
--conf_strategy robust                # 置信度策略
--conf_teacher_temp 1.0               # 教师温度
--llm_model GPT5                      # LLM模型
--describe_nums 80                    # 描述数量
--use_freq_aug                        # 启用频域增强
--use_curr_conf                       # 启用当前分区检查
--use_next_conf                       # 启用下一分区检查
```

**SynFoC + 域课程学习 - 前列腺数据集命令**:

```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name SynFoC_UNet_Only_DC_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

**⚠️ 注意**: SynFoC的train.py目前可能不支持所有域课程学习参数，需要先进行以下修改：
1. 在 `/app/MixDSemi/SynFoC/code/train.py` 中添加域课程学习相关参数定义
2. 注释或移除SAM相关的初始化代码
3. 确保使用UNet而非SAM进行分割

---

## 第二部分：MiDSS + 域课程学习

### 消融实验方案：MiDSS baseline

**目标**: 在MiDSS基础上添加域课程学习，验证其对现有方法的改进

**需要取消的项**:
```
--load (加载预训练权重模式，用新训练替代)
--load_path (预训练权重路径)
```

**需要添加的域课程学习参数**:
```
--dc_parts 5                           # 课程分区数
--dc_distance_mode sqrt_prod          # 距离度量方式
--enable_piecewise_tau               # 启用分段自适应阈值
--tau_min 0.80                        # 最小阈值
--tau_max 0.95                        # 最大阈值
--expend_test_steps_interval 300      # 测试间隔
--expend_max_steps 5000               # 最大扩展步数
--expend_test_samples 256             # 测试样本数
--expand_conf_threshold 0.75          # 扩展置信度阈值
--curr_conf_threshold 0.75            # 当前分区置信度阈值
--curr_conf_samples 256               # 当前分区样本数
--use_symgd                           # 启用对称梯度引导
--symgd_mode full                     # 完整SymGD模式
--conf_strategy robust                # 置信度策略
--conf_teacher_temp 1.0               # 教师温度
--llm_model GPT5                      # LLM模型
--describe_nums 80                    # 描述数量
--use_freq_aug                        # 启用频域增强
--use_curr_conf                       # 启用当前分区检查
--use_next_conf                       # 启用下一分区检查
```

**MiDSS + 域课程学习 - 前列腺数据集命令**:

```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MiDSS_DC_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

**⚠️ 注意**: MiDSS的train.py目前可能不支持所有域课程学习参数，需要先进行以下修改：
1. 在 `/app/MixDSemi/MiDSS/code/train.py` 中添加域课程学习相关参数定义
2. 集成课程采样器和置信度计算模块
3. 添加LLM相关的预处理支持

---

## 第三部分：完整的消融实验对比

### 推荐的消融实验方案

#### 实验1: 基线 - MiDSS (不含任何新特性)

```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_Baseline --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --deterministic 1
```

#### 实验2: MiDSS + 频域增强

```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_FreqAug --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --use_freq_aug --deterministic 1
```

#### 实验3: MiDSS + 域课程学习 (不含LLM预处理)

```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_DC_NoLLM --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --use_symgd --symgd_mode full --conf_strategy robust --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

#### 实验4: MiDSS + 域课程学习 + 完整LLM流程 (最终方法)

```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_DC_Full --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

#### 实验5: 基线 - SynFoC (UNet only, 不含SAM)

```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_UNet_Only --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --deterministic 1
```

#### 实验6: SynFoC + 域课程学习 (不含LLM)

```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_DC_NoLLM --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --use_symgd --symgd_mode full --conf_strategy robust --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

---

## 第四部分：实现建议

### 为SynFoC添加域课程学习

**修改步骤:**

1. **复制参数定义** (在 `/app/MixDSemi/SynFoC/code/train.py` 中):
   ```python
   # 在parser.add_argument中添加以下部分
   # ==================== Domain Curriculum Learning ====================
   parser.add_argument('--dc_parts', type=int, default=5, ...)
   parser.add_argument('--dc_distance_mode', type=str, default='prototype', ...)
   # ... (其他参数同train_unet_MiDSS_DC_v2.py)
   ```

2. **关闭SAM模块** (在 `train()` 函数中):
   ```python
   # 注释SAM初始化代码
   # sam_model = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
   # 改为仅使用UNet
   ```

3. **集成课程采样器**:
   从 `train_unet_MiDSS_DC_v2.py` 复制以下部分：
   - `build_distance_curriculum()`
   - `DomainDistanceCurriculumSampler`
   - 课程扩展逻辑

### 为MiDSS添加域课程学习

**修改步骤:**

1. **复制参数定义** (在 `/app/MixDSemi/MiDSS/code/train.py` 中):
   同SynFoC步骤1

2. **集成必要的模块**:
   ```python
   from utils.conf import available_conf_strategies, compute_self_consistency
   from utils.domain_curriculum import DomainDistanceCurriculumSampler, build_distance_curriculum
   from utils.label_ops import to_2d, to_3d
   from utils.tp_ram import extract_amp_spectrum, source_to_target_freq, source_to_target_freq_midss
   ```

3. **修改数据加载**:
   将随机采样改为课程采样器：
   ```python
   ulb_loader = DataLoader(ulb_dataset, batch_size=args.unlabel_bs, 
                          sampler=curriculum_sampler, ...)
   ```

4. **添加置信度计算**:
   在主训练循环中添加置信度检查和课程扩展逻辑

---

## 第五部分：参数对比表

| 参数类型 | 基线 (MiDSS/SynFoC) | + 域课程学习 | 说明 |
|--------|------------------|-----------|------|
| `--dc_parts` | ❌ | 5 | 课程分区数 |
| `--enable_piecewise_tau` | ❌ | ✅ | 自适应阈值 |
| `--use_symgd` | ❌ | ✅ | 对称梯度引导 |
| `--use_freq_aug` | ❌ | ✅ | 频域增强 |
| `--use_curr_conf` | ❌ | ✅ | 当前分区置信度 |
| `--use_next_conf` | ❌ | ✅ | 下一分区置信度 |
| `--llm_model` | ❌ | GPT5 | LLM模型 |
| `--describe_nums` | ❌ | 80 | 描述数量 |

---

## 第六部分：预期结果对比

```
实验1 (MiDSS Baseline):
  - 基础半监督学习
  - 固定伪标签阈值
  - 随机域采样
  - 预期 Dice: ~70-75%

实验2 (MiDSS + FreqAug):
  - 增加频域多样性
  - 预期 Dice: ~72-77%

实验3 (MiDSS + DC):
  - 添加课程学习
  - 动态阈值调整
  - 自适应域采样
  - 预期 Dice: ~75-80%

实验4 (MiDSS + DC + Full):
  - 完整消融实验
  - LLM辅助置信度
  - 最复杂的设置
  - 预期 Dice: ~76-81%

实验5 (SynFoC UNet Only):
  - SynFoC但无SAM
  - 作为SynFoC基线
  - 预期 Dice: ~70-76%

实验6 (SynFoC + DC):
  - SynFoC + 域课程学习
  - 预期 Dice: ~76-82%
```

---

## 注意事项

1. **数据预处理**: 确保LLM生成的分数张量已预处理到指定目录
2. **内存限制**: 某些配置可能需要调整批大小
3. **时间成本**: 域课程学习会增加评估开销
4. **参数文件**: 每次实验会自动保存 `training_config.json`
5. **日志记录**: 检查log.txt文件了解详细训练过程

