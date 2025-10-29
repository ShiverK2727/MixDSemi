# 消融实验总结 - 域课程学习集成方案

## 📌 核心问题

如何在现有的半监督分割方法（MiDSS、SynFoC）中集成域课程学习，并验证其有效性？

---

## ✅ 解决方案概览

### 核心策略

| 方法 | 需要取消的项 | 需要添加的项 |
|-----|-----------|-----------|
| **MiDSS + DC** | 无（保留所有MiDSS特性） | 域课程学习、SymGD、LLM置信度、频域增强 |
| **SynFoC + DC (UNet only)** | SAM相关参数（rank、AdamW、module等） | 域课程学习、SymGD、LLM置信度、频域增强 |

---

## 🎯 具体操作

### 第一步：确定基线配置

#### MiDSS 基线
```bash
# 无任何改进的纯MiDSS
python /app/MixDSemi/MiDSS/code/train.py \
  --dataset prostate --lb_domain 1 --lb_num 20 \
  --save_name Ablation_MiDSS_Baseline_v1 --gpu 3 \
  --save_model --overwrite \
  --max_iterations 30000 --deterministic 1
```

#### SynFoC 基线 (UNet Only)
```bash
# SynFoC但禁用SAM，仅使用UNet
python /app/MixDSemi/SynFoC/code/train.py \
  --dataset prostate --lb_domain 1 --lb_num 20 \
  --save_name Ablation_SynFoC_UNet_Only_v1 --gpu 3 \
  --save_model --overwrite \
  --max_iterations 30000 --deterministic 1
# 需要在train.py中注释SAM初始化
```

### 第二步：集成域课程学习

#### 参数配置

**取消项**:
- 无（MiDSS）
- SAM相关（SynFoC）

**添加项**（所有都是新增）:
```bash
--dc_parts 5                          # 课程分区
--dc_distance_mode sqrt_prod          # 距离度量
--enable_piecewise_tau                # 自适应阈值
--tau_min 0.80 --tau_max 0.95         # 阈值范围
--expend_test_steps_interval 300      # 评估间隔
--expend_max_steps 5000               # 最大步数
--expend_test_samples 256             # 测试样本数
--expand_conf_threshold 0.75          # 扩展阈值
--curr_conf_threshold 0.75            # 当前分区阈值
--curr_conf_samples 256               # 当前分区样本数
--use_curr_conf --use_next_conf       # 启用置信度检查
--use_symgd --symgd_mode full         # 对称梯度引导
--ul_weight 1.0 --lu_weight 1.0       # CutMix权重
--cons_weight 1.0                     # 一致性权重
--use_freq_aug                        # 频域增强
--conf_strategy robust                # 置信度策略
--llm_model GPT5 --describe_nums 80   # LLM配置
```

#### MiDSS + 完整DC 命令
```bash
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_DC_Full_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

#### SynFoC + 完整DC 命令
```bash
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_DC_Full_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf
```

### 第三步：代码修改

#### MiDSS 修改清单

1. **参数定义** (`train.py` 第30-70行):
   - 添加所有域课程学习参数
   - 添加SymGD参数
   - 添加LLM参数
   - 添加频域增强参数

2. **模块导入** (`train.py` 顶部):
   ```python
   from utils.conf import available_conf_strategies, compute_self_consistency
   from utils.domain_curriculum import DomainDistanceCurriculumSampler, build_distance_curriculum
   from utils.label_ops import to_2d, to_3d
   from utils.tp_ram import extract_amp_spectrum, source_to_target_freq, source_to_target_freq_midss
   from utils.training import Statistics, cycle, obtain_cutmix_box
   ```

3. **课程采样器初始化**:
   ```python
   curriculum_sampler = DomainDistanceCurriculumSampler(...)
   ulb_loader = DataLoader(..., sampler=curriculum_sampler, ...)
   ```

4. **主训练循环**:
   - 添加置信度检查
   - 添加课程扩展逻辑
   - 使用自适应阈值

#### SynFoC 修改清单

1-4. 同MiDSS

5. **关闭SAM模块**:
   ```python
   # 注释SAM初始化和前向传播
   # 仅使用UNet进行分割
   ```

---

## 📊 消融实验设计

### 推荐的7个实验

| ID | 实验名称 | 基线 | +DC | +频域 | +SymGD | +LLM | 预期Dice |
|----|--------|------|-----|------|---------|------|---------|
| 1 | MiDSS基线 | ✅ | ❌ | ❌ | ❌ | ❌ | 70-75% |
| 2 | MiDSS+频域 | ✅ | ❌ | ✅ | ❌ | ❌ | 72-77% |
| 3 | MiDSS+DC | ✅ | ✅ | ✅ | ✅ | ❌ | 75-80% |
| 4 | MiDSS+DC+LLM | ✅ | ✅ | ✅ | ✅ | ✅ | 76-81% |
| 5 | SynFoC基线 | ✅ | ❌ | ❌ | ❌ | ❌ | 70-76% |
| 6 | SynFoC+DC | ✅ | ✅ | ✅ | ✅ | ❌ | 76-82% |
| 7 | SynFoC+DC+LLM | ✅ | ✅ | ✅ | ✅ | ✅ | 77-82% |

### 实验运行顺序建议

1. 先运行两个基线（Exp 1, 5）- 快速
2. 验证频域增强效果（Exp 2）- 快速
3. 测试DC核心（Exp 3, 6）- 中等
4. 完整验证（Exp 4, 7）- 完整

---

## 🔑 关键实现要点

### 为什么这些参数很重要

| 参数 | 作用 | 推荐值 | 调整建议 |
|-----|------|--------|---------|
| `--dc_parts` | 课程难度分级 | 5 | 增加→更细粒度；减少→更快收敛 |
| `--enable_piecewise_tau` | 动态阈值 | 启用 | 禁用→使用固定阈值 |
| `--expend_test_steps_interval` | 评估频率 | 300 | 增加→评估少、训练快；减少→评估多、更精确 |
| `--use_curr_conf` | 当前分区检查 | 启用 | 保守策略，防止过早扩展 |
| `--use_next_conf` | 下一分区检查 | 启用 | 激进策略，加速课程进度 |
| `--llm_model` | LLM置信度源 | GPT5 | 更好的文本理解 |

### 取消vs保留决策

**MiDSS**:
- ✅ 保留: 所有MiDSS原有参数（CutMix、EMA等）
- ❌ 取消: 无需取消任何参数
- ✨ 新增: 所有DC相关参数

**SynFoC**:
- ✅ 保留: EMA、CutMix、频域参数
- ❌ 取消: SAM相关（rank、AdamW、ckpt等）
- ✨ 新增: 所有DC相关参数

---

## 📁 生成的文档

已创建以下文档供参考：

1. **ABLATION_EXPERIMENT_GUIDE.md** - 详细的消融实验指南
2. **IMPLEMENTATION_GUIDE.md** - 代码修改和集成步骤
3. **COMMANDS_QUICK_REFERENCE.md** - 命令速查表
4. **run_ablation_experiments.sh** - 自动化脚本

---

## 🚀 快速开始

### 方式1: 运行单个实验
```bash
# MiDSS + 完整DC
cd /app/MixDSemi/MiDSS/code
python train.py --dataset prostate --lb_domain 1 --lb_num 20 \
  --save_name Ablation_Test_v1 --gpu 3 --save_model --overwrite \
  --max_iterations 30000 --dc_parts 5 --enable_piecewise_tau \
  --use_symgd --use_freq_aug --use_curr_conf --use_next_conf
```

### 方式2: 批量运行
```bash
bash /app/MixDSemi/SynFoCLIP/code/run_ablation_experiments.sh
```

### 方式3: 自定义组合
```bash
# 根据需要从COMMANDS_QUICK_REFERENCE.md中复制命令
```

---

## ✨ 预期收获

### 通过这些实验将证明:

1. **域课程学习的有效性** - DC能否改进MiDSS性能？
2. **不同特性的贡献** - 频域增强、SymGD、LLM各贡献多少？
3. **方法的通用性** - DC在SynFoC上是否也有效？
4. **特性的互补性** - 特性组合是否产生协同效应？

### 预期性能提升:

- **MiDSS**: 70-75% → 76-81% (+6-11%)
- **SynFoC**: 70-76% → 77-82% (+7-12%)

---

## 📝 论文撰写建议

### 消融表示例

```
Table 1: Ablation Study on Prostate Dataset

Method          DC  FreqAug SymGD  LLM  Dice↑  
─────────────────────────────────────────────
MiDSS           ✗   ✗      ✗      ✗    72.3%
+FreqAug        ✗   ✓      ✗      ✗    74.1%  (+1.8%)
+DC             ✓   ✓      ✓      ✗    77.8%  (+5.5%)
+DC+LLM         ✓   ✓      ✓      ✓    79.2%  (+7.0%)
SynFoC          ✗   ✗      ✗      ✗    73.5%
+DC             ✓   ✓      ✓      ✗    79.1%  (+5.6%)
+DC+LLM         ✓   ✓      ✓      ✓    80.5%  (+7.0%)
```

---

## 📞 常见问题

**Q: 需要修改train.py吗?**
A: 是的，需要添加新参数和集成课程采样器。参考IMPLEMENTATION_GUIDE.md。

**Q: 所有参数都必须设置吗?**
A: 不必。可以逐步添加：先DC核心，再加SymGD，最后加LLM。

**Q: 如何快速测试修改是否成功?**
A: 用少量迭代运行: `--max_iterations 1000 --num_eval_iter 100`

**Q: 哪个实验最重要?**
A: Exp 3和6（+DC不含LLM），因为这些证明了核心有效性。

