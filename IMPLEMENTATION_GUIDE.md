# 集成域课程学习到现有方法 - 实现指南

## 快速参考

### 当前配置摘要
```
python train_unet_MiDSS_DC_v2.py \
  --dataset prostate \
  --lb_domain 1 \
  --lb_num 20 \
  --dc_parts 5 \
  --dc_distance_mode sqrt_prod \
  --enable_piecewise_tau \
  --tau_min 0.80 \
  --tau_max 0.95 \
  --expend_test_steps_interval 300 \
  --expend_max_steps 5000 \
  --use_symgd \
  --symgd_mode full \
  --use_freq_aug \
  --use_curr_conf \
  --use_next_conf
```

---

## 一、集成到 MiDSS 的步骤

### 第1步：修改参数定义

**文件**: `/app/MixDSemi/MiDSS/code/train.py` (第30-70行)

**添加以下参数块**:

```python
# ==================== Domain Curriculum Learning ====================
parser.add_argument('--dc_parts', type=int, default=5,
                    help='Number of curriculum partitions')
parser.add_argument('--dc_distance_mode', type=str, default='prototype',
                    choices=['prototype', 'sqrt_prod'],
                    help='Distance metric for curriculum')
parser.add_argument('--expend_test_samples', type=int, default=32,
                    help='Number of samples to test from next partition')
parser.add_argument('--expend_test_steps_interval', type=int, default=100,
                    help='Test next partition every N steps')
parser.add_argument('--expend_max_steps', type=int, default=1000,
                    help='Maximum steps before forcing curriculum expansion')
parser.add_argument('--use_curr_conf', action='store_true',
                    help='Require current-partition confidence for expansion')
parser.add_argument('--curr_conf_threshold', type=float, default=0.6,
                    help='Confidence threshold for current partition')
parser.add_argument('--curr_conf_samples', type=int, default=32,
                    help='Number of current-partition samples to test')
parser.add_argument('--use_next_conf', action='store_true',
                    help='Require next-partition confidence for expansion')
parser.add_argument('--expand_conf_threshold', type=float, default=0.6,
                    help='Confidence threshold for next partition expansion')

# ==================== Pseudo-Label Threshold ====================
parser.add_argument('--enable_piecewise_tau', action='store_true',
                    help='Enable curriculum-adaptive threshold')
parser.add_argument('--tau_min', type=float, default=0.80,
                    help='Minimum threshold at initial curriculum stage')
parser.add_argument('--tau_max', type=float, default=0.95,
                    help='Maximum threshold at final curriculum stage')

# ==================== Symmetric Gradient Guidance (SymGD) ====================
parser.add_argument('--use_symgd', dest='use_symgd', action='store_true',
                    help='Enable Symmetric Gradient guidance')
parser.add_argument('--no_symgd', dest='use_symgd', action='store_false',
                    help='Disable Symmetric Gradient guidance')
parser.set_defaults(use_symgd=True)
parser.add_argument('--symgd_mode', type=str, default='full',
                    choices=['full', 'ul_only'],
                    help='SymGD mode: full (UL+LU) or ul_only (UL only)')
parser.add_argument('--ul_weight', type=float, default=1.0,
                    help='Weight for UL CutMix pseudo-label loss')
parser.add_argument('--lu_weight', type=float, default=1.0,
                    help='Weight for LU CutMix pseudo-label loss')
parser.add_argument('--cons_weight', type=float, default=1.0,
                    help='Weight for strong-augmentation consistency loss')

# ==================== Preprocessing & Confidence Strategy ====================
parser.add_argument('--preprocess_dir', type=str, default=None,
                    help='Override preprocessing directory')
parser.add_argument('--llm_model', type=str, default='gemini',
                    choices=['gemini', 'GPT5', 'DeepSeek'],
                    help='LLM model used to generate score tensors')
parser.add_argument('--describe_nums', type=int, default=40,
                    choices=[20, 40, 60, 80],
                    help='Number of textual descriptions for preprocessing')
parser.add_argument('--conf_strategy', type=str, default='robust',
                    help='Confidence strategy for self-consistency filtering')
parser.add_argument('--conf_teacher_temp', type=float, default=1.0,
                    help='Temperature for softening teacher probabilities')

# ==================== Data Augmentation ====================
parser.add_argument('--use_freq_aug', action='store_true',
                    help='Enable frequency-domain augmentation')
parser.add_argument('--LB', type=float, default=0.01,
                    help='Low-frequency band ratio for frequency augmentation')
```

### 第2步：导入必要的模块

**在train.py顶部添加**:

```python
# 在 from utils import losses, metrics, ramps, util 之后添加
from utils.conf import available_conf_strategies, compute_self_consistency
from utils.domain_curriculum import DomainDistanceCurriculumSampler, build_distance_curriculum
from utils.label_ops import to_2d, to_3d
from utils.tp_ram import extract_amp_spectrum, source_to_target_freq, source_to_target_freq_midss
from utils.training import Statistics, cycle, obtain_cutmix_box
```

### 第3步：修改数据加载

**在train()函数中，修改unlabeled数据加载部分**:

```python
# 原代码（约第380-390行）:
# ulb_loader = DataLoader(ulb_dataset, batch_size=args.unlabel_bs, shuffle=True, ...)

# 修改为:
curriculum_partitions = args.dc_parts
labeled_scores = lb_dataset.get_scores()
unlabeled_scores = ulb_dataset.get_scores()
unlabeled_indices = list(range(len(unlabeled_scores)))

_, _, _, partitions = build_distance_curriculum(
    labeled_scores,
    unlabeled_scores,
    unlabeled_indices,
    num_partitions=curriculum_partitions,
    distance_mode=getattr(args, 'dc_distance_mode', 'prototype'),
)
curriculum_sampler = DomainDistanceCurriculumSampler(
    partitions,
    initial_stage=1,
    seed=args.seed,
    shuffle=True,
)

ulb_loader = DataLoader(ulb_dataset, batch_size=args.unlabel_bs, 
                       sampler=curriculum_sampler, num_workers=2, 
                       pin_memory=True, drop_last=True)
```

### 第4步：集成课程扩展逻辑

**在主训练循环中添加置信度检查**（参考train_unet_MiDSS_DC_v2.py第904-1010行）

---

## 二、集成到 SynFoC 的步骤

### 第1步：参数定义

**文件**: `/app/MixDSemi/SynFoC/code/train.py`

**操作**: 同MiDSS第1步，添加相同的参数块

**额外操作**: 注释或移除SAM相关参数

```python
# 注释这些参数（或保持但不使用）:
# parser.add_argument('--rank', ...)
# parser.add_argument('--AdamW', ...)
# parser.add_argument('--module', ...)
# parser.add_argument('--img_size', ...)
# parser.add_argument('--vit_name', ...)
# parser.add_argument('--ckpt', ...)
# parser.add_argument('--eval', ...)
```

### 第2步：关闭SAM模块

**在train()函数中（约第700-750行）**:

```python
# 原代码可能使用SAM模型：
# sam_model = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
# 改为仅使用UNet：

# 只保留UNet初始化，注释SAM相关代码
unet_model = create_model()
ema_unet_model = create_model(ema=True)

# 注释:
# sam_model = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
# seg_model = TwoHeadSegModel(sam_model, unet_model, args.rank, args.AdamW)
```

### 第3步：使用UNet的伪标签而非SAM

**在伪标签生成部分（约第750-800行）**:

```python
# 原代码可能：
# with torch.no_grad():
#     sam_pred = seg_model.forward_sam(...)
#     unet_pred = seg_model.forward_unet_w(...)

# 修改为仅使用UNet：
with torch.no_grad():
    unet_logits = ema_unet_model(ulb_unet_size_x_w)
    unet_prob = torch.softmax(unet_logits, dim=1)
    unet_prob, unet_pseudo_label = torch.max(unet_prob, dim=1)
    mask_teacher = (unet_prob > threshold).unsqueeze(1).float()
```

### 第4步：参数和数据加载

**同MiDSS第2和3步**

---

## 三、关键参数调整建议

### 对于快速实验（节省时间）

```bash
--expend_test_steps_interval 500    # 增加评估间隔
--expend_test_samples 128           # 减少测试样本
--max_iterations 15000              # 减少总迭代数
```

### 对于深度实验（获得更好结果）

```bash
--expend_test_steps_interval 200    # 减少评估间隔
--expend_test_samples 512           # 增加测试样本
--max_iterations 60000              # 标准迭代数
--dc_parts 8                        # 增加分区数
```

### 对于消融实验

```bash
# 仅测试域课程学习（不含其他新特性）
--disable_freq_aug                  # 禁用频域增强
--no_symgd                          # 禁用对称梯度引导
--conf_strategy simple              # 使用简单置信度

# 逐步启用特性
# 第1步: + 频域增强
# 第2步: + 对称梯度引导
# 第3步: + LLM置信度
```

---

## 四、命令对比表

### MiDSS 集成选项

```bash
# 选项1: 最小化修改（仅DC核心）
python train.py --dataset prostate --lb_num 20 \
  --dc_parts 5 --use_curr_conf --use_next_conf

# 选项2: 完整DC（无LLM）
python train.py --dataset prostate --lb_num 20 \
  --dc_parts 5 --enable_piecewise_tau --use_symgd \
  --use_freq_aug --use_curr_conf --use_next_conf

# 选项3: 完整DC（含LLM）
python train.py --dataset prostate --lb_num 20 \
  --dc_parts 5 --enable_piecewise_tau --use_symgd \
  --use_freq_aug --use_curr_conf --use_next_conf \
  --llm_model GPT5 --describe_nums 80
```

### SynFoC 集成选项

```bash
# 选项1: UNet + DC (无SAM, 无LLM)
python train.py --dataset prostate --lb_num 20 \
  --model unet_only \
  --dc_parts 5 --use_curr_conf --use_next_conf

# 选项2: UNet + 完整DC
python train.py --dataset prostate --lb_num 20 \
  --model unet_only \
  --dc_parts 5 --enable_piecewise_tau --use_symgd \
  --use_freq_aug --use_curr_conf --use_next_conf
```

---

## 五、验证检查清单

实现完成后，检查以下项：

- [ ] 参数定义已添加
- [ ] 模块导入无误
- [ ] 课程采样器初始化成功
- [ ] 数据加载使用课程采样器
- [ ] 主训练循环中有置信度计算
- [ ] 课程扩展逻辑已实现
- [ ] 伪标签使用当前阈值
- [ ] 对称梯度引导已启用
- [ ] 频域增强已集成
- [ ] 日志输出包含课程信息

### 测试命令

```bash
# 快速测试（1000迭代）
python train.py --dataset prostate --lb_num 20 \
  --max_iterations 1000 --num_eval_iter 100 \
  --save_name test_dc_integration \
  --dc_parts 3 --use_curr_conf --use_next_conf
```

---

## 六、故障排除

| 问题 | 解决方案 |
|------|--------|
| 缺少 `get_scores()` 方法 | 数据集需支持LLM分数返回 |
| 课程采样器初始化失败 | 检查分区数是否合理 |
| OOM错误 | 减少 `expend_test_samples` 或 `batch_size` |
| 置信度始终为0 | 检查LLM预处理目录配置 |
| 课程不扩展 | 调整 `expand_conf_threshold` 或 `expend_max_steps` |

