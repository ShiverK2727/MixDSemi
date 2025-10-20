# 学习率和训练循环优化说明

## 修改日期
2025-10-20

## 主要修改

### 1. 学习率配置优化

**原始行为（有问题）：**
- `--base_lr` 默认值为 `0.03`
- 如果启用 `--warmup`，SAM optimizer 的初始学习率会被设置为 `base_lr / warmup_period = 0.03 / 250 = 0.00012`
- **问题**: 学习率永远保持在 `0.00012`，没有warmup调度器来逐步增加到 `0.03`
- 这导致warmup实际上**不生效**，反而使用了极低且固定的学习率

**新行为（已修复）：**
- `--base_lr` 默认值改为 `None`
- **如果不设置 `--base_lr`**（默认情况）：
  - 使用固定学习率 `0.03`
  - 自动禁用warmup（保持原始训练行为）
  - 日志: "Using default fixed learning rate: 0.03 (warmup disabled)"
  
- **如果设置 `--base_lr`**（例如 `--base_lr 0.01`）：
  - 使用用户指定的学习率
  - 可以选择启用 `--warmup`
  - 如果启用warmup，会在前 `warmup_period` 步内线性增加学习率从 `base_lr/warmup_period` 到 `base_lr`
  - 日志: "Using custom learning rate: 0.01 with warmup enabled (warmup_period=250)"

### 2. Warmup实现修复

**新增功能：**
```python
# 在训练循环中添加了实际的warmup调度
if args.warmup and iter_num < args.warmup_period:
    lr_scale = (iter_num + 1) / args.warmup_period
    for param_group in sam_optimizer.param_groups:
        param_group['lr'] = base_lr * lr_scale
```

**效果：**
- 第1步: lr = base_lr * (1/250) = base_lr * 0.004
- 第125步: lr = base_lr * (125/250) = base_lr * 0.5
- 第250步: lr = base_lr * (250/250) = base_lr
- 第251步及之后: lr = base_lr (固定)

### 3. 训练循环优化

**原始代码：**
```python
start_epoch = 0
max_epoch = max_iterations // args.num_eval_iter
for epoch_num in range(start_epoch, max_epoch):
    # ...
```

**问题：**
- `epoch_num` 和 `start_epoch` 的命名容易混淆
- 实际上这不是真正的epoch，而是evaluation cycle
- `start_epoch` 总是0，从未被修改

**新代码：**
```python
num_eval_cycles = max_iterations // args.num_eval_iter
for eval_cycle in range(num_eval_cycles):
    # ...
```

**改进：**
- 更清晰的命名：`eval_cycle` 表示评估周期
- 移除了无用的 `start_epoch` 变量
- 进度条显示: `Cycle 1/120` 而不是 `No. 1`

## 训练参数说明

### 关于"epoch"的说明
代码实际上是**按步数(iterations)训练**，而不是按epoch：
- `max_iterations`: 总训练步数（例如60000）
- `num_eval_iter`: 每隔多少步进行一次评估（例如500）
- `num_eval_cycles`: 评估周期数 = max_iterations / num_eval_iter（例如120）

每个evaluation cycle包含 `num_eval_iter` 个训练步，因此：
- 总步数 = num_eval_cycles × num_eval_iter
- 60000 = 120 × 500

### 关于warmup_period
- `warmup_period` 是**步数(iterations)**，不是epoch
- 默认值: 250步
- 只有在用户设置了 `--base_lr` 且启用了 `--warmup` 时才生效

## 使用示例

### 1. 默认行为（保持原始训练方式）
```bash
python train_synfoc.py --dataset prostate --lb_domain 6 --lb_num 20 \
    --save_name test --gpu 0 --AdamW --model MedSAM --save_model --overwrite
```
- 使用固定学习率 0.03
- warmup不生效
- 与原始代码行为一致（但修复了bug）

### 2. 使用自定义学习率（不使用warmup）
```bash
python train_synfoc.py --dataset prostate --lb_domain 6 --lb_num 20 \
    --base_lr 0.01 --save_name test --gpu 0 --AdamW --model MedSAM --save_model --overwrite
```
- 使用固定学习率 0.01
- warmup不生效

### 3. 使用自定义学习率 + warmup
```bash
python train_synfoc.py --dataset prostate --lb_domain 6 --lb_num 20 \
    --base_lr 0.01 --warmup --warmup_period 500 \
    --save_name test --gpu 0 --AdamW --model MedSAM --save_model --overwrite
```
- 前500步：学习率从 0.00002 线性增加到 0.01
- 第501步及之后：学习率固定为 0.01

## 兼容性

**向后兼容性：**
- 所有原有的训练命令仍然有效
- 如果不指定 `--base_lr`，行为与原始代码相同（固定lr=0.03，无warmup）
- 只有在用户明确设置 `--base_lr` 时，才能选择性地启用warmup

**注意事项：**
- UNet optimizer 的学习率始终使用 `base_lr`（无warmup）
- SAM optimizer 的学习率会受warmup影响（如果启用）
- 建议在新实验中使用自定义 `--base_lr` 和 `--warmup` 来探索更好的训练策略
