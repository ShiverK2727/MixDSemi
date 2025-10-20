#!/bin/bash# U-Net Only 训练脚本（简化版，移除SAM相关参数）

# UNet-Only Training Examples

# This script provides example commands for training UNet-only baseline# 域6训练

screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_8000 --gpu 2 --save_model --overwrite --save_img --max_iterations 8000"

# Fundus dataset - 10 labeled samples from domain 1

python train_unet_only.py \# 域5训练

    --dataset fundus \screen -dmS unet-prostate-d5 bash -c "python train_unet_only.py --dataset prostate --lb_domain 5 --lb_num 20 --save_name UNet_only_d5_8000 --gpu 1 --save_model --overwrite --save_img --max_iterations 8000"

    --lb_num 10 \

    --lb_domain 1 \# 域4训练

    --domain_num 4 \screen -dmS unet-prostate-d4 bash -c "python train_unet_only.py --dataset prostate --lb_domain 4 --lb_num 20 --save_name UNet_only_d4_8000 --gpu 2 --save_model --overwrite --save_img --max_iterations 8000"

    --threshold 0.95 \

    --base_lr 0.01 \# 域3训练

    --warmup \screen -dmS unet-prostate-d3 bash -c "python train_unet_only.py --dataset prostate --lb_domain 3 --lb_num 20 --save_name UNet_only_d3_8000 --gpu 3 --save_model --overwrite --save_img --max_iterations 8000"

    --warmup_period 250 \

    --save_model \# 域2训练

    --gpu 0screen -dmS unet-prostate-d2 bash -c "python train_unet_only.py --dataset prostate --lb_domain 2 --lb_num 20 --save_name UNet_only_d2_8000 --gpu 2 --save_model --overwrite --save_img --max_iterations 8000"



# Prostate dataset - 20 labeled samples from domain 1# 域1训练

python train_unet_only.py \screen -dmS unet-prostate-d1 bash -c "python train_unet_only.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name UNet_only_d1_8000 --gpu 3 --save_model --overwrite --save_img --max_iterations 8000"

    --dataset prostate \

    --lb_num 20 \# =====================================================================

    --lb_domain 1 \# 可选：如果想补偿单模型限制，可以调整以下参数

    --domain_num 6 \# =====================================================================

    --threshold 0.95 \

    --base_lr 0.01 \# 选项1：增加训练迭代数（推荐）

    --warmup \# screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_12000 --gpu 0 --save_model --overwrite --save_img --max_iterations 12000"

    --warmup_period 250 \

    --save_model \# 选项2：降低置信度阈值以获取更多伪标签

    --gpu 0# screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_threshold90 --gpu 0 --save_model --overwrite --save_img --max_iterations 8000 --threshold 0.90"



# MNMS dataset - 8 labeled samples from domain 1# 选项3：提高一致性权重

python train_unet_only.py \# screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_cons1.5 --gpu 0 --save_model --overwrite --save_img --max_iterations 8000 --consistency 1.5"

    --dataset MNMS \

    --lb_num 8 \# 选项4：综合调整（推荐用于对比实验）

    --lb_domain 1 \# screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_optimized --gpu 0 --save_model --overwrite --save_img --max_iterations 12000 --threshold 0.90 --consistency 1.2 --consistency_rampup 150.0"

    --domain_num 4 \
    --threshold 0.95 \
    --base_lr 0.01 \
    --warmup \
    --warmup_period 250 \
    --save_model \
    --gpu 0

# BUSI dataset - 10 labeled samples from domain 1
python train_unet_only.py \
    --dataset BUSI \
    --lb_num 10 \
    --lb_domain 1 \
    --domain_num 2 \
    --threshold 0.95 \
    --base_lr 0.01 \
    --warmup \
    --warmup_period 250 \
    --save_model \
    --gpu 0
