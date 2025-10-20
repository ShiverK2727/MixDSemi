# U-Net Only 训练脚本（简化版，移除SAM相关参数）

# 域6训练
screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_8000 --gpu 0 --save_model --overwrite --save_img --max_iterations 8000"

# 域5训练
screen -dmS unet-prostate-d5 bash -c "python train_unet_only.py --dataset prostate --lb_domain 5 --lb_num 20 --save_name UNet_only_d5_8000 --gpu 1 --save_model --overwrite --save_img --max_iterations 8000"

# 域4训练
screen -dmS unet-prostate-d4 bash -c "python train_unet_only.py --dataset prostate --lb_domain 4 --lb_num 20 --save_name UNet_only_d4_8000 --gpu 2 --save_model --overwrite --save_img --max_iterations 8000"

# 域3训练
screen -dmS unet-prostate-d3 bash -c "python train_unet_only.py --dataset prostate --lb_domain 3 --lb_num 20 --save_name UNet_only_d3_8000 --gpu 3 --save_model --overwrite --save_img --max_iterations 8000"

# 域2训练
screen -dmS unet-prostate-d2 bash -c "python train_unet_only.py --dataset prostate --lb_domain 2 --lb_num 20 --save_name UNet_only_d2_8000 --gpu 2 --save_model --overwrite --save_img --max_iterations 8000"

# 域1训练
screen -dmS unet-prostate-d1 bash -c "python train_unet_only.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name UNet_only_d1_8000 --gpu 3 --save_model --overwrite --save_img --max_iterations 8000"

# =====================================================================
# 可选：如果想补偿单模型限制，可以调整以下参数
# =====================================================================

# 选项1：增加训练迭代数（推荐）
# screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_12000 --gpu 0 --save_model --overwrite --save_img --max_iterations 12000"

# 选项2：降低置信度阈值以获取更多伪标签
# screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_threshold90 --gpu 0 --save_model --overwrite --save_img --max_iterations 8000 --threshold 0.90"

# 选项3：提高一致性权重
# screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_cons1.5 --gpu 0 --save_model --overwrite --save_img --max_iterations 8000 --consistency 1.5"

# 选项4：综合调整（推荐用于对比实验）
# screen -dmS unet-prostate-d6 bash -c "python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_optimized --gpu 0 --save_model --overwrite --save_img --max_iterations 12000 --threshold 0.90 --consistency 1.2 --consistency_rampup 150.0"
