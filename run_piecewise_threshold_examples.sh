#!/bin/bash
# 分段动态阈值训练示例脚本

# ============================================================================
# 1. Prostate数据集 - 推荐配置（6域，难度梯度大）
# ============================================================================
echo "示例1: Prostate数据集 - 启用分段动态阈值"
python train_unet_MiDSS_DC_v2.py \
  --dataset prostate \
  --lb_domain 1 \
  --lb_num 20 \
  --save_name MIDSS_DC_v2_piecewise_tau_test \
  --gpu 0 \
  --save_model \
  --overwrite \
  --dc_parts 5 \
  --expend_test_steps_interval 300 \
  --expend_max_steps 5000 \
  --expand_conf_threshold 0.75 \
  --curr_conf_threshold 0.75 \
  --ul_weight 1.0 \
  --lu_weight 1.0 \
  --cons_weight 1.0 \
  --llm_model GPT5 \
  --describe_nums 80 \
  --warmup \
  --max_iterations 30000 \
  --use_freq_aug \
  --enable_piecewise_tau \
  --tau_min 0.75 \
  --tau_max 0.95

# ============================================================================
# 2. 固定阈值基线（对比实验）
# ============================================================================
echo ""
echo "示例2: Prostate数据集 - 固定高阈值基线（对比组）"
python train_unet_MiDSS_DC_v2.py \
  --dataset prostate \
  --lb_domain 1 \
  --lb_num 20 \
  --save_name MIDSS_DC_v2_fixed_tau_baseline \
  --gpu 1 \
  --save_model \
  --overwrite \
  --dc_parts 5 \
  --expend_test_steps_interval 300 \
  --expend_max_steps 5000 \
  --expand_conf_threshold 0.75 \
  --curr_conf_threshold 0.75 \
  --ul_weight 1.0 \
  --lu_weight 1.0 \
  --cons_weight 1.0 \
  --llm_model GPT5 \
  --describe_nums 80 \
  --warmup \
  --max_iterations 30000 \
  --use_freq_aug \
  --threshold 0.95
  # 注意：不加 --enable_piecewise_tau，使用固定阈值

# ============================================================================
# 3. Fundus数据集 - 标准配置（4域）
# ============================================================================
echo ""
echo "示例3: Fundus数据集 - 分段动态阈值"
python train_unet_MiDSS_DC_v2.py \
  --dataset fundus \
  --lb_domain 1 \
  --lb_num 20 \
  --save_name Fundus_DC_v2_piecewise_tau \
  --gpu 2 \
  --save_model \
  --overwrite \
  --dc_parts 5 \
  --expend_test_steps_interval 200 \
  --expend_max_steps 3000 \
  --expand_conf_threshold 0.70 \
  --curr_conf_threshold 0.70 \
  --ul_weight 1.0 \
  --lu_weight 1.0 \
  --cons_weight 1.0 \
  --llm_model gemini \
  --describe_nums 40 \
  --max_iterations 30000 \
  --use_freq_aug \
  --enable_piecewise_tau \
  --tau_min 0.80 \
  --tau_max 0.95

# ============================================================================
# 4. MNMS数据集 - 困难配置（心脏分割，3类）
# ============================================================================
echo ""
echo "示例4: MNMS数据集 - 分段动态阈值（困难场景）"
python train_unet_MiDSS_DC_v2.py \
  --dataset MNMS \
  --lb_domain 1 \
  --lb_num 50 \
  --save_name MNMS_DC_v2_piecewise_tau_hard \
  --gpu 3 \
  --save_model \
  --overwrite \
  --dc_parts 5 \
  --expend_test_steps_interval 500 \
  --expend_max_steps 6000 \
  --expand_conf_threshold 0.80 \
  --curr_conf_threshold 0.80 \
  --ul_weight 1.0 \
  --lu_weight 1.0 \
  --cons_weight 1.0 \
  --llm_model GPT5 \
  --describe_nums 80 \
  --warmup \
  --max_iterations 60000 \
  --use_freq_aug \
  --enable_piecewise_tau \
  --tau_min 0.85 \
  --tau_max 0.97
  # 注意：困难数据集使用更高的阈值范围

# ============================================================================
# 5. 消融实验：不同 tau_min/tau_max 组合
# ============================================================================
echo ""
echo "示例5: 消融实验 - 不同阈值范围"

# 5a. 较窄范围（0.85-0.95）
python train_unet_MiDSS_DC_v2.py \
  --dataset prostate \
  --lb_domain 1 \
  --lb_num 20 \
  --save_name ablation_tau_narrow_range \
  --gpu 0 \
  --overwrite \
  --dc_parts 5 \
  --max_iterations 30000 \
  --enable_piecewise_tau \
  --tau_min 0.85 \
  --tau_max 0.95

# 5b. 较宽范围（0.70-0.97）
python train_unet_MiDSS_DC_v2.py \
  --dataset prostate \
  --lb_domain 1 \
  --lb_num 20 \
  --save_name ablation_tau_wide_range \
  --gpu 1 \
  --overwrite \
  --dc_parts 5 \
  --max_iterations 30000 \
  --enable_piecewise_tau \
  --tau_min 0.70 \
  --tau_max 0.97

# ============================================================================
# 6. 使用screen在后台运行（推荐用于长时间训练）
# ============================================================================
echo ""
echo "示例6: 使用screen在后台运行"

screen -dmS piecewise_tau_exp bash -c "
python train_unet_MiDSS_DC_v2.py \
  --dataset prostate \
  --lb_domain 1 \
  --lb_num 20 \
  --save_name MIDSS_DC_v2_piecewise_production \
  --gpu 0 \
  --save_model \
  --overwrite \
  --dc_parts 5 \
  --expend_test_steps_interval 300 \
  --expend_max_steps 5000 \
  --expand_conf_threshold 0.75 \
  --curr_conf_threshold 0.75 \
  --ul_weight 1.0 \
  --lu_weight 1.0 \
  --cons_weight 1.0 \
  --llm_model GPT5 \
  --describe_nums 80 \
  --warmup \
  --max_iterations 30000 \
  --use_freq_aug \
  --enable_piecewise_tau \
  --tau_min 0.75 \
  --tau_max 0.95
"

echo "训练已在后台启动，使用 'screen -r piecewise_tau_exp' 查看日志"

# ============================================================================
# 7. 禁用SymGD，仅测试分段阈值效果
# ============================================================================
echo ""
echo "示例7: 禁用SymGD，测试分段阈值独立效果"
python train_unet_MiDSS_DC_v2.py \
  --dataset prostate \
  --lb_domain 1 \
  --lb_num 20 \
  --save_name piecewise_tau_no_symgd \
  --gpu 2 \
  --overwrite \
  --dc_parts 5 \
  --max_iterations 30000 \
  --no_symgd \
  --enable_piecewise_tau \
  --tau_min 0.75 \
  --tau_max 0.95

# ============================================================================
# 监控提示
# ============================================================================
echo ""
echo "============================================================================"
echo "训练监控提示:"
echo "============================================================================"
echo "1. TensorBoard监控:"
echo "   tensorboard --logdir=../model/prostate/train_unet_MiDSS_DC_v2/"
echo ""
echo "2. 关键监控指标:"
echo "   - train/threshold           # 当前阈值变化曲线"
echo "   - train/curriculum_stage    # 课程阶段进度"
echo "   - train/mask_teacher        # 伪标签保留比例"
echo "   - train/unet_ulb_*_dice     # 无标签数据Dice（质量指标）"
echo ""
echo "3. 日志关键词:"
echo "   grep 'Piecewise threshold ENABLED' ../model/*/log.txt"
echo "   grep 'threshold updated' ../model/*/log.txt"
echo "============================================================================"
