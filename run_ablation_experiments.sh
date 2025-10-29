#!/bin/bash

################################################################################
# 消融实验脚本 - 域课程学习在现有方法中的效果验证
# 实验设置: 前列腺数据集 (Prostate)
# 基线方法: MiDSS 和 SynFoC
################################################################################

# ===================== 实验1: MiDSS 基线 =====================
echo "============================================"
echo "Exp1: MiDSS Baseline (No DC, No FreqAug)"
echo "============================================"
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_Baseline_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --deterministic 1

# ===================== 实验2: MiDSS + 频域增强 =====================
echo "============================================"
echo "Exp2: MiDSS + Frequency Augmentation"
echo "============================================"
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_FreqAug_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --use_freq_aug --deterministic 1

# ===================== 实验3: MiDSS + 域课程学习 (无LLM) =====================
echo "============================================"
echo "Exp3: MiDSS + Domain Curriculum (No LLM)"
echo "============================================"
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_DC_NoLLM_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --use_symgd --symgd_mode full --conf_strategy robust --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf

# ===================== 实验4: MiDSS + 域课程学习 (完整版) =====================
echo "============================================"
echo "Exp4: MiDSS + Domain Curriculum (Full LLM)"
echo "============================================"
python /app/MixDSemi/MiDSS/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_MiDSS_DC_Full_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --use_symgd --symgd_mode full --conf_strategy robust --conf_teacher_temp 1.0 --llm_model GPT5 --describe_nums 80 --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf

# ===================== 实验5: SynFoC UNet Only (不含SAM) =====================
echo "============================================"
echo "Exp5: SynFoC UNet Only (No SAM)"
echo "============================================"
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_UNet_Only_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --deterministic 1

# ===================== 实验6: SynFoC + 域课程学习 (无LLM) =====================
echo "============================================"
echo "Exp6: SynFoC + Domain Curriculum (No LLM)"
echo "============================================"
python /app/MixDSemi/SynFoC/code/train.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name Ablation_SynFoC_DC_NoLLM_v1 --gpu 3 --save_model --overwrite --max_iterations 30000 --num_eval_iter 500 --base_lr 0.03 --threshold 0.95 --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expend_test_samples 256 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 256 --use_symgd --symgd_mode full --conf_strategy robust --use_freq_aug --deterministic 1 --use_curr_conf --use_next_conf

################################################################################
# 所有实验完成
################################################################################
echo "============================================"
echo "All ablation experiments completed!"
echo "============================================"
