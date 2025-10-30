#!/bin/bash

# ============ 简化版本（移除默认值参数）============
# 可选：如果想使用更简洁的命令（保持相同的训练效果）
python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D1 --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 128 --conf_strategy robust --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    

python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D6 --gpu 2 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 128 --conf_strategy robust --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    

python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D1_DC --gpu 1 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 128 --conf_strategy dice --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    

python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D1_JS --gpu 2 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.4 --curr_conf_threshold 0.4 --curr_conf_samples 128 --conf_strategy js_teacher_student --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    

python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D1_RO_NoFG --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.5 --curr_conf_threshold 0.5 --curr_conf_samples 128 --conf_strategy robust_no_fg --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    





python train_unet_MiDSS_DC_v2_dunet.py --dataset MNMS --lb_domain 1 --lb_num 5 --save_name MIDSS_DC_NEW_DEBUG_D1_RO_NoFG_dunet --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.5 --curr_conf_threshold 0.5 --curr_conf_samples 128 --conf_strategy robust_no_fg --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    

python train_unet_MiDSS_DC_v2_dunet.py --dataset MNMS --lb_domain 6 --lb_num 5 --save_name MIDSS_DC_NEW_DEBUG_D6_RO_NoFG_dunet --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.5 --curr_conf_threshold 0.5 --curr_conf_samples 128 --conf_strategy robust_no_fg --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000 




# ============ 最小版本（仅关键参数）============
# 使用代码预定义的默认参数最多的版本
python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_5_DC_v2_d1_v7_2_cu1_30000_gpt5_80 --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --conf_strategy dice --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000
