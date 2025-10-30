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



CUDA_VISIBLE_DEVICES=3 python train_unet_MiDSS_DC_v3_dunet.py --dataset prostate --save_name MIDSS_DC_CLIP_D1_RO_NoFG_dunet --overwrite --save_model --gpu 3 --seed 1337 --max_iterations 20000 --num_eval_iter 500 --warmup --warmup_period 2000 --domain_num 6 --lb_domain 1 --lb_num 20 --use_freq_aug --LB 0.01 --ema_decay 0.99 --threshold 0.95 --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --use_symgd --symgd_mode full --ul_weight 1.0 --lu_weight 1.0 --cons_weight 1.0 --dc_parts 5 --dc_distance_mode sqrt_prod --expend_test_samples 128 --expend_test_steps_interval 250 --expend_max_steps 1250 --use_curr_conf --curr_conf_threshold 0.5 --curr_conf_samples 128 --use_next_conf --expand_conf_threshold 0.5 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/MNMS --llm_model GPT5 --describe_nums 80 --conf_strategy robust_no_fg --conf_teacher_temp 1.5 --du_bilinear --du_style_dim 512 --du_modulation conditional --du_r_rank 8 --du_alpha_schedule "0.04,0.08,0.12,0.16" --du_disable_style_head --du_disable_invariance_head --consistency 1.0 --consistency_rampup 200.0 --clip_loss_mv_anchor_weight 1.0 --clip_loss_ortho_weight 1.0 --clip_loss_sw_reg_weight 1.0 --distill_weight 1.0 --distill_mode both --biomedclip_path /root/models/BiomedCLIP --biomedclip_num_prompts 4 --biomedclip_embed_dim 768 --biomedclip_init_std 0.02 --biomedclip_prompt_scale_init 1.0 --biomedclip_disable_scale --biomedclip_lr 1e-4 --biomedclip_weight_decay 1e-2 --text_root /app/MixDSemi/SynFoCLIP/code/text --text_num_subsets 4



# ============ 最小版本（仅关键参数）============
# 使用代码预定义的默认参数最多的版本
python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_5_DC_v2_d1_v7_2_cu1_30000_gpt5_80 --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.80 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 5000 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --conf_strategy dice --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000



