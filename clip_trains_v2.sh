python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D1 --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 128 --conf_strategy robust --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    

python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D6 --gpu 2 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 128 --conf_strategy robust --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    


python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D1_2 --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 128 --conf_strategy dice --no_symgd --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    

python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D1_3 --gpu 2 --save_model --overwrite --dc_parts 5 --dc_distance_mode prototype --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 128 --conf_strategy dice --no_symgd --llm_model GPT5 --describe_nums 80 --use_freq_aug --max_iterations 30000 --use_next_conf --use_curr_conf --warmup --warmup_period 2000    


screen -dmS MIDSS_DC_DEBUG_D1 bash -c "python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D1 --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 128 --conf_strategy robust --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --use_next_conf --use_curr_conf --warmup --warmup_period 2000 "

screen -dmS MIDSS_DC_DEBUG_D6 bash -c "python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name MIDSS_DC_NEW_DEBUG_D6 --gpu 2 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 128 --expand_conf_threshold 0.75 --curr_conf_threshold 0.75 --curr_conf_samples 128 --conf_strategy robust --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug  --use_next_conf --use_curr_conf --warmup --warmup_period 2000 "



screen -dmS MIDSS_DC_D1 bash -c "python train_unet_MiDSS_DC_v2.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name MIDSS_DC_NEW_D1 --gpu 3 --save_model --overwrite --dc_parts 5 --dc_distance_mode sqrt_prod --enable_piecewise_tau --tau_min 0.85 --tau_max 0.95 --expend_test_steps_interval 300 --expend_max_steps 1200 --expend_test_samples 64 --expand_conf_threshold 0.7 --curr_conf_threshold 0.7 --curr_conf_samples 128 --conf_strategy robust --use_symgd --symgd_mode full --llm_model GPT5 --describe_nums 80 --use_freq_aug --use_next_conf --use_curr_conf --warmup --warmup_period 2000 "


python train_MiDSS_MARK3_CLIP_VPT_TPT_1_SEMI_EMA.py --dataset prostate --save_name vpt_tpt_experiment_10000_3 --overwrite --save_model --gpu 2 --seed 1337  --max_iterations 10000 --lb_domain 1 --lb_num 20 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 6 --text_num_prompts 12 --vpt_tpt_lr 1e-4 --vpt_tpt_weight_decay 1e-2 --freeze_backbone --text_root /app/MixDSemi/SynFoCLIP/code/text-an --lambda_sup 1.0 --lambda_consis 1.0 --feat_consis_scale 0.1 --T_consis 0.5


python train_MiDSS_MARK3_CLIP_VPT_TPT_1_SEMI_EMA.py --dataset prostate --save_name vpt_tpt_experiment_1000 --overwrite --save_model --gpu 2 --seed 1337  --max_iterations 1000 --lb_domain 1 --lb_num 20 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 4 --text_num_prompts 4 --vpt_tpt_lr 1e-4 --vpt_tpt_weight_decay 1e-2 --freeze_backbone --text_root /app/MixDSemi/SynFoCLIP/code/text-an --lambda_sup 1.0 --lambda_consis 1.0 --feat_consis_scale 0.1 --T_consis 0.8

python train_MiDSS_MARK3_CLIP_VPT_TPT_1_SEMI_EMA.py --dataset prostate --save_name vpt_tpt_experiment_10000_2 --overwrite --save_model --gpu 2 --seed 1337  --max_iterations 10000 --lb_domain 1 --lb_num 20 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 4 --text_num_prompts 4 --vpt_tpt_lr 1e-4 --vpt_tpt_weight_decay 1e-2 --freeze_backbone --text_root /app/MixDSemi/SynFoCLIP/code/text-an --lambda_sup 1.0 --lambda_consis 1.0 --feat_consis_scale 0.1 --T_consis 0.2


python test_MiDSS_MARK3_CLIP_VPT_TPT_1_SEMI_EMA.py --dataset prostate --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 6 --text_num_prompts 12 --freeze_backbone --prompts_path ../model/prostate/train_MiDSS_MARK3_CLIP_VPT_TPT_1_SEMI_EMA/vpt_tpt_experiment_10000_3/vpt_tpt_teacher_final_weights.pth --output_dir ../results/prostate_test_vpt_tpt_ema --text_root /app/MixDSemi/SynFoCLIP/code/text-an --preprocess_root /app/MixDSemi/SynFoCLIP/preprocess --llm_model GPT5 --describe_nums 80 --gpu 2 --batch_size 1 --num_workers 2 --seed 1337 --overwrite



python train_MiDSS_MARK3_CLIP_VPT_TPT_2_SEMI_EMA.py --dataset prostate --save_name vpt_tpt_experiment_2_10000_3 --overwrite --save_model --gpu 2 --seed 1337  --max_iterations 10000 --lb_domain 1 --lb_num 20 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 6 --text_num_prompts 12 --vpt_tpt_lr 1e-4 --vpt_tpt_weight_decay 1e-2 --freeze_backbone --text_root /app/MixDSemi/SynFoCLIP/code/text-an --lambda_sup 1.0 --lambda_consis 1.0 --feat_consis_scale 0.1 --T_consis 0.5

python train_MiDSS_MARK3_CLIP_VPT_TPT_3_SEMI_EMA.py --dataset prostate --save_name vpt_tpt_experiment_3_10000_3 --overwrite --save_model --gpu 3 --seed 1337 --max_iterations 10000 --lb_domain 1 --lb_num 20 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 6 --text_num_prompts 12 --vpt_tpt_lr 1e-4 --vpt_tpt_weight_decay 1e-2 --freeze_backbone --text_root /app/MixDSemi/SynFoCLIP/code/text-an --lambda_ce 1.0 --lambda_dice 50.0 --lambda_consis 1.0 --lambda_sccm 0.1 --feat_consis_scale 0.1 --T_consis 0.5 --ema_decay 0.999 --tau 0.9

python train_MiDSS_MARK3_CLIP_VPT_TPT_3_SEMI_EMA.py --dataset prostate --save_name vpt_tpt_experiment_3_10000_2 --overwrite --save_model --gpu 2 --seed 1337 --max_iterations 10000 --lb_domain 1 --lb_num 20 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 4 --text_num_prompts 4 --vpt_tpt_lr 1e-4 --vpt_tpt_weight_decay 1e-2 --freeze_backbone --text_root /app/MixDSemi/SynFoCLIP/code/text-an --lambda_ce 1.0 --lambda_dice 50.0 --lambda_consis 1.0 --lambda_sccm 0.1 --feat_consis_scale 0.1 --T_consis 0.5 --ema_decay 0.999 --tau 0.9


python train_MiDSS_MARK3_CLIP_VPT_TPT_3_SEMI_EMA.py --dataset prostate --save_name vpt_tpt_experiment_3_10000_2 --overwrite --save_model --gpu 2 --seed 1337 --max_iterations 10000 --lb_domain 1 --lb_num 20 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 4 --text_num_prompts 4 --vpt_tpt_lr 1e-4 --vpt_tpt_weight_decay 1e-2 --freeze_backbone --text_root /app/MixDSemi/SynFoCLIP/code/text-an --lambda_ce 1.0 --lambda_dice 50.0 --lambda_consis 1.0 --lambda_sccm 0.1 --feat_consis_scale 0.1 --T_consis 0.5 --ema_decay 0.999 --tau 0.9


python test_MiDSS_MARK3_CLIP_VPT_TPT_1_SEMI_EMA.py --dataset prostate --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 4 --text_num_prompts 4 --freeze_backbone --prompts_path ../model/prostate/train_MiDSS_MARK3_CLIP_VPT_TPT_3_SEMI_EMA/vpt_tpt_experiment_3_10000_2/vpt_tpt_student_iter001000.pth --output_dir ../results/prostate_test_vpt_tpt_ema_4 --text_root /app/MixDSemi/SynFoCLIP/code/text-an --preprocess_root /app/MixDSemi/SynFoCLIP/preprocess --llm_model GPT5 --describe_nums 80 --gpu 3 --batch_size 1 --num_workers 2 --seed 1337 --overwrite



python test_MiDSS_MARK3_CLIP_VPT_TPT_1_SEMI_EMA.py --dataset prostate --biomedclip_path /root/models/BiomedCLIP --visual_num_prompts 6 --text_num_prompts 12 --freeze_backbone --prompts_path ../model/prostate/train_MiDSS_MARK3_CLIP_VPT_TPT_3_SEMI_EMA/vpt_tpt_experiment_3_10000_3/vpt_tpt_teacher_iter003000.pth --output_dir ../results/prostate_test_vpt_tpt_ema_4 --text_root /app/MixDSemi/SynFoCLIP/code/text-an --preprocess_root /app/MixDSemi/SynFoCLIP/preprocess --llm_model GPT5 --describe_nums 80 --gpu 3 --batch_size 1 --num_workers 2 --seed 1337 --overwrite

/app/MixDSemi/SynFoCLIP/model/prostate/train_MiDSS_MARK3_CLIP_TPT_3_SEMI_EMA/tpt_experiment_3_10000_4/tpt_student_iter001000.pth
python test_MiDSS_MARK3_CLIP_TPT_1_SEMI_EMA.py --dataset prostate --biomedclip_path /root/models/BiomedCLIP --text_num_prompts 20 --freeze_backbone --prompts_path ../model/prostate/train_MiDSS_MARK3_CLIP_TPT_3_SEMI_EMA/tpt_experiment_3_10000_4/tpt_student_iter001000.pth --output_dir ../results/prostate_test_tpt_ema_4 --text_root /app/MixDSemi/SynFoCLIP/code/text-an --preprocess_root /app/MixDSemi/SynFoCLIP/preprocess --llm_model GPT5 --describe_nums 80 --gpu 3 --batch_size 1 --num_workers 2 --seed 1337 --overwrite


python train_MiDSS_MARK3_CLIP_TPT_3_SEMI_EMA.py --dataset prostate --save_name tpt_experiment_3_10000_4 --overwrite --save_model --gpu 3 --seed 1337 --max_iterations 10000 --lb_domain 1 --lb_num 20 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --biomedclip_path /root/models/BiomedCLIP --text_num_prompts 20 --tpt_lr 1e-4 --tpt_weight_decay 1e-2 --freeze_backbone --text_root /app/MixDSemi/SynFoCLIP/code/text-an --lambda_ce 1.0 --lambda_dice 50.0 --lambda_consis 1.0 --lambda_sccm 0.1 --feat_consis_scale 0.1 --T_consis 0.5 --ema_decay 0.999 --tau 0.9 --save_every 1000


conda activate mixsemi
python visualize_biomedclip.py \
  --dataset prostate \
  --biomedclip_path /root/models/BiomedCLIP \
  --gpu 0 \
  --output_dir /app/MixDSemi/SynFoCLIP/results/debug1 \
  --per_domain 10 \
  --num_workers 2



python visualize_patches_batch.py --dataset fundus --batch-size 5 --num-workers 0 --patch-size 256 --lb-domain 1 --lb-num 20 --debug-mask-uniques && \
python visualize_patches_batch.py --dataset prostate --batch-size 5 --num-workers 0 --patch-size 256 --lb-domain 1 --lb-num 20 --debug-mask-uniques && \
python visualize_patches_batch.py --dataset MNMS --batch-size 5 --num-workers 0 --patch-size 256 --lb-domain 1 --lb-num 20 --debug-mask-uniques && \
python visualize_patches_batch.py --dataset BUSI --batch-size 5 --num-workers 0 --patch-size 256 --lb-domain 1 --lb-num 20 --debug-mask-uniques




python visualize_patches_batch.py --dataset fundus --batch-size 5 --num-workers 0 --patch-size 256 --lb-domain 4 --lb-num 20 --debug-mask-uniques && \
python visualize_patches_batch.py --dataset prostate --batch-size 5 --num-workers 0 --patch-size 256 --lb-domain 6 --lb-num 20 --debug-mask-uniques && \
python visualize_patches_batch.py --dataset MNMS --batch-size 5 --num-workers 0 --patch-size 256 --lb-domain 4 --lb-num 20 --debug-mask-uniques && \
python visualize_patches_batch.py --dataset BUSI --batch-size 5 --num-workers 0 --patch-size 256 --lb-domain 2 --lb-num 20 --debug-mask-uniques
