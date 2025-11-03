# Synfoc 低步数

screen -dmS synfoc-prostate-d6 bash -c "python train_synfoc.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name SynFoC_d6_8000 --gpu 1 --AdamW --model MedSAM --save_model --overwrite  --max_iterations 8000"

screen -dmS synfoc-prostate-d5 bash -c "python train_synfoc.py --dataset prostate --lb_domain 5 --lb_num 20 --save_name SynFoC_d5_8000 --gpu 1 --AdamW --model MedSAM --save_model --overwrite  --max_iterations 8000"

screen -dmS synfoc-prostate-d4 bash -c "python train_synfoc.py --dataset prostate --lb_domain 4 --lb_num 20 --save_name SynFoC_d4_8000 --gpu 2 --AdamW --model MedSAM --save_model --overwrite  --max_iterations 8000"

screen -dmS synfoc-prostate-d3 bash -c "python train_synfoc.py --dataset prostate --lb_domain 3 --lb_num 20 --save_name SynFoC_d3_8000 --gpu 3 --AdamW --model MedSAM --save_model --overwrite  --max_iterations 8000"

screen -dmS synfoc-prostate-d2 bash -c "python train_synfoc.py --dataset prostate --lb_domain 2 --lb_num 20 --save_name SynFoC_d2_8000 --gpu 2 --AdamW --model MedSAM --save_model --overwrite  --max_iterations 8000"

screen -dmS synfoc-prostate-d1 bash -c "python train_synfoc.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name SynFoC_d1_8000 --gpu 3 --AdamW --model MedSAM --save_model --overwrite  --max_iterations 8000"


screen -dmS synfoc-prostate-d6-val bash -c "python train_synfoc.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name SynFoC_d6_val --gpu 0 --AdamW --model MedSAM --save_model --overwrite "

# U-Net-only 低步数
python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_8000 --gpu 2 --save_model --overwrite  --max_iterations 8000

# Test domain 6 model (domain will be auto-detected from save_name)
python test_unet_only.py --dataset prostate --save_name UNet_only_d6_8000 --gpu 2 --lb_domain 6


# U-Net-only 简易版 完整
python train_unet_only_UCP.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_UCP_d6 --gpu 2 --save_model --overwrite 

screen -dmS unet-ucp-prostate-d6 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_UCP_d6 --gpu 2 --save_model --overwrite "

screen -dmS unet-ucp-prostate-d5 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 5 --lb_num 20 --save_name UNet_UCP_d5 --gpu 2 --save_model --overwrite "
screen -dmS unet-ucp-prostate-d4 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 4 --lb_num 20 --save_name UNet_UCP_d4 --gpu 2 --save_model --overwrite "
screen -dmS unet-ucp-prostate-d3 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 3 --lb_num 20 --save_name UNet_UCP_d3 --gpu 2 --save_model --overwrite "
screen -dmS unet-ucp-prostate-d2 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 2 --lb_num 20 --save_name UNet_UCP_d2 --gpu 0 --save_model --overwrite "
screen -dmS unet-ucp-prostate-d1 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name UNet_UCP_d1 --gpu 2 --save_model --overwrite "


screen -dmS unet-ucp-mnms-d1 bash -c "python train_unet_only_UCP.py --dataset MNMS --lb_domain 1 --lb_num 5 --save_name UNet_UCP_d1 --gpu 2 --save_model --overwrite"
screen -dmS unet-ucp-mnms-d2 bash -c "python train_unet_only_UCP.py --dataset MNMS --lb_domain 2 --lb_num 5 --save_name UNet_UCP_d2 --gpu 2 --save_model --overwrite"
screen -dmS unet-ucp-mnms-d3 bash -c "python train_unet_only_UCP.py --dataset MNMS --lb_domain 3 --lb_num 5 --save_name UNet_UCP_d3 --gpu 2 --save_model --overwrite"
screen -dmS unet-ucp-mnms-d4 bash -c "python train_unet_only_UCP.py --dataset MNMS --lb_domain 4 --lb_num 5 --save_name UNet_UCP_d4 --gpu 2 --save_model --overwrite"

screen -dmS unet-ucp-fundus-d1 bash -c "python train_unet_only_UCP.py --dataset fundus --lb_domain 1 --lb_num 20 --save_name UNet_UCP_d1 --gpu 3 --save_model --overwrite"
screen -dmS unet-ucp-fundus-d2 bash -c "python train_unet_only_UCP.py --dataset fundus --lb_domain 2 --lb_num 20 --save_name UNet_UCP_d2 --gpu 3 --save_model --overwrite"
screen -dmS unet-ucp-fundus-d3 bash -c "python train_unet_only_UCP.py --dataset fundus --lb_domain 3 --lb_num 20 --save_name UNet_UCP_d3 --gpu 3 --save_model --overwrite"
screen -dmS unet-ucp-fundus-d4 bash -c "python train_unet_only_UCP.py --dataset fundus --lb_domain 4 --lb_num 20 --save_name UNet_UCP_d4 --gpu 3 --save_model --overwrite"


screen -dmS unet-ucp-busi-d1-64 bash -c "python train_unet_only_UCP.py --dataset BUSI --lb_domain 1 --lb_num 64 --save_name UNet_UCP_d1_64 --gpu 0 --save_model --overwrite"
screen -dmS unet-ucp-busi-d2-64 bash -c "python train_unet_only_UCP.py --dataset BUSI --lb_domain 2 --lb_num 64 --save_name UNet_UCP_d2_64 --gpu 0 --save_model --overwrite"

screen -dmS unet-ucp-busi-d1-129 bash -c "python train_unet_only_UCP.py --dataset BUSI --lb_domain 1 --lb_num 129 --save_name UNet_UCP_d1_129 --gpu 0 --save_model --overwrite"
screen -dmS unet-ucp-busi-d2-129 bash -c "python train_unet_only_UCP.py --dataset BUSI --lb_domain 2 --lb_num 129 --save_name UNet_UCP_d2_129 --gpu 0 --save_model --overwrite"


screen -dmS unet-ucp-symGD-prostate-d6 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_UCP_d6_symGD --gpu 3 --save_model --overwrite "
screen -dmS unet-ucp-symGD-prostate-d5 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 5 --lb_num 20 --save_name UNet_UCP_d5_symGD --gpu 3 --save_model --overwrite "
screen -dmS unet-ucp-symGD-prostate-d4 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 4 --lb_num 20 --save_name UNet_UCP_d4_symGD --gpu 3 --save_model --overwrite "
screen -dmS unet-ucp-symGD-prostate-d3 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 3 --lb_num 20 --save_name UNet_UCP_d3_symGD --gpu 3 --save_model --overwrite "
screen -dmS unet-ucp-symGD-prostate-d2 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 2 --lb_num 20 --save_name UNet_UCP_d2_symGD --gpu 1 --save_model --overwrite "
screen -dmS unet-ucp-symGD-prostate-d1 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name UNet_UCP_d1_symGD --gpu 2 --save_model --overwrite "

screen -dmS unet-ucp-symGD-mnms-d4 bash -c "python train_unet_only_UCP_symGD.py --dataset MNMS --lb_domain 4 --lb_num 5 --save_name UNet_UCP_d4_symGD --gpu 2 --save_model --overwrite"
screen -dmS unet-ucp-symGD-mnms-d3 bash -c "python train_unet_only_UCP_symGD.py --dataset MNMS --lb_domain 3 --lb_num 5 --save_name UNet_UCP_d3_symGD --gpu 2 --save_model --overwrite"
screen -dmS unet-ucp-symGD-mnms-d2 bash -c "python train_unet_only_UCP_symGD.py --dataset MNMS --lb_domain 2 --lb_num 5 --save_name UNet_UCP_d2_symGD --gpu 2 --save_model --overwrite"
screen -dmS unet-ucp-symGD-mnms-d1 bash -c "python train_unet_only_UCP_symGD.py --dataset MNMS --lb_domain 1 --lb_num 5 --save_name UNet_UCP_d1_symGD --gpu 2 --save_model --overwrite"

screen -dmS unet-ucp-symGD-fundus-d4 bash -c "python train_unet_only_UCP_symGD.py --dataset fundus --lb_domain 4 --lb_num 20 --save_name UNet_UCP_d4_symGD --gpu 3 --save_model --overwrite"
screen -dmS unet-ucp-symGD-fundus-d3 bash -c "python train_unet_only_UCP_symGD.py --dataset fundus --lb_domain 3 --lb_num 20 --save_name UNet_UCP_d3_symGD --gpu 3 --save_model --overwrite"
screen -dmS unet-ucp-symGD-fundus-d2 bash -c "python train_unet_only_UCP_symGD.py --dataset fundus --lb_domain 2 --lb_num 20 --save_name UNet_UCP_d2_symGD --gpu 1 --save_model --overwrite"
screen -dmS unet-ucp-symGD-fundus-d1 bash -c "python train_unet_only_UCP_symGD.py --dataset fundus --lb_domain 1 --lb_num 20 --save_name UNet_UCP_d1_symGD --gpu 0 --save_model --overwrite"



screen -dmS unet-ucp-symGD-busi-d1-64 bash -c "python train_unet_only_UCP_symGD.py --dataset BUSI --lb_domain 1 --lb_num 64 --save_name UNet_UCP_d1_64_symGD --gpu 1 --save_model --overwrite"
screen -dmS unet-ucp-symGD-busi-d2-64 bash -c "python train_unet_only_UCP_symGD.py --dataset BUSI --lb_domain 2 --lb_num 64 --save_name UNet_UCP_d2_64_symGD --gpu 1 --save_model --overwrite"

screen -dmS unet-ucp-symGD-busi-d1-129 bash -c "python train_unet_only_UCP_symGD.py --dataset BUSI --lb_domain 1 --lb_num 129 --save_name UNet_UCP_d1_129_symGD --gpu 1 --save_model --overwrite"
screen -dmS unet-ucp-symGD-busi-d2-129 bash -c "python train_unet_only_UCP_symGD.py --dataset BUSI --lb_domain 2 --lb_num 129 --save_name UNet_UCP_d2_129_symGD --gpu 1 --save_model --overwrite"



#  UCP_DC_v2

python train_unet_only_UCP_DC_v2.py \
  --dataset prostate \
  --lb_domain 6 \
  --lb_num 20 \
  --save_name UCP_DC_v2_d6_debug \
  --gpu 0 \
  --save_model \
  --overwrite \
  --dc_parts 5 \
  --expend_test_steps_interval 250 \
  --expend_max_steps 2000 \
  --expand_conf_threshold 0.6 \
  --expand_conf_weight 1.0 \
  --expend_test_samples 32 \
  --curr_conf_weight 1.0 \
  --curr_conf_threshold 0.75 \
  --curr_conf_samples 64 \
  --ul_weight 1.0 \
  --lu_weight 1.0 \
  --cons_weight 1.0 \
  --llm_model gemini \
  --describe_nums 80 



screen -dmS unet-ucp-symGD-DC-5-busi-d1-129 bash -c python train_unet_only_UCP_DC_v2.py   --dataset prostate   --lb_domain 6   --lb_num 20   --save_name Unet_UCP_symGD_5_DC_v2_d6   --gpu 0   --save_model   --overwrite   --dc_parts 5   --expend_test_steps_interval 250   --expend_max_steps 2000   --expand_conf_threshold 0.6   --expand_conf_weight 1.0   --expend_test_samples 32   --curr_conf_weight 1.0   --curr_conf_threshold 0.75   --curr_conf_samples 64   --ul_weight 1.0   --lu_weight 1.0   --cons_weight 1.0   --llm_model gemini   --describe_nums 80 





python train_MiDSS_NDC_MARK2_CLIP.py --dataset prostate --lb_num 20 --lb_domain 1 --max_iterations 10000 --biomedclip_path /root/models/BiomedCLIP --gpu 0 --save_model --save_name vpt_basic_test_d1 --seed 1337 --label_bs 4 --unlabel_bs 4 --text_num_subsets 4 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --clip_loss_mv_anchor_weight 1.0 --clip_loss_ortho_weight 1.0 --clip_loss_sw_reg_weight 1.0 --biomedclip_num_prompts 4 --biomedclip_embed_dim 768 --biomedclip_init_std 0.02 --biomedclip_prompt_scale_init 1.0 --biomedclip_lr 0.0001 --biomedclip_weight_decay 0.01 --overwrite




python train_MiDSS_NDC_MARK2_CLIP.py --dataset prostate --lb_num 20 --lb_domain 6 --max_iterations 10000 --biomedclip_path /root/models/BiomedCLIP --gpu 1 --save_model --save_name vpt_basic_test_d6 --seed 1337 --label_bs 4 --unlabel_bs 4 --text_num_subsets 4 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --clip_loss_mv_anchor_weight 1.0 --clip_loss_ortho_weight 1.0 --clip_loss_sw_reg_weight 1.0 --biomedclip_num_prompts 4 --biomedclip_embed_dim 768 --biomedclip_init_std 0.02 --biomedclip_prompt_scale_init 1.0 --biomedclip_lr 0.0001 --biomedclip_weight_decay 0.01 --overwrite

#####

python train_MiDSS_NDC_MARK2_CLIP_DEEP.py --dataset prostate --lb_num 20 --lb_domain 1 --max_iterations 2000 --biomedclip_path /root/models/BiomedCLIP --gpu 2 --save_model --save_name vpt_basic_test_d1 --seed 1337 --label_bs 4 --unlabel_bs 4 --text_num_subsets 4 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --clip_loss_mv_anchor_weight 1.0 --clip_loss_ortho_weight 1.0 --clip_loss_sw_reg_weight 1.0 --biomedclip_num_prompts 4 --biomedclip_embed_dim 768 --biomedclip_init_std 0.02 --biomedclip_prompt_scale_init 1.0 --biomedclip_lr 0.0001 --biomedclip_weight_decay 0.01 --overwrite


python train_MiDSS_NDC_MARK2_CLIP_DEEP.py --dataset prostate --lb_num 20 --lb_domain 6 --max_iterations 2000 --biomedclip_path /root/models/BiomedCLIP --gpu 3 --save_model --save_name vpt_basic_test_d6 --seed 1337 --label_bs 4 --unlabel_bs 4 --text_num_subsets 4 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --clip_loss_mv_anchor_weight 1.0 --clip_loss_ortho_weight 1.0 --clip_loss_sw_reg_weight 1.0 --biomedclip_num_prompts 4 --biomedclip_embed_dim 768 --biomedclip_init_std 0.02 --biomedclip_prompt_scale_init 1.0 --biomedclip_lr 0.0001 --biomedclip_weight_decay 0.01 --overwrite



python train_MiDSS_NDC_MARK2_CLIP_DEEP.py --dataset prostate --lb_num 20 --lb_domain 1 --max_iterations 2000 --biomedclip_path /root/models/BiomedCLIP --gpu 1 --save_model --save_name vpt_basic_test_d1_vpt6 --seed 1337 --label_bs 4 --unlabel_bs 4 --text_num_subsets 4 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --clip_loss_mv_anchor_weight 1.0 --clip_loss_ortho_weight 1.0 --clip_loss_sw_reg_weight 1.0 --biomedclip_num_prompts 6 --biomedclip_embed_dim 768 --biomedclip_init_std 0.02 --biomedclip_prompt_scale_init 1.0 --biomedclip_lr 0.0001 --biomedclip_weight_decay 0.01 --overwrite


python train_MiDSS_NDC_MARK2_CLIP_DEEP_INV.py --dataset prostate --lb_num 20 --lb_domain 1 --max_iterations 2000 --biomedclip_path /root/models/BiomedCLIP --gpu 1 --save_model --save_name vpt_basic_test_d1 --seed 1337 --label_bs 4 --unlabel_bs 4 --text_num_subsets 4 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --clip_loss_mv_anchor_weight 1.0 --clip_loss_ortho_weight 1.0 --clip_loss_sw_reg_weight 1.0 --biomedclip_num_prompts 4 --biomedclip_embed_dim 768 --biomedclip_init_std 0.02 --biomedclip_prompt_scale_init 1.0 --biomedclip_lr 0.0001 --biomedclip_weight_decay 0.01 --overwrite


python train_MiDSS_NDC_MARK2_CLIP_DEEP_INV.py --dataset prostate --lb_num 20 --lb_domain 1 --max_iterations 500 --biomedclip_path /root/models/BiomedCLIP --gpu 2 --save_model --save_name vpt_basic_test_d1_vpt6 --seed 1337 --label_bs 4 --unlabel_bs 4 --text_num_subsets 4 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model GPT5 --describe_nums 80 --clip_loss_mv_anchor_weight 1.0 --clip_loss_sw_reg_weight 1.0 --biomedclip_num_prompts 4 --biomedclip_embed_dim 768 --biomedclip_init_std 0.02 --biomedclip_prompt_scale_init 1.0 --biomedclip_lr 0.0001 --biomedclip_weight_decay 0.01 --overwrite


python train_MiDSS_NDC_MARK2_CLIP_DEEP_INV_MASK.py --dataset prostate --save_name vpt_invariant_test_d1_crop --overwrite --save_model --gpu 0 --seed 1337 --max_iterations 2000 --lb_domain 1 --lb_num 40 --lb_ratio 0.0 --preprocess_dir /app/MixDSemi/SynFoCLIP/preprocess/ProstateSlice --llm_model gemini --describe_nums 40 --clip_loss_mv_anchor_weight 1.0 --clip_loss_sw_reg_weight 1.0 --clip_loss_crop_weight 0.5 --crop_morph_k_min 3 --crop_morph_k_max 15 --biomedclip_path /root/models/BiomedCLIP --biomedclip_num_prompts 4 --biomedclip_embed_dim 768 --biomedclip_init_std 0.02 --biomedclip_prompt_scale_init 1.0 --biomedclip_lr 0.0001 --biomedclip_weight_decay 0.01 --text_root /app/MixDSemi/SynFoCLIP/code/text --text_num_subsets 4 --save_name complete_crop_test --overwrite



