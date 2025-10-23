# Synfoc 低步数

screen -dmS synfoc-prostate-d6 bash -c "python train_synfoc.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name SynFoC_d6_8000 --gpu 1 --AdamW --model MedSAM --save_model --overwrite --save_img --max_iterations 8000"

screen -dmS synfoc-prostate-d5 bash -c "python train_synfoc.py --dataset prostate --lb_domain 5 --lb_num 20 --save_name SynFoC_d5_8000 --gpu 1 --AdamW --model MedSAM --save_model --overwrite --save_img --max_iterations 8000"

screen -dmS synfoc-prostate-d4 bash -c "python train_synfoc.py --dataset prostate --lb_domain 4 --lb_num 20 --save_name SynFoC_d4_8000 --gpu 2 --AdamW --model MedSAM --save_model --overwrite --save_img --max_iterations 8000"

screen -dmS synfoc-prostate-d3 bash -c "python train_synfoc.py --dataset prostate --lb_domain 3 --lb_num 20 --save_name SynFoC_d3_8000 --gpu 3 --AdamW --model MedSAM --save_model --overwrite --save_img --max_iterations 8000"

screen -dmS synfoc-prostate-d2 bash -c "python train_synfoc.py --dataset prostate --lb_domain 2 --lb_num 20 --save_name SynFoC_d2_8000 --gpu 2 --AdamW --model MedSAM --save_model --overwrite --save_img --max_iterations 8000"

screen -dmS synfoc-prostate-d1 bash -c "python train_synfoc.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name SynFoC_d1_8000 --gpu 3 --AdamW --model MedSAM --save_model --overwrite --save_img --max_iterations 8000"


screen -dmS synfoc-prostate-d6-val bash -c "python train_synfoc.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name SynFoC_d6_val --gpu 0 --AdamW --model MedSAM --save_model --overwrite --save_img"

# U-Net-only 低步数
python train_unet_only.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_only_d6_8000 --gpu 2 --save_model --overwrite --save_img --max_iterations 8000

# Test domain 6 model (domain will be auto-detected from save_name)
python test_unet_only.py --dataset prostate --save_name UNet_only_d6_8000 --gpu 2 --lb_domain 6


# U-Net-only 简易版 完整
python train_unet_only_UCP.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_UCP_d6 --gpu 2 --save_model --overwrite --save_img

screen -dmS unet-ucp-prostate-d6 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_UCP_d6 --gpu 2 --save_model --overwrite --save_img"

screen -dmS unet-ucp-prostate-d5 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 5 --lb_num 20 --save_name UNet_UCP_d5 --gpu 2 --save_model --overwrite --save_img"
screen -dmS unet-ucp-prostate-d4 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 4 --lb_num 20 --save_name UNet_UCP_d4 --gpu 2 --save_model --overwrite --save_img"
screen -dmS unet-ucp-prostate-d3 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 3 --lb_num 20 --save_name UNet_UCP_d3 --gpu 2 --save_model --overwrite --save_img"
screen -dmS unet-ucp-prostate-d2 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 2 --lb_num 20 --save_name UNet_UCP_d2 --gpu 0 --save_model --overwrite --save_img"
screen -dmS unet-ucp-prostate-d1 bash -c "python train_unet_only_UCP.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name UNet_UCP_d1 --gpu 2 --save_model --overwrite --save_img"


python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_UCP_d6_symGD --gpu 3 --save_model --overwrite --save_img

screen -dmS unet-ucp-symGD-prostate-d6 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 6 --lb_num 20 --save_name UNet_UCP_d6_symGD --gpu 3 --save_model --overwrite --save_img"
screen -dmS unet-ucp-symGD-prostate-d5 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 5 --lb_num 20 --save_name UNet_UCP_d5_symGD --gpu 3 --save_model --overwrite --save_img"
screen -dmS unet-ucp-symGD-prostate-d4 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 4 --lb_num 20 --save_name UNet_UCP_d4_symGD --gpu 3 --save_model --overwrite --save_img"
screen -dmS unet-ucp-symGD-prostate-d3 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 3 --lb_num 20 --save_name UNet_UCP_d3_symGD --gpu 3 --save_model --overwrite --save_img"
screen -dmS unet-ucp-symGD-prostate-d2 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 2 --lb_num 20 --save_name UNet_UCP_d2_symGD --gpu 1 --save_model --overwrite --save_img"
screen -dmS unet-ucp-symGD-prostate-d1 bash -c "python train_unet_only_UCP_symGD.py --dataset prostate --lb_domain 1 --lb_num 20 --save_name UNet_UCP_d1_symGD --gpu 2 --save_model --overwrite --save_img"



