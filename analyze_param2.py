# tools/compute_trainable_params.py
import torch
from importlib import import_module
from segment_anything import sam_model_registry

# 1) MedSAM LoRA
def build_lora_sam(rank=4, vit_name='vit_b', ckpt_path='/app/MixDSemi/ckpt/sam_vit_b_01ec64.pth'):
    sam, img_embedding_size = sam_model_registry[vit_name](image_size=512, num_classes=1, checkpoint=ckpt_path,
                                                          pixel_mean=[0,0,0], pixel_std=[1,1,1])
    pkg = import_module('sam_lora_image_encoder')
    model = pkg.LoRA_Sam(sam, rank)
    return model

# 2) CLIP variants
def build_vpt_rd(num_prompts=4, embed_dim=768, biomedclip_path='/root/models/BiomedCLIP'):
    pkg = import_module('biomedclip_vpt_RD')
    model = pkg.VPT_CLIP_RD(model_path=biomedclip_path, num_prompts=num_prompts, embed_dim=embed_dim,
                             init_std=0.02, prompt_scale_init=1.0, enable_scale=True, device='cpu')
    return model

def build_vpt_invariant(num_prompts=4, embed_dim=768, biomedclip_path='/root/models/BiomedCLIP'):
    pkg = import_module('biomedclip_vpt_invariant_only')
    # 使用构造函数或 builder 根据文件实现
    model, preprocess, tokenizer = pkg.build_invariant_prompt_image_encoder(biomedclip_path, device='cpu',
                                                                             num_prompts=num_prompts,
                                                                             embed_dim=embed_dim)
    return model

def count_trainable_params(obj):
    total = sum(p.numel() for p in obj.parameters() if p.requires_grad)
    by_module = {}
    for name, p in obj.named_parameters():
        if p.requires_grad:
            prefix = name.split('.')[0]
            by_module.setdefault(prefix, 0)
            by_module[prefix] += p.numel()
    return total, by_module

if __name__ == '__main__':
    print('Counting (may need large mem to load weights) ...')
    # MedSAM LoRA
    sam_lora = build_lora_sam(rank=4)
    total_sam_lora, by_sam = count_trainable_params(sam_lora)
    print('MedSAM LoRA total trainable:', total_sam_lora)
    print('Breakdown (top-level prefixes):', by_sam)

    # VPT_RD
    vpt_rd = build_vpt_rd(num_prompts=4)
    total_vpt_rd, by_vpt_rd = count_trainable_params(vpt_rd)
    print('VPT_RD total trainable:', total_vpt_rd)
    print('Breakdown:', by_vpt_rd)

    # VPT_InvariantOnly
    vpt_inv = build_vpt_invariant(num_prompts=4)
    total_vpt_inv, by_vpt_inv = count_trainable_params(vpt_inv)
    print('VPT_InvariantOnly total trainable:', total_vpt_inv)
    print('Breakdown:', by_vpt_inv)