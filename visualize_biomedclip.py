import os
import argparse
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import dataloaders.custom_transforms as tr
from dataloaders.dataloader import (BUSISegmentation, FundusSegmentation,
                                   MNMSSegmentation, ProstateSegmentation)
from utils.clip import create_biomedclip_model_and_preprocess_local
from biomedclip_vpt_tpt_seg import preprocess_tensor_images


TEXTS = ["A photo of prostate", "A photo of T2w MRI"]


def get_class_and_modality(dataset: str):
    """Return (class_list, modality_str) for supported datasets."""
    if dataset == 'fundus':
        return ['background', 'optic cup', 'optic disc'], 'color fundus photograph'
    if dataset == 'prostate':
        return ['background', 'prostate'], 'T2-weighted MRI'
    if dataset == 'MNMS':
        return ['background', 'left ventricle', 'left ventricle myocardium', 'right ventricle'], 'cardiac MRI'
    if dataset == 'BUSI':
        return ['background', 'breast tumor'], 'breast ultrasound'
    # fallback: generic classes (background + numbered)
    return ['background'] + [f'class_{i}' for i in range(1, 1 + 1)], 'image'


def build_text_prototypes(model, tokenizer, classes: list, modality: str, device, templates_per_class: int = 4):
    """Build a prototype vector for each class by encoding multiple templates and averaging.

    Returns: Tensor [K, D]
    """
    # generic template set; will be formatted with {class} and {modality}
    base_templates = [
        'A {modality} showing the {classname}.',
        'A {classname} region in a {modality}.',
        'This {modality} image contains {classname}.',
        '{classname} in a {modality}.',
        'An image of the {classname} from a {modality}.',
        'A medical {modality} with visible {classname}.'
    ]

    per_class_templates = []
    for cls in classes:
        # choose first templates_per_class templates (cycle if needed)
        tlist = []
        for i in range(templates_per_class):
            tmpl = base_templates[i % len(base_templates)]
            # special-case background wording
            if cls.lower() == 'background':
                text = f'Background region in a {modality} image.'
            else:
                text = tmpl.format(classname=cls, modality=modality)
            tlist.append(text)
        per_class_templates.append(tlist)

    # encode templates per class and average
    prototypes = []
    with torch.no_grad():
        for tlist in per_class_templates:
            # tokenize batch
            try:
                tokens = tokenizer(tlist, context_length=256).to(device)
            except TypeError:
                tokens = tokenizer(tlist).to(device)
            feats = model.encode_text(tokens).float()  # [T, D]
            feats = F.normalize(feats, dim=-1)
            mean_feat = feats.mean(dim=0, keepdim=True)  # [1, D]
            mean_feat = F.normalize(mean_feat, dim=-1)
            prototypes.append(mean_feat)

    prototypes = torch.cat(prototypes, dim=0)  # [K, D]
    return prototypes


def get_dataset_config(dataset: str) -> dict:
    if dataset == 'fundus':
        return {
            'train_data_path': '/app/MixDSemi/data/Fundus',
            'dataset_class': FundusSegmentation,
            'num_channels': 3,
            'patch_size': 256,
            'num_classes': 2,
            'default_domain_num': 4,
        }
    if dataset == 'prostate':
        return {
            'train_data_path': '/app/MixDSemi/data/ProstateSlice',
            'dataset_class': ProstateSegmentation,
            'num_channels': 1,
            'patch_size': 384,
            'num_classes': 1,
            'default_domain_num': 6,
        }
    if dataset == 'MNMS':
        return {
            'train_data_path': '/app/MixDSemi/data/mnms',
            'dataset_class': MNMSSegmentation,
            'num_channels': 1,
            'patch_size': 288,
            'num_classes': 3,
            'default_domain_num': 4,
        }
    if dataset == 'BUSI':
        return {
            'train_data_path': '/app/MixDSemi/data/Dataset_BUSI_with_GT',
            'dataset_class': BUSISegmentation,
            'num_channels': 1,
            'patch_size': 256,
            'num_classes': 1,
            'default_domain_num': 2,
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def build_test_loader(cfg: dict, batch_size: int, num_workers: int, domains: List[int]) -> DataLoader:
    base_transform = transforms.Compose([
        tr.Normalize_tf(dataRange=[0, 1]),
        tr.ToTensor(unet_size=cfg['patch_size'])
    ])

    dataset = cfg['dataset_class'](
        base_dir=cfg['train_data_path'],
        phase='test',
        splitid=-1,
        domain=domains,
        weak_transform=None,
        strong_tranform=None,
        normal_toTensor=base_transform,
        img_size=cfg['patch_size'],
        is_RGB=(cfg['num_channels'] == 3),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def ensure_mask_shape(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    if mask.max() > 1.5:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:
            return tensor[0].numpy()
        return tensor.permute(1, 2, 0).numpy()
    if tensor.dim() == 2:
        return tensor.numpy()
    raise ValueError(f"Unexpected tensor shape for display: {tuple(tensor.shape)}")


def visualize_domain(domain_idx: int, samples: list, scores: list, out_path: str, class_names: list):
    # samples: list of image tensors (C,H,W)
    # scores: list of [score_text0, score_text1]
    n = len(samples)
    cols = 2
    fig_h = max(4, n * 1.5)
    fig, axes = plt.subplots(n, cols, figsize=(8 * cols, 2.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        img = tensor_to_display_image(samples[i])
        ax_img = axes[i, 0]
        cmap = 'gray' if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1) else None
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        ax_img.imshow(img, cmap=cmap)
        ax_img.axis('off')
        ax_img.set_title(f'Domain {domain_idx} - sample {i+1}')

        ax_bar = axes[i, 1]
        vals = np.array(scores[i])
        ax_bar.bar(np.arange(len(vals)), vals)
        ax_bar.set_ylim(0.0, 1.0)
        ax_bar.set_xticks(np.arange(len(vals)))
        # Use provided class names (may include modality as context separately)
        labels = [str(c) for c in class_names]
        ax_bar.set_xticklabels(labels, rotation=20)
        ax_bar.set_ylabel('softmax probability')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate', 'MNMS', 'BUSI'])
    parser.add_argument('--biomedclip_path', type=str, default='/root/models/BiomedCLIP')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--output_dir', type=str, default='/app/MixDSemi/SynFoCLIP/results')
    parser.add_argument('--per_domain', type=int, default=10, help='images per domain to visualize')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--templates_per_class', type=int, default=4,
                        help='Number of text templates to generate per class for prototype averaging')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    cfg = get_dataset_config(args.dataset)
    default_domains = list(range(1, cfg['default_domain_num'] + 1))

    # Load BiomedCLIP
    print(f'Loading BiomedCLIP from {args.biomedclip_path} on device {device}...')
    model, preprocess, tokenizer = create_biomedclip_model_and_preprocess_local(args.biomedclip_path, str(device))
    model = model.to(device)
    model.eval()

    # text features (encode once) - build per-class prototypes using multiple templates
    classes, modality = get_class_and_modality(args.dataset)
    print(f'Building text prototypes for classes={classes} modality="{modality}" using {args.templates_per_class} templates/class')
    text_feats = build_text_prototypes(model, tokenizer, classes, modality, device, templates_per_class=args.templates_per_class)
    # scale if available
    try:
        scale = model.logit_scale.exp().item()
    except Exception:
        scale = 1.0

    os.makedirs(args.output_dir, exist_ok=True)

    # If dataset is prostate we will read raw images directly from disk (per-domain directories)
    # For other datasets we fall back to the DataLoader behavior
    prostate_domain_names = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']

    for domain in default_domains:
        print(f'Processing domain {domain}...')
        samples = []
        scores = []
        taken = 0

        if args.dataset == 'prostate':
            # Construct directory: base_dir / domain_name / test / image /
            domain_idx = domain - 1
            if domain_idx < 0 or domain_idx >= len(prostate_domain_names):
                print(f'  Invalid domain index {domain}, skipping')
                continue
            domain_name = prostate_domain_names[domain_idx]
            img_dir = os.path.join(cfg['train_data_path'], domain_name, 'test', 'image')
            if not os.path.exists(img_dir):
                # fallback to domain root /image (sometimes structure differs)
                img_dir = os.path.join(cfg['train_data_path'], domain_name, 'image')
            if not os.path.isdir(img_dir):
                print(f'  Image directory not found for domain {domain}: {img_dir}, skipping')
                continue

            img_list = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if len(img_list) == 0:
                print(f'  No images found in {img_dir}, skipping')
                continue

            with torch.no_grad():
                for img_path in img_list:
                    pil = Image.open(img_path)
                    # Keep original for display
                    orig_pil = pil.copy()
                    # Ensure RGB for CLIP preprocess
                    pil_rgb = pil.convert('RGB')
                    # Use BiomedCLIP's preprocess on PIL image (this matches the example pipeline)
                    try:
                        input_tensor = preprocess(pil_rgb).unsqueeze(0).to(device)
                    except Exception:
                        # Fallback: use preprocess_tensor_images if preprocess not callable as expected
                        tmp = transforms.ToTensor()(pil_rgb).unsqueeze(0)
                        input_tensor = preprocess_tensor_images(tmp, preprocess, str(device))

                    im_feats = model.encode_image(input_tensor)
                    im_feats = F.normalize(im_feats.float(), dim=-1)
                    logits = scale * (im_feats @ text_feats.t())  # [1, 2]
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    logits_np = logits.cpu().numpy()[0]

                    # Save original image as tensor for display (C,H,W) in range [0,1]
                    orig_np = np.array(orig_pil)
                    if orig_np.ndim == 2:
                        # grayscale -> H,W -> expand to H,W,1
                        disp_np = orig_np.astype(np.float32) / 255.0
                        disp_tensor = torch.from_numpy(disp_np).unsqueeze(0)
                    else:
                        disp_np = orig_np.astype(np.float32) / 255.0
                        disp_tensor = torch.from_numpy(disp_np.transpose(2, 0, 1))

                    samples.append(disp_tensor)
                    scores.append(probs.tolist())

                    # Debug: save logits and top prediction
                    out_debug = os.path.join(args.output_dir, f'domain_{domain}_debug.txt')
                    top_idx = int(logits_np.argmax())
                    top_label = classes[top_idx]
                    with open(out_debug, 'a', encoding='utf-8') as f:
                        f.write(f'{os.path.basename(img_path)} logits={logits_np.tolist()} top={top_idx}:{top_label}\n')

                    taken += 1
                    if taken >= args.per_domain:
                        break
        else:
            # fallback: use DataLoader pipeline
            loader = build_test_loader(cfg, batch_size=1, num_workers=args.num_workers, domains=[domain])
            with torch.no_grad():
                for batch in loader:
                    images = batch['image']  # [B, C, H, W]
                    # preprocess for CLIP
                    imgs_pre = preprocess_tensor_images(images, preprocess, str(device))
                    im_feats = model.encode_image(imgs_pre)
                    im_feats = F.normalize(im_feats.float(), dim=-1)
                    # similarity and softmax
                    logits = scale * (im_feats @ text_feats.t())  # [B, 2]
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [B,2]
                    logits_np = logits.cpu().numpy()

                    for i in range(images.shape[0]):
                        samples.append(images[i].cpu())
                        scores.append(probs[i].tolist())
                        # Debug: save logits and top prediction
                        out_debug = os.path.join(args.output_dir, f'domain_{domain}_debug.txt')
                        top_idx = int(logits_np[i].argmax())
                        top_label = classes[top_idx]
                        with open(out_debug, 'a', encoding='utf-8') as f:
                            f.write(f'sample_index={len(samples)-1} logits={logits_np[i].tolist()} top={top_idx}:{top_label}\n')
                        taken += 1
                        if taken >= args.per_domain:
                            break
                    if taken >= args.per_domain:
                        break

        if len(samples) == 0:
            print(f'  No samples found for domain {domain}, skipping')
            continue

        out_path = os.path.join(args.output_dir, f'domain_{domain}.png')
        visualize_domain(domain, samples, scores, out_path, classes)
        print(f'  Saved visualization for domain {domain} -> {out_path}')

    print('All domains processed.')


if __name__ == '__main__':
    main()
