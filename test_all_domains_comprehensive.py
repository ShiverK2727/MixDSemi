#!/usr/bin/env python
"""
Test all trained domain models comprehensively
This script tests each domain-trained model on all domains and computes statistics
"""
import argparse
import logging
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from networks.unet_model import UNet
from dataloaders.dataloader import FundusSegmentation, ProstateSegmentation, MNMSSegmentation, BUSISegmentation
import dataloaders.custom_transforms as tr
from utils import losses, metrics, ramps, util
from medpy.metric import binary
from scipy.ndimage import zoom

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='prostate', choices=['fundus', 'prostate', 'MNMS', 'BUSI'])
parser.add_argument("--save_prefix", type=str, default="UNet_only", help="prefix of model names")
parser.add_argument("--save_suffix", type=str, default="8000", help="suffix of model names (e.g., '8000' for 8000 iterations)")
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--output_csv', type=str, default='comprehensive_results.csv', help='output CSV file name')

args = parser.parse_args()

def to_2d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    temp_prob = input_tensor == torch.ones_like(input_tensor)
    tensor_list.append(temp_prob)
    temp_prob2 = input_tensor > torch.zeros_like(input_tensor)
    tensor_list.append(temp_prob2)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def to_3d(input_tensor):
    input_tensor = input_tensor.unsqueeze(1)
    tensor_list = []
    for i in range(1, 4):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

@torch.no_grad()
def test_single_model(model, test_dataloader, trained_domain, domain_num, part, dice_calcu, dataset_name, patch_size):
    """
    Test a single model on all domains
    
    Returns:
        results: dict with structure {test_domain: {metric: value}}
    """
    model.eval()
    results = {}
    
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        test_domain = i+1
        
        domain_val_dice = [0.0] * len(part)
        domain_val_dc = [0.0] * len(part)
        domain_val_jc = [0.0] * len(part)
        domain_val_hd = [0.0] * len(part)
        domain_val_asd = [0.0] * len(part)
        
        for batch_num, sample in enumerate(cur_dataloader):
            assert(test_domain == sample['dc'][0].item())
            mask = sample['label']
            
            if dataset_name == 'fundus':
                lb_mask = (mask<=128) * 2
                lb_mask[mask==0] = 1
                mask = lb_mask
            elif dataset_name == 'prostate':
                mask = mask.eq(0).long()
            elif dataset_name == 'MNMS':
                mask = mask.long()
            elif dataset_name == 'BUSI':
                mask = mask.eq(255).long()
            
            data = sample['unet_size_img'].cuda()
            output = model(data)
            pred_label = torch.max(torch.softmax(output,dim=1), dim=1)[1]
            pred_label = torch.from_numpy(zoom(pred_label.cpu(), (1, patch_size / data.shape[-2], patch_size / data.shape[-1]), order=0))
            
            if dataset_name == 'fundus':
                pred_label = to_2d(pred_label)
                mask = to_2d(mask)
                pred_onehot = pred_label.clone()
                mask_onehot = mask.clone()
            elif dataset_name == 'prostate' or dataset_name == 'BUSI':
                pred_onehot = pred_label.clone().unsqueeze(1)
                mask_onehot = mask.clone().unsqueeze(1)
            elif dataset_name == 'MNMS':
                pred_onehot = to_3d(pred_label)
                mask_onehot = to_3d(mask)
            
            dice = dice_calcu[dataset_name](np.asarray(pred_label.cpu()),mask.cpu())
            
            dc, jc, hd, asd = [0.0] * len(part), [0.0] * len(part), [0.0] * len(part), [0.0] * len(part)
            for j in range(len(data)):
                for idx, p in enumerate(part):
                    dc[idx] += binary.dc(np.asarray(pred_onehot[j,idx], dtype=bool),
                                         np.asarray(mask_onehot[j,idx], dtype=bool))
                    jc[idx] += binary.jc(np.asarray(pred_onehot[j,idx], dtype=bool),
                                         np.asarray(mask_onehot[j,idx], dtype=bool))
                    if pred_onehot[j,idx].float().sum() < 1e-4:
                        hd[idx] += 100
                        asd[idx] += 100
                    else:
                        hd[idx] += binary.hd95(np.asarray(pred_onehot[j,idx], dtype=bool),
                                               np.asarray(mask_onehot[j,idx], dtype=bool))
                        asd[idx] += binary.asd(np.asarray(pred_onehot[j,idx], dtype=bool),
                                               np.asarray(mask_onehot[j,idx], dtype=bool))
            
            for idx in range(len(part)):
                dc[idx] /= len(data)
                jc[idx] /= len(data)
                hd[idx] /= len(data)
                asd[idx] /= len(data)
            
            for idx in range(len(part)):
                domain_val_dice[idx] += dice[idx]
                domain_val_dc[idx] += dc[idx]
                domain_val_jc[idx] += jc[idx]
                domain_val_hd[idx] += hd[idx]
                domain_val_asd[idx] += asd[idx]
        
        # Average over all samples in current test domain
        for idx in range(len(part)):
            domain_val_dice[idx] /= len(cur_dataloader)
            domain_val_dc[idx] /= len(cur_dataloader)
            domain_val_jc[idx] /= len(cur_dataloader)
            domain_val_hd[idx] /= len(cur_dataloader)
            domain_val_asd[idx] /= len(cur_dataloader)
        
        # Store results
        results[test_domain] = {
            'dice': domain_val_dice.copy(),
            'dc': domain_val_dc.copy(),
            'jc': domain_val_jc.copy(),
            'hd': domain_val_hd.copy(),
            'asd': domain_val_asd.copy()
        }
    
    model.train()
    return results

def main(args):
    # Dataset configuration
    if args.dataset == 'fundus':
        num_channels = 3
        num_classes = 2
        train_data_path='/app/MixDSemi/data/Fundus'
        patch_size = 256
        part = ['cup', 'disc']
        dataset = FundusSegmentation
        domain_num = 4
    elif args.dataset == 'prostate':
        num_channels = 1
        num_classes = 1
        train_data_path="/app/MixDSemi/data/ProstateSlice"
        patch_size = 384
        part = ['base']
        dataset = ProstateSegmentation
        domain_num = 6
    elif args.dataset == 'MNMS':
        num_channels = 1
        num_classes = 3
        train_data_path="/app/MixDSemi/data/mnms"
        patch_size = 288
        part = ['lv', 'myo', 'rv']
        dataset = MNMSSegmentation
        domain_num = 4
    elif args.dataset == 'BUSI':
        num_channels = 1
        num_classes = 1
        train_data_path="../../data/Dataset_BUSI_with_GT"
        patch_size = 256
        part = ['base']
        dataset = BUSISegmentation
        domain_num = 2
    
    n_part = len(part)
    dice_calcu = {'fundus':metrics.dice_coeff_2label, 'prostate':metrics.dice_coeff, 
                  'MNMS':metrics.dice_coeff_3label, 'BUSI':metrics.dice_coeff}
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Setup logging
    log_dir = f'../model/{args.dataset}/comprehensive_test/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'comprehensive_test_log.txt')
    
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info("="*100)
    logging.info("COMPREHENSIVE TESTING: ALL DOMAIN-TRAINED MODELS")
    logging.info("="*100)
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Total domains: {domain_num}")
    logging.info(f"Model prefix: {args.save_prefix}")
    logging.info(f"Model suffix: {args.save_suffix}")
    logging.info(f"Output CSV: {args.output_csv}")
    logging.info("="*100 + "\n")
    
    # Prepare data loaders
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0,1]),
        tr.ToTensor(unet_size=patch_size)
    ])
    
    test_dataset = []
    test_dataloader = []
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, 
                            domain=[i], normal_toTensor=normal_toTensor, img_size=patch_size)
        test_dataset.append(cur_dataset)
    for i in range(0, domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size=args.test_bs, 
                                   shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)
    
    # Store all results: all_results[trained_domain][test_domain][metric]
    all_results = {}
    available_models = []
    
    # Test each domain-trained model
    for trained_domain in range(1, domain_num+1):
        model_name = f"{args.save_prefix}_d{trained_domain}_{args.save_suffix}"
        model_path = f'../model/{args.dataset}/train_unet_only/{model_name}/unet_avg_dice_best_model.pth'
        
        if not os.path.exists(model_path):
            logging.warning(f"Model for domain {trained_domain} not found: {model_path}")
            logging.warning(f"Skipping domain {trained_domain}\n")
            continue
        
        logging.info("="*100)
        logging.info(f"TESTING MODEL TRAINED ON DOMAIN {trained_domain}")
        logging.info("="*100)
        logging.info(f"Model name: {model_name}")
        logging.info(f"Model path: {model_path}")
        
        # Load model
        model = UNet(n_channels=num_channels, n_classes=num_classes+1).cuda()
        model.load_state_dict(torch.load(model_path))
        
        # Test on all domains
        results = test_single_model(model, test_dataloader, trained_domain, domain_num, 
                                   part, dice_calcu, args.dataset, patch_size)
        all_results[trained_domain] = results
        available_models.append(trained_domain)
        
        # Log results for this trained model
        logging.info(f"\nResults for model trained on domain {trained_domain}:")
        logging.info("-"*100)
        
        avg_metrics = {'dice': [0.0]*n_part, 'dc': [0.0]*n_part, 'jc': [0.0]*n_part, 
                      'hd': [0.0]*n_part, 'asd': [0.0]*n_part}
        
        for test_domain in range(1, domain_num+1):
            is_source = " (SOURCE DOMAIN)" if test_domain == trained_domain else " (target domain)"
            logging.info(f"\n  Test domain {test_domain}{is_source}:")
            
            for p_idx, p_name in enumerate(part):
                logging.info(f"    {p_name}: dice={results[test_domain]['dice'][p_idx]:.4f}, "
                           f"dc={results[test_domain]['dc'][p_idx]:.4f}, "
                           f"jc={results[test_domain]['jc'][p_idx]:.4f}, "
                           f"hd={results[test_domain]['hd'][p_idx]:.4f}, "
                           f"asd={results[test_domain]['asd'][p_idx]:.4f}")
            
            for metric in avg_metrics:
                for p_idx in range(n_part):
                    avg_metrics[metric][p_idx] += results[test_domain][metric][p_idx]
        
        # Compute average across all test domains
        for metric in avg_metrics:
            for p_idx in range(n_part):
                avg_metrics[metric][p_idx] /= domain_num
        
        logging.info(f"\n  AVERAGE ACROSS ALL {domain_num} TEST DOMAINS:")
        for p_idx, p_name in enumerate(part):
            logging.info(f"    {p_name}: dice={avg_metrics['dice'][p_idx]:.4f}, "
                       f"dc={avg_metrics['dc'][p_idx]:.4f}, "
                       f"jc={avg_metrics['jc'][p_idx]:.4f}, "
                       f"hd={avg_metrics['hd'][p_idx]:.4f}, "
                       f"asd={avg_metrics['asd'][p_idx]:.4f}")
        logging.info("="*100 + "\n")
    
    # Compute overall statistics
    if len(available_models) > 0:
        logging.info("\n" + "="*100)
        logging.info("OVERALL STATISTICS ACROSS ALL TRAINED MODELS")
        logging.info("="*100)
        logging.info(f"Available models: {len(available_models)} out of {domain_num} domains")
        logging.info(f"Trained domains: {available_models}\n")
        
        # Compute mean and std across all trained models
        for metric_name in ['dice', 'dc', 'jc', 'hd', 'asd']:
            logging.info(f"\n{metric_name.upper()} Statistics:")
            logging.info("-"*100)
            
            for p_idx, p_name in enumerate(part):
                # Collect all values for this metric and part across all models and test domains
                all_values = []
                for trained_domain in available_models:
                    for test_domain in range(1, domain_num+1):
                        all_values.append(all_results[trained_domain][test_domain][metric_name][p_idx])
                
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                
                logging.info(f"  {p_name}: mean={mean_val:.4f} ± {std_val:.4f}")
                
                # Also compute source vs target statistics
                source_values = []  # trained_domain == test_domain
                target_values = []  # trained_domain != test_domain
                
                for trained_domain in available_models:
                    for test_domain in range(1, domain_num+1):
                        val = all_results[trained_domain][test_domain][metric_name][p_idx]
                        if trained_domain == test_domain:
                            source_values.append(val)
                        else:
                            target_values.append(val)
                
                if source_values:
                    source_mean = np.mean(source_values)
                    source_std = np.std(source_values)
                    logging.info(f"    └─ Source domain (train=test): {source_mean:.4f} ± {source_std:.4f}")
                
                if target_values:
                    target_mean = np.mean(target_values)
                    target_std = np.std(target_values)
                    logging.info(f"    └─ Target domains (train≠test): {target_mean:.4f} ± {target_std:.4f}")
        
        # Save to CSV
        csv_path = os.path.join(log_dir, args.output_csv)
        with open(csv_path, 'w') as f:
            # Header
            header = "trained_domain,test_domain"
            for p_name in part:
                header += f",{p_name}_dice,{p_name}_dc,{p_name}_jc,{p_name}_hd,{p_name}_asd"
            f.write(header + "\n")
            
            # Data rows
            for trained_domain in available_models:
                for test_domain in range(1, domain_num+1):
                    row = f"{trained_domain},{test_domain}"
                    for p_idx in range(n_part):
                        row += f",{all_results[trained_domain][test_domain]['dice'][p_idx]:.6f}"
                        row += f",{all_results[trained_domain][test_domain]['dc'][p_idx]:.6f}"
                        row += f",{all_results[trained_domain][test_domain]['jc'][p_idx]:.6f}"
                        row += f",{all_results[trained_domain][test_domain]['hd'][p_idx]:.6f}"
                        row += f",{all_results[trained_domain][test_domain]['asd'][p_idx]:.6f}"
                    f.write(row + "\n")
        
        logging.info(f"\n\nResults saved to: {csv_path}")
        logging.info("="*100)
    else:
        logging.error("No trained models found! Please train models first.")

if __name__ == "__main__":
    main(args)
