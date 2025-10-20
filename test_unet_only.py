import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
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
parser.add_argument("--save_name", type=str, default="debug", help="experiment_name")
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--eval',type=bool, default=True)

parser.add_argument("--test_bs", type=int, default=1)
parser.add_argument('--domain_num', type=int, default=6, help='total number of domains')
parser.add_argument('--lb_domain', type=int, default=1, help='labeled domain used during training (only for logging/display purpose)')
# Note: removed --test_domains parameter, always test on all domains like original code

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
def test(args, model, test_dataloader, trained_domain):
    """
    Test a model on all domains
    
    Args:
        args: arguments
        model: the model to test
        test_dataloader: list of dataloaders for all test domains
        trained_domain: which domain this model was trained on (for logging)
    
    Returns:
        val_dice, val_dc, val_jc, val_hd, val_asd: average metrics across all test domains
    """
    model.eval()
    val_dice = [0.0] * n_part
    val_dc, val_jc, val_hd, val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
    domain_num = len(test_dataloader)
    num = 0
    for i in range(domain_num):
        cur_dataloader = test_dataloader[i]
        domain_val_dice = [0.0] * n_part
        domain_val_dc, domain_val_jc, domain_val_hd, domain_val_asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
        test_domain = i+1  # Current test domain
        for batch_num,sample in enumerate(cur_dataloader):
            assert(test_domain == sample['dc'][0].item())
            mask = sample['label']
            if args.dataset == 'fundus':
                lb_mask = (mask<=128) * 2
                lb_mask[mask==0] = 1
                mask = lb_mask
            elif args.dataset == 'prostate':
                mask = mask.eq(0).long()
            elif args.dataset == 'MNMS':
                mask = mask.long()
            elif args.dataset == 'BUSI':
                mask = mask.eq(255).long()
            data = sample['unet_size_img'].cuda()
            output = model(data)
            pred_label = torch.max(torch.softmax(output,dim=1), dim=1)[1]
            pred_label = torch.from_numpy(zoom(pred_label.cpu(), (1, patch_size / data.shape[-2], patch_size / data.shape[-1]), order=0))
            
            if args.dataset == 'fundus':
                pred_label = to_2d(pred_label)
                mask = to_2d(mask)
                pred_onehot = pred_label.clone()
                mask_onehot = mask.clone()
            elif args.dataset == 'prostate' or args.dataset == 'BUSI':
                pred_onehot = pred_label.clone().unsqueeze(1)
                mask_onehot = mask.clone().unsqueeze(1)
            elif args.dataset == 'MNMS':
                pred_onehot = to_3d(pred_label)
                mask_onehot = to_3d(mask)
            dice = dice_calcu[args.dataset](np.asarray(pred_label.cpu()),mask.cpu())
            avg_dice = sum(dice)/len(dice)
                    
            dc, jc, hd, asd = [0.0] * n_part, [0.0] * n_part, [0.0] * n_part, [0.0] * n_part
            for j in range(len(data)):
                for i, p in enumerate(part):
                    dc[i] += binary.dc(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                    jc[i] += binary.jc(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                    if pred_onehot[j,i].float().sum() < 1e-4:
                        hd[i] += 100
                        asd[i] += 100
                    else:
                        hd[i] += binary.hd95(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
                        asd[i] += binary.asd(np.asarray(pred_onehot[j,i], dtype=bool),
                                            np.asarray(mask_onehot[j,i], dtype=bool))
            for i, p in enumerate(part):
                dc[i] /= len(data)
                jc[i] /= len(data)
                hd[i] /= len(data)
                asd[i] /= len(data)
            for i in range(len(domain_val_dice)):
                domain_val_dice[i] += dice[i]
                domain_val_dc[i] += dc[i]
                domain_val_jc[i] += jc[i]
                domain_val_hd[i] += hd[i]
                domain_val_asd[i] += asd[i]
                
        for i in range(len(domain_val_dice)):
            domain_val_dice[i] /= len(cur_dataloader)
            val_dice[i] += domain_val_dice[i]
            domain_val_dc[i] /= len(cur_dataloader)
            val_dc[i] += domain_val_dc[i]
            domain_val_jc[i] /= len(cur_dataloader)
            val_jc[i] += domain_val_jc[i]
            domain_val_hd[i] /= len(cur_dataloader)
            val_hd[i] += domain_val_hd[i]
            domain_val_asd[i] /= len(cur_dataloader)
            val_asd[i] += domain_val_asd[i]
        
        # Log results for current test domain
        text = '[Trained on domain %d] Testing on domain %d:' % (trained_domain, test_domain)
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dice: %f, ' % (p, domain_val_dice[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_dc: %f, ' % (p, domain_val_dc[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_jc: %f, ' % (p, domain_val_jc[n])
        text += '\n\t'
        for n, p in enumerate(part):
            text += 'val_%s_hd: %f, ' % (p, domain_val_hd[n])
        text += '\t'
        for n, p in enumerate(part):
            text += 'val_%s_asd: %f, ' % (p, domain_val_asd[n])
        logging.info(text)
        
    model.train()
    for i in range(len(val_dice)):
        val_dice[i] /= domain_num
        val_dc[i] /= domain_num
        val_jc[i] /= domain_num
        val_hd[i] /= domain_num
        val_asd[i] /= domain_num
    
    # Log average results across all test domains
    text = '\n' + '='*80
    text += '\n[Trained on domain %d] Average across all %d test domains:' % (trained_domain, domain_num)
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_dice: %f, ' % (p, val_dice[n])
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_dc: %f, ' % (p, val_dc[n])
    text += '\t'
    for n, p in enumerate(part):
        text += 'val_%s_jc: %f, ' % (p, val_jc[n])
    text += '\n\t'
    for n, p in enumerate(part):
        text += 'val_%s_hd: %f, ' % (p, val_hd[n])
    text += '\t'
    for n, p in enumerate(part):
        text += 'val_%s_asd: %f, ' % (p, val_asd[n])
    text += '\n' + '='*80
    logging.info(text)
    return val_dice, val_dc, val_jc, val_hd, val_asd
    
def main(args, snapshot_path):

    def create_model(ema=False):
        # Network definition
        model = UNet(n_channels = num_channels, n_classes = num_classes+1)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model.cuda()
    
    normal_toTensor = transforms.Compose([
        tr.Normalize_tf(dataRange=[0,1]),
        tr.ToTensor(unet_size=patch_size)
    ])

    domain_num = args.domain_num
    
    # Always test on all domains (like original code)
    test_dataset = []
    test_dataloader = []
    for i in range(1, domain_num+1):
        cur_dataset = dataset(base_dir=train_data_path, phase='test', splitid=-1, domain=[i], normal_toTensor=normal_toTensor, img_size=patch_size)
        test_dataset.append(cur_dataset)
    for i in range(0, domain_num):
        cur_dataloader = DataLoader(test_dataset[i], batch_size = args.test_bs, shuffle=False, num_workers=0, pin_memory=True)
        test_dataloader.append(cur_dataloader)

    if args.eval:
        model = create_model()
        model_path = '../model/{}/train_unet_only/{}/unet_avg_dice_best_model.pth'.format(args.dataset, args.save_name)
        
        # Try to automatically extract trained domain from save_name
        # Expected format: *_d{domain}_* (e.g., UNet_only_d6_8000)
        import re
        match = re.search(r'_d(\d+)_', args.save_name)
        if match:
            auto_detected_domain = int(match.group(1))
            if args.lb_domain != auto_detected_domain:
                logging.warning("="*80)
                logging.warning("WARNING: lb_domain mismatch detected!")
                logging.warning(f"  --lb_domain argument: {args.lb_domain}")
                logging.warning(f"  Domain from save_name: {auto_detected_domain}")
                logging.warning(f"  Using auto-detected domain: {auto_detected_domain}")
                logging.warning("="*80)
                trained_domain = auto_detected_domain
            else:
                trained_domain = args.lb_domain
        else:
            logging.warning(f"Cannot auto-detect domain from save_name '{args.save_name}', using --lb_domain={args.lb_domain}")
            trained_domain = args.lb_domain
        
        # Log testing configuration
        logging.info("="*80)
        logging.info("TESTING CONFIGURATION")
        logging.info("="*80)
        logging.info(f"Model trained on domain: {trained_domain}")
        logging.info(f"Model name: {args.save_name}")
        logging.info(f"Model path: {model_path}")
        logging.info(f"Testing on all {args.domain_num} domains")
        logging.info("="*80 + "\n")
        
        model.load_state_dict(torch.load(model_path))
        test(args, model, test_dataloader, trained_domain)
        exit()


if __name__ == "__main__":
    snapshot_path = "../model/" + args.dataset + "/train/" + args.save_name + "/"
    if args.dataset == 'fundus':
        num_channels = 3
        num_classes = 2
        train_data_path='/app/MixDSemi/data/Fundus'
        patch_size = 256
        part = ['cup', 'disc']
        dataset = FundusSegmentation
        if args.domain_num >=4:
            args.domain_num = 4
    elif args.dataset == 'prostate':
        num_channels = 1
        num_classes = 1
        train_data_path="/app/MixDSemi/data/ProstateSlice"
        patch_size = 384
        part = ['base'] 
        dataset = ProstateSegmentation
        if args.domain_num >= 6:
            args.domain_num = 6
    elif args.dataset == 'MNMS':
        num_channels = 1
        num_classes = 3
        train_data_path="/app/MixDSemi/data/mnms"
        patch_size = 288
        part = ['lv', 'myo', 'rv'] 
        dataset = MNMSSegmentation
        if args.domain_num >= 4:
            args.domain_num = 4
    elif args.dataset == 'BUSI':
        num_channels = 1
        num_classes = 1
        train_data_path="../../data/Dataset_BUSI_with_GT"
        patch_size = 256
        part = ['base'] 
        dataset = BUSISegmentation
        if args.domain_num >= 2:
            args.domain_num = 2
        
    if num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False
    n_part = len(part)
    dice_calcu = {'fundus':metrics.dice_coeff_2label, 'prostate':metrics.dice_coeff, 'MNMS':metrics.dice_coeff_3label, 'BUSI':metrics.dice_coeff}

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + f"/log_test_unet.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    cmd = " ".join(["python"] + sys.argv)
    logging.info(cmd)
    logging.info(str(args))

    main(args, snapshot_path)