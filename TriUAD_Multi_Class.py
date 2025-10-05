import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch,evaluation_batch_vis_ZS,WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger

# Dataset-Related Modules
from dataset import MVTecDataset, RealIADDataset
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

# Model-Related Modules
from models import vit_encoder
from models.uad import TriUAD
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from models.DynamicAggregationBlock import DynamicAggregationAttention_Block

from models.CrossEnhanceModule import MCFM

class MultiSourceDataset:
    """
    多源数据集类，用于同时加载原始图片、分割图片和线稿图片
    要求三个数据集中的图片文件名完全一致
    """
    def __init__(self, image_data, segment_data=None, sketch_data=None, class_idx=0):
        self.image_data = image_data
        self.segment_data = segment_data
        self.sketch_data = sketch_data
        self.class_idx = class_idx
        
        # 设置类别信息
        self.classes = image_data.classes
        self.class_to_idx = image_data.class_to_idx
        
        # 验证数据集长度一致性
        if segment_data and len(segment_data) != len(image_data):
            raise ValueError(f"Segment dataset length ({len(segment_data)}) != Image dataset length ({len(image_data)})")
        if sketch_data and len(sketch_data) != len(image_data):
            raise ValueError(f"Sketch dataset length ({len(sketch_data)}) != Image dataset length ({len(image_data)})")
        
        print(f"MultiSourceDataset created:")
        print(f"  - Images: {len(image_data)}")
        print(f"  - Segments: {len(segment_data) if segment_data else 0}")
        print(f"  - Sketches: {len(sketch_data) if sketch_data else 0}")
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        # 获取原始图片
        image, _ = self.image_data[idx]
        
        # 获取分割图片（如果存在）
        segment = None
        if self.segment_data:
            segment, _ = self.segment_data[idx]
        
        # 获取线稿图片（如果存在）
        sketch = None
        if self.sketch_data:
            sketch, _ = self.sketch_data[idx]
        
        # 返回元组：(image, label, segment, sketch)
        return image, self.class_idx, segment, sketch


warnings.filterwarnings("ignore")
def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    # Data Preparation
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA' or args.dataset == 'PCB':
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_path = os.path.join(args.data_path, item, 'train')
            test_path = os.path.join(args.data_path, item)
            train_segment_path = os.path.join(args.data_path, item, 'train_segment')
            train_sketch_path = os.path.join(args.data_path, item, 'train_sketch')
            # print(train_segment_path)
            # print(train_sketch_path)
            # test_path = os.path.join(args.data_path, item)
            # train_data = ImageFolder(root=train_path, transform=data_transform)
            
            
            # train_image_path = os.path.join(args.data_path, item, 'train', 'good')
            # train_segment_path = os.path.join(args.data_path, item, 'train', 'segment')
            # train_sketch_path = os.path.join(args.data_path, item, 'train', 'sketch')
            train_image_data = ImageFolder(root=train_path, transform=data_transform)
            train_image_data.classes = item
            train_image_data.class_to_idx = {item: i}
            
            # 创建分割图片数据集（如果存在）
            train_segment_data = None
            if train_segment_path:
                train_segment_data = ImageFolder(root=train_segment_path, transform=data_transform)
            
            # 创建线稿图片数据集（如果存在）
            train_sketch_data = None
            if train_sketch_path:
                train_sketch_data = ImageFolder(root=train_sketch_path, transform=data_transform)
            
            # 创建自定义数据集，包含所有三种图片
            train_data = MultiSourceDataset(
                image_data=train_image_data,
                segment_data=train_segment_data,
                sketch_data=train_sketch_data,
                class_idx=i
            )
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif args.dataset == 'Real-IAD' :
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                        gt_transform=gt_transform,
                                        phase='train')
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            test_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                       gt_transform=gt_transform,
                                       phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True)
    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    # Model Preparation
    Bottleneck = []
    TriUAD_Guided_Decoder = []
    TriUAD_Extractor = []
    TriUAD_Segment_Extractor = []
    TriUAD_Sketch_Extractor = []
    
    # PVM = PVMamba(input_dim=embed_dim,output_dim=embed_dim)

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)
    
    # FuseAgg
    Fuse_Agg = Mlp(embed_dim*3, embed_dim * 4, embed_dim, drop=0.)


    Tri = nn.ParameterList(
                    [nn.Parameter(torch.randn(6, embed_dim))
                     for _ in range(1)])

    # TriUAD Extractor
    for i in range(1):
        blk = DynamicAggregationAttention_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        TriUAD_Extractor.append(blk)
    TriUAD_Extractor = nn.ModuleList(TriUAD_Extractor)
    for i in range(1):
        blk = DynamicAggregationAttention_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        TriUAD_Segment_Extractor.append(blk)
    TriUAD_Segment_Extractor = nn.ModuleList(TriUAD_Segment_Extractor)
    for i in range(1):
        blk = DynamicAggregationAttention_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        TriUAD_Sketch_Extractor.append(blk)
    TriUAD_Sketch_Extractor = nn.ModuleList(TriUAD_Sketch_Extractor)

    # TriUAD_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))# nn.LayerNorm->nn.BatchNorm2d
        TriUAD_Guided_Decoder.append(blk)
    TriUAD_Guided_Decoder = nn.ModuleList(TriUAD_Guided_Decoder)
    
    SSI_Fuse = MCFM(in_channels=embed_dim,num_heads=num_heads)
    
    

    model = TriUAD(encoder=encoder, bottleneck=Bottleneck, aggregation=TriUAD_Extractor, segment_aggregation=TriUAD_Segment_Extractor, 
                       sketch_aggregation=TriUAD_Sketch_Extractor,decoder=TriUAD_Guided_Decoder, Fuse_Agg = Fuse_Agg,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=Tri,SSI_Fuse=SSI_Fuse)
    model = model.to(device)

    if args.phase == 'train':
        # Model Initialization
        trainable = nn.ModuleList([Bottleneck, TriUAD_Guided_Decoder, TriUAD_Extractor, Tri])
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=args.total_epochs*len(train_dataloader),
                                           warmup_iters=100)
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, label, segment, sketch in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                segment = segment.to(device)
                sketch = sketch.to(device)
                en, de, g_loss = model(img,segment,sketch)
                # en, de = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.5 * g_loss
                # loss = loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()
            print_fn('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list)))
            # if (epoch + 1) % args.total_epochs == 0:
            if (epoch + 1) % args.total_epochs == 0:
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(args.item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)
                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))
                model.train()
    elif args.phase == 'test':
        # Test
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'model.pth')), strict=True)
        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
        model.eval()
        
        for item, test_data in zip(args.item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=4)
            results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            # results = evaluation_batch_vis_ZS(model, test_dataloader, device, max_ratio=0.01, resize_mask=256, save_root=save_root)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            auroc_px_list.append(auroc_px)
            ap_px_list.append(ap_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)
            print_fn(
                '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

        print_fn(
            'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'PCB') # 'MVTec-AD' or 'VisA' or 'PCB' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'/home/admin1/disk_data5/datasets/PCB') # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='TriUAD-Former-Multi-Class')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)# 392

    # training info
    parser.add_argument('--total_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')

    args = parser.parse_args()
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # category info
    if args.dataset == 'MVTec-AD':
        args.item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    elif args.dataset == 'VisA':
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Real-IAD':

        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
        # args.item_list = ['audiojack']
    elif args.dataset == 'PCB':
        args.item_list = ['pcb1', 'pcb2', 'pcb3']
    main(args)
