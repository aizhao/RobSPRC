import torch
import os
import json
import argparse
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from lavis.models import load_model_and_preprocess

# 引入项目中的工具函数和数据集类
from data_utils import CIRRDataset, FashionIQDataset, targetpad_transform, squarepad_transform
from utils import device, collate_fn
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def precompute_features(args):
    print(f"🚀 Start Precomputing features for {args.dataset} ({args.split})...")
    
    # 1. 准备保存路径
    save_dir = os.path.join("./features", args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    feat_save_path = os.path.join(save_dir, f"{args.split}_vit_feats.pt")
    idx_save_path = os.path.join(save_dir, f"{args.split}_name2idx.json")
    
    if os.path.exists(feat_save_path) and not args.overwrite:
        print(f"⚠️  Features already exist at {feat_save_path}. Use --overwrite to regenerate.")
        return

    # 2. 加载模型 (BLIP-2)
    print(f"Loading Model: {args.blip_model_name}...")
    model, vis_processors, _ = load_model_and_preprocess(
        name=args.blip_model_name, 
        model_type=args.backbone, 
        is_eval=True, 
        device=device
    )
    model.eval()

    # 3. 准备数据预处理
    input_dim = 224
    if args.transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif args.transform == "targetpad":
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    else:
        preprocess = vis_processors["eval"] # Fallback

    # 4. 加载数据集
    print(f"Loading Dataset ({args.split})...")
    if args.dataset == 'CIRR':
        dataset = CIRRDataset(args.split, 'classic', preprocess, args.data_path)
    elif args.dataset == 'FashionIQ':
        dress_types = ['dress', 'toptee', 'shirt']
        dataset = FashionIQDataset(args.split, dress_types, 'classic', preprocess, args.data_path)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=True,
        collate_fn=None 
    )

    # 5. 开始提取特征
    name_to_idx = {}
    current_idx = 0
    total_images = len(dataset)
    
    # 🔥 修改点：放弃 all_feats 列表，改为预分配内存 (Pre-allocation)
    big_tensor = None 
    
    print("Extracting Visual Features (ViT only)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=100):
            if len(batch) == 2:
                names, images = batch
            elif len(batch) == 3:
                if isinstance(batch[0], torch.Tensor):
                    images = batch[0]
                    names = batch[2] 
                elif isinstance(batch[-1], torch.Tensor):
                    images = batch[-1]
                    names = batch[0] 
                else:
                    images = batch[1]
                    names = batch[0]
            else:
                raise ValueError(f"Unexpected batch structure with length {len(batch)}")

            if not isinstance(images, torch.Tensor):
                if isinstance(names, torch.Tensor):
                    images, names = names, images
                else:
                    raise TypeError(f"Expected tensor for images, got {type(images)}")

            images = images.to(device)
            
            # 🔥 将 torch.cuda.amp 替换为 torch.amp (修复你之前的 Warning)
            with torch.amp.autocast('cuda'):
                vit_feats = model.visual_encoder(images)
            
            # 转存到 CPU 以节省显存
            vit_feats = vit_feats.cpu()
            batch_size = vit_feats.shape[0]
            
            # 🔥 核心修改：如果是第一批数据，获知张量维度后，立马预先分配好完整物理内存
            if big_tensor is None:
                seq_len = vit_feats.shape[1]
                feat_dim = vit_feats.shape[2]
                print(f"\n[Memory Opt] Pre-allocating tensor space: ({total_images}, {seq_len}, {feat_dim}) ...")
                # 使用 torch.empty 开辟空间，避免内存碎片化
                big_tensor = torch.empty((total_images, seq_len, feat_dim), dtype=vit_feats.dtype)
            
            # 将当前 batch 的特征直接“填入”预分配好的大张量中对应的位置
            big_tensor[current_idx : current_idx + batch_size] = vit_feats
            
            # 记录索引映射
            for name in names:
                name_to_idx[str(name)] = current_idx
                current_idx += 1

    # 6. 保存逻辑
    print("Saving tensors (Memory safe mode)...")
    if big_tensor is not None:
        # 万一 dataloader 遍历出来的图片总数和 len(dataset) 不一致，做一个安全裁剪
        if current_idx < total_images:
            big_tensor = big_tensor[:current_idx]

        print(f"Saving Tensor {big_tensor.shape} to {feat_save_path}...")
        torch.save(big_tensor, feat_save_path)
        
        print(f"Saving Index ({len(name_to_idx)} images) to {idx_save_path}...")
        with open(idx_save_path, 'w') as f:
            json.dump(name_to_idx, f)
            
        print("✅ Pre-computation Finished!")
    else:
        print("❌ Error: No features extracted. Check dataset path.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Precompute ViT features for fast training")
    parser.add_argument("--dataset", type=str, required=True, choices=['CIRR', 'FashionIQ'])
    parser.add_argument("--data-path", type=str, default="./cirr_dataset")
    parser.add_argument("--split", type=str, default='train', help="Which split to process (train/val)")
    
    # 模型参数
    parser.add_argument("--blip-model-name", default="blip2_cir_cat", type=str)
    parser.add_argument("--backbone", type=str, default="pretrain")
    
    # 数据处理参数
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--target-ratio", default=1.25, type=float)
    parser.add_argument("--transform", default="targetpad", type=str)
    
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files")

    args = parser.parse_args()
    
    # 简单的路径修正逻辑
    if args.dataset == 'FashionIQ' and args.data_path == "./cirr_dataset":
        args.data_path = "./fashionIQ_dataset"

    precompute_features(args)