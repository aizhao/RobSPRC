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
    # is_eval=True 很重要，确保不用加载优化器参数，且 Dropout 关闭
    model, vis_processors, _ = load_model_and_preprocess(
        name=args.blip_model_name, 
        model_type=args.backbone, 
        is_eval=True, 
        device=device
    )
    model.eval()

    # 3. 准备数据预处理 (必须与 train.py 保持一致)
    input_dim = 224
    if args.transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
    elif args.transform == "targetpad":
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    else:
        preprocess = vis_processors["eval"] # Fallback

    # 4. 加载数据集
    # 注意：我们使用 'classic' 模式，因为这个模式通常只返回 (image, index, name)
    # 这样我们可以遍历所有图片，而不是遍历三元组
    print(f"Loading Dataset ({args.split})...")
    if args.dataset == 'CIRR':
        dataset = CIRRDataset(args.split, 'classic', preprocess, args.data_path)
    elif args.dataset == 'FashionIQ':
        # FashionIQ 需要指定类别，这里我们需要所有类别的图片
        dress_types = ['dress', 'toptee', 'shirt']
        dataset = FashionIQDataset(args.split, dress_types, 'classic', preprocess, args.data_path)
    
    # 使用 DataLoader 加速
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=True,
        collate_fn=None # classic 模式通常不需要特殊的 collate_fn
    )

    # 5. 开始提取特征
    all_feats = []
    name_to_idx = {}
    current_idx = 0
    
    print("Extracting Visual Features (ViT only)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=100):
            # 自动适配 Dataset 的返回顺序
            # 1. 常见情况: (name, image) -> CIRR/FashionIQ classic 模式
            if len(batch) == 2:
                names, images = batch
            # 2. 可能的情况: (image, index, name) -> 某些版本的代码
            elif len(batch) == 3:
                # 这里需要根据你的 dataset 实际返回来判断
                # 通常是 names, index, images 或者 images, index, names
                # 为了保险，我们通过类型判断一下
                if isinstance(batch[0], torch.Tensor):
                    images = batch[0]
                    names = batch[2] # 假设 name 在最后
                elif isinstance(batch[-1], torch.Tensor):
                    images = batch[-1]
                    names = batch[0] # 假设 name 在最前
                else:
                    # 尝试中间是 image 的情况
                    images = batch[1]
                    names = batch[0]
            else:
                raise ValueError(f"Unexpected batch structure with length {len(batch)}")

            # 再次检查 images 是否真的是 Tensor，防止顺序还没对
            if not isinstance(images, torch.Tensor):
                # 最后的防线：如果 names 是 tensor (几乎不可能)，或者 images 是 tuple
                if isinstance(names, torch.Tensor):
                    print("⚠️ Warning: Swapping images and names based on type check.")
                    images, names = names, images
                else:
                    raise TypeError(f"Expected tensor for images, got {type(images)}")

            images = images.to(device)
            
            # 🔥 核心：只运行 Visual Encoder
            with torch.cuda.amp.autocast():
                vit_feats = model.visual_encoder(images)
            
            # 立即转存到 CPU 以节省显存
            vit_feats = vit_feats.cpu()
            
            # 存入列表
            all_feats.append(vit_feats)
            
            # 记录索引映射
            for name in names:
                name_to_idx[str(name)] = current_idx
                current_idx += 1

    # 6. 拼接并保存
    print("Concatenating tensors (this may take a moment)...")
    if len(all_feats) > 0:
        big_tensor = torch.cat(all_feats, dim=0)
        
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
    
    # 模型参数 (需与 train.py 一致)
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