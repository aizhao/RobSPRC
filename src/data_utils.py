import json
from pathlib import Path
from typing import Union, List, Dict, Literal
import os

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch 

# 获取项目根目录 (假设 data_utils.py 在 src/ 下，根目录是上一级)
# 如果你的文件结构不同，请根据实际情况调整 base_path
base_path = Path(__file__).absolute().parents[1].absolute()

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """
    def __init__(self, target_ratio: float, size: int):
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable, use_cache: bool = False, data_path: str = None):
        """
        :param split: dataset split
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode
        :param preprocess: function which preprocesses the image
        :param use_cache: whether to use precomputed features (for training speedup)
        :param data_path: explicit path to dataset root (optional)
        """
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        self.preprocess = preprocess
        self.use_cache = use_cache
        
        # Determine dataset root
        if data_path:
            self.dataset_root = Path(data_path)
        else:
            self.dataset_root = base_path / 'fashionIQ_dataset'

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        # Load cached features if enabled
        if self.use_cache and self.split == 'train' and self.mode == 'relative':
            feat_path = f"./features/FashionIQ/{split}_vit_feats.pt"
            idx_path = f"./features/FashionIQ/{split}_name2idx.json"
            
            if os.path.exists(feat_path) and os.path.exists(idx_path):
                print(f"🚀 Loading cached features from {feat_path}...")
                self.cached_feats = torch.load(feat_path) # Load to memory (CPU)
                with open(idx_path, 'r') as f:
                    self.name2idx = json.load(f)
                print(f"✅ Cache loaded! Vision Encoder will be bypassed for {len(self.name2idx)} images.")
            else:
                print(f"⚠️ Cache files not found at {feat_path}. Fallback to standard image loading.")
                self.use_cache = False

        # get triplets
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(self.dataset_root / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(self.dataset_root / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']
                target_name = self.triplets[index]['target']

                if self.split == 'train':
                    # [极速模式] 查表读取特征
                    if self.use_cache:
                        if reference_name in self.name2idx:
                            ref_idx = self.name2idx[reference_name]
                            reference_image = self.cached_feats[ref_idx]
                        else:
                            # Fallback if missing
                            reference_image_path = self.dataset_root / 'images' / f"{reference_name}.png"
                            reference_image = self.preprocess(PIL.Image.open(reference_image_path))

                        if target_name in self.name2idx:
                            tgt_idx = self.name2idx[target_name]
                            target_image = self.cached_feats[tgt_idx]
                        else:
                            target_image_path = self.dataset_root / 'images' / f"{target_name}.png"
                            target_image = self.preprocess(PIL.Image.open(target_image_path))
                            
                        return reference_image, target_image, image_captions

                    # [普通模式] 读图
                    reference_image_path = self.dataset_root / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    
                    target_image_path = self.dataset_root / 'images' / f"{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    
                    return reference_image, target_image, image_captions

                elif self.split == 'val':
                    # 🔥【修正】验证集必须加载 Reference Image！
                    reference_image_path = self.dataset_root / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    
                    # 返回：(图片, ref名字, tgt名字, caption)
                    return reference_image, reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = self.dataset_root / 'images' / f"{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = self.dataset_root / 'images' / f"{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")
            return None

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
    """

    def __init__(self, split: str, mode: str, preprocess: callable, use_cache: bool = False, data_path: str = None):
        """
        :param split: dataset split
        :param mode: dataset mode
        :param preprocess: function which preprocesses the image
        :param use_cache: whether to use precomputed features
        :param data_path: explicit path to dataset root
        """
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.use_cache = use_cache

        if data_path:
            self.dataset_root = Path(data_path)
        else:
            self.dataset_root = base_path / 'cirr_dataset'

        if split not in ['test1', 'train', 'val','test2']:
            raise ValueError("split should be in ['test1', 'train', 'val','test2']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # Load cached features if enabled
        if self.use_cache and self.split == 'train' and self.mode == 'relative':
            feat_path = f"./features/CIRR/{split}_vit_feats.pt"
            idx_path = f"./features/CIRR/{split}_name2idx.json"
            
            if os.path.exists(feat_path) and os.path.exists(idx_path):
                print(f"🚀 Loading cached features from {feat_path}...")
                self.cached_feats = torch.load(feat_path) # Load to memory (CPU)
                with open(idx_path, 'r') as f:
                    self.name2idx = json.load(f)
                print(f"✅ Cache loaded! Vision Encoder will be bypassed for {len(self.name2idx)} images.")
            else:
                print(f"⚠️ Cache files not found at {feat_path}. Fallback to standard image loading.")
                self.use_cache = False

        # get triplets made by (reference_image, target_image, relative caption)
        with open(self.dataset_root / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(self.dataset_root / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    # [极速模式] 查表读取特征
                    if self.use_cache:
                        # 获取 Reference Feature
                        if reference_name in self.name2idx:
                            ref_idx = self.name2idx[reference_name]
                            reference_image = self.cached_feats[ref_idx]
                        else:
                            # Fallback
                            reference_image_path = self.dataset_root / self.name_to_relpath[reference_name]
                            reference_image = self.preprocess(PIL.Image.open(reference_image_path))

                        # 获取 Target Feature
                        target_hard_name = self.triplets[index]['target_hard']
                        if target_hard_name in self.name2idx:
                            tgt_idx = self.name2idx[target_hard_name]
                            target_image = self.cached_feats[tgt_idx]
                        else:
                            target_image_path = self.dataset_root / self.name_to_relpath[target_hard_name]
                            target_image = self.preprocess(PIL.Image.open(target_image_path))

                        return reference_image, target_image, rel_caption

                    # [普通模式] 读图
                    reference_image_path = self.dataset_root / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = self.dataset_root / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    
                    return reference_image, target_image, rel_caption

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    
                    # 🔥【修改这里】必须加载 Reference Image！
                    # 原来的代码可能只返回了名字，导致报错
                    
                    reference_image_path = self.dataset_root / self.name_to_relpath[reference_name]
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    
                    # 返回 5 个元素：(图片, ref名, tgt名, caption, group)
                    return reference_image, reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members
                elif self.split == 'test2':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = self.dataset_root / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                
                # 返回 name 和 image
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")
            return None

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRCODataset(Dataset):
    """
    CIRCO dataset
    """
    # ... CIRCODataset 的代码保持不变，因为它不参与训练，不涉及加速 ...
    # 为了完整性，你可以保留原有的 CIRCODataset 代码
    # 这里为了篇幅省略，如有需要请复制原来的 CIRCO 代码
    def __init__(self, data_path: Union[str, Path], split: Literal['val', 'test'],
                 mode: Literal['relative', 'classic'], preprocess: callable):
        data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.data_path = data_path

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")

        with open(data_path / 'COCO2017_unlabeled' / "annotations" / "image_info_unlabeled2017.json", "r") as f:
            imgs_info = json.load(f)

        self.img_paths = [data_path / 'COCO2017_unlabeled' / "unlabeled2017" / img_info["file_name"] for img_info in
                          imgs_info["images"]]
        self.img_ids = [img_info["id"] for img_info in imgs_info["images"]]
        self.img_ids_indexes_map = {str(img_id): i for i, img_id in enumerate(self.img_ids)}

        with open(data_path / 'annotations' / f'{split}.json', "r") as f:
            self.annotations: List[dict] = json.load(f)

        self.max_num_gts = 23
        print(f"CIRCODataset {split} dataset in {mode} mode initialized")

    def __getitem__(self, index) -> dict:
        if self.mode == 'relative':
            query_id = str(self.annotations[index]['id'])
            relative_caption = self.annotations[index]['relative_caption']
            shared_concept = self.annotations[index]['shared_concept']
            reference_img_id = str(self.annotations[index]['reference_img_id'])
            reference_img_path = self.img_paths[self.img_ids_indexes_map[reference_img_id]]
            reference_img = self.preprocess(PIL.Image.open(reference_img_path))

            if self.split == 'val':
                target_img_id = str(self.annotations[index]['target_img_id'])
                gt_img_ids = [str(x) for x in self.annotations[index]['gt_img_ids']]
                target_img_path = self.img_paths[self.img_ids_indexes_map[target_img_id]]
                target_img = self.preprocess(PIL.Image.open(target_img_path))
                gt_img_ids += [''] * (self.max_num_gts - len(gt_img_ids))

                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'target_img': target_img,
                    'target_img_id': target_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'gt_img_ids': gt_img_ids,
                    'query_id': query_id,
                }
            elif self.split == 'test':
                return {
                    'reference_img': reference_img,
                    'reference_imd_id': reference_img_id,
                    'relative_caption': relative_caption,
                    'shared_concept': shared_concept,
                    'query_id': query_id,
                }
        elif self.mode == 'classic':
            img_id = str(self.img_ids[index])
            img_path = self.img_paths[index]
            img = self.preprocess(PIL.Image.open(img_path))
            return {'img': img, 'img_id': img_id}

    def __len__(self):
        if self.mode == 'relative':
            return len(self.annotations)
        elif self.mode == 'classic':
            return len(self.img_ids)