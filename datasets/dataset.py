import os
import math
import random
import logging
import pathlib
import numpy as np
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import tifffile

from .semantic_sync_transforms import SyncCompose, SyncToTensor

logger = logging.getLogger(__name__)

NUM_TARGET_CLASSES = 7
TARGET_CLASS_NAMES = {
    0: 'Фон',                    # Background (no scene)
    1: 'Лесной массив',          # Forest
    2: 'Поле',                   # Field/Agriculture
    3: 'Водоём',                 # Water
    4: 'Городская территория',   # Urban
    5: 'Горный район',           # Mountain
    6: 'Прочее',                 # Other
}

# Маппинги для датасетов
DEEPGLOBE_COLOR_MAP = {
    (0, 0, 0): 0,        # unknown → Фон
    (0, 255, 255): 4,    # urban_land → Городская территория
    (255, 255, 0): 2,    # agriculture_land → Поле
    (255, 0, 255): 6,    # rangeland → Прочее
    (0, 255, 0): 1,      # forest_land → Лесной массив
    (0, 0, 255): 3,      # water → Водоём
    (255, 255, 255): 5,  # barren_land → Горный район
}

LANDCOVERAI_CLASS_MAP = {
    0: 0,  # Background → Фон
    1: 4,  # Building → Городская территория
    2: 1,  # Woodland → Лесной массив
    3: 3,  # Water → Водоём
    4: 4,  # Road → Городская территория
}

GID_COLOR_MAP = {
    (0, 0, 0): 0,        # unlabeled → Фон
    (255, 0, 0): 4,      # built-up → Городская территория
    (0, 255, 0): 2,      # farmland → Поле
    (0, 255, 255): 1,    # forest → Лесной массив
    (255, 255, 0): 6,    # meadow → Прочее
    (0, 0, 255): 3,      # water → Водоём
}

WHU_CLASS_MAP = {
    0: 0,    # Background → Фон
    10: 2,   # Farmland → Поле
    20: 4,   # City → Городская территория
    30: 4,   # Village → Городская территория
    40: 3,   # Water → Водоём
    50: 1,   # Forest → Лесной массив
    60: 4,   # Road → Городская территория
    70: 6,   # Others → Прочее
}

DODW_CLASS_MAP = {
    0: 3,   # water → Водоём
    1: 1,   # trees → Лесной массив
    2: 6,   # grass → Прочее
    3: 6,   # flooded_vegetation → Прочее
    4: 2,   # crops → Поле
    5: 6,   # shrub_and_scrub → Прочее
    6: 4,   # built → Городская территория
    7: 5,   # bare → Горный район
    8: 5,   # snow_and_ice → Горный район
}


def n_e_augs_shuffle(augs_list, n=-3, skip_first=False):
    """
    Перемешивает первые (или срединные) элементы списка аугментаций.
    """
    augs_list = list(augs_list)
    if len(augs_list) > 3:
        s = 1 if skip_first else 0
        first_n = augs_list[s:n]
        random.shuffle(first_n)
        augs_list[s:n] = first_n
    return augs_list


def pad_image_centered(img: Image.Image, target_size: int) -> Image.Image:
    """Центрированный паддинг изображения нулями."""
    w, h = img.size
    if w >= target_size and h >= target_size:
        return img
    pad_w = max(0, target_size - w)
    pad_h = max(0, target_size - h)
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    new_img = Image.new(img.mode, (max(w, target_size), max(h, target_size)), (0, 0, 0))
    new_img.paste(img, (pad_left, pad_top))
    return new_img


def pad_mask_centered(mask: Image.Image, target_size: int) -> Image.Image:
    """Центрированный паддинг маски фоновым классом (0)."""
    w, h = mask.size
    if w >= target_size and h >= target_size:
        return mask
    pad_w = max(0, target_size - w)
    pad_h = max(0, target_size - h)
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    new_mask = Image.new(mask.mode, (max(w, target_size), max(h, target_size)), 0)
    new_mask.paste(mask, (pad_left, pad_top))
    return new_mask


def compute_patch_starts(w: int, h: int, patch_size: int) -> List[Tuple[int, int]]:
    """Вычисляет начальные координаты (x, y) для нарезки изображения скользящим окном."""
    if w <= patch_size:
        x_starts = [0]
    else:
        x_starts = list(range(0, w, patch_size))
        if x_starts[-1] + patch_size > w:
            x_starts[-1] = w - patch_size

    if h <= patch_size:
        y_starts = [0]
    else:
        y_starts = list(range(0, h, patch_size))
        if y_starts[-1] + patch_size > h:
            y_starts[-1] = h - patch_size

    return [(x, y) for x in x_starts for y in y_starts]


def apply_color_map(mask_np: np.ndarray, color_map: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    """Векторизованное применение цветового маппинга."""
    out = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
    for color, class_idx in color_map.items():
        match = (mask_np[:, :, 0] == color[0]) & \
                (mask_np[:, :, 1] == color[1]) & \
                (mask_np[:, :, 2] == color[2])
        out[match] = class_idx
    return out


def apply_index_map(mask_np: np.ndarray, index_map: Dict[int, int]) -> np.ndarray:
    """Векторизованное применение маппинга индексов."""
    out = np.zeros_like(mask_np, dtype=np.uint8)
    for in_idx, out_idx in index_map.items():
        out[mask_np == in_idx] = out_idx
    return out


class BasePatchDataset(Dataset):
    """Базовый класс для датасетов с кэшированием и нарезкой патчей."""
    def __init__(self, transforms=None, resolution=512):
        self.transforms = transforms or []
        self.resolution = resolution
        self.patches = []
        self._cached_idx = -1
        self._cached_img = None
        self._cached_mask = None

    def __len__(self):
        return len(self.patches)

    def _get_scaled_image_and_mask(self, idx_file: int) -> Tuple[Image.Image, Image.Image]:
        raise NotImplementedError

    def __getitem__(self, idx):
        file_idx, px, py = self.patches[idx]
        img, mask = self._get_scaled_image_and_mask(file_idx)

        img_patch = img.crop((px, py, px + self.resolution, py + self.resolution))
        mask_patch = mask.crop((px, py, px + self.resolution, py + self.resolution))

        img_patch = pad_image_centered(img_patch, self.resolution)
        mask_patch = pad_mask_centered(mask_patch, self.resolution)

        if self.transforms:
            sync_transforms = SyncCompose(n_e_augs_shuffle(self.transforms))
            names = list(range(len(self.transforms)))
            img_tensor, mask_tensor = sync_transforms(img=img_patch, mask=mask_patch, names=names)
            return img_tensor, mask_tensor
        else:
            tt = SyncToTensor()
            return tt(img=img_patch, mask=mask_patch)


class DeepGlobeData(BasePatchDataset):
    def __init__(self, root_dir, split='train', transforms=None, resolution=512, target_m_per_px=10.0, class_mapping=None):
        super().__init__(transforms, resolution)
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.target_m_per_px = target_m_per_px
        self.class_mapping = class_mapping or DEEPGLOBE_COLOR_MAP
        
        self.native_gsd = 0.5
        self.scale_factor = self.native_gsd / self.target_m_per_px

        self.files = []
        split_dir = self.root_dir / split
        if split_dir.exists():
            for img_path in sorted(split_dir.glob("*_sat.jpg")):
                mask_path = img_path.parent / img_path.name.replace("_sat.jpg", "_mask.png")
                if mask_path.exists():
                    self.files.append((img_path, mask_path))

        logger.info("Инициализация DeepGlobeData (%s): найдено %d пар.", split, len(self.files))
        
        if self.files:
            with Image.open(self.files[0][0]) as tmp:
                w, h = tmp.size
            sw, sh = max(1, int(w * self.scale_factor)), max(1, int(h * self.scale_factor))
            starts = compute_patch_starts(sw, sh, self.resolution)
            for i in range(len(self.files)):
                for px, py in starts:
                    self.patches.append((i, px, py))

    def _get_scaled_image_and_mask(self, idx_file):
        if self._cached_idx == idx_file:
            return self._cached_img, self._cached_mask

        img_path, mask_path = self.files[idx_file]
        
        img = Image.open(img_path).convert('RGB')
        sw, sh = max(1, int(img.size[0] * self.scale_factor)), max(1, int(img.size[1] * self.scale_factor))
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)

        mask = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask)
        idx_mask_np = apply_color_map(mask_np, self.class_mapping)
        idx_mask = Image.fromarray(idx_mask_np, mode='L')
        idx_mask = idx_mask.resize((sw, sh), Image.Resampling.NEAREST)

        self._cached_idx = idx_file
        self._cached_img = img
        self._cached_mask = idx_mask
        return img, idx_mask


class LandCoverAIData(BasePatchDataset):
    def __init__(self, root_dir, split='train', transforms=None, resolution=512, target_m_per_px=10.0, class_mapping=None):
        super().__init__(transforms, resolution)
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.target_m_per_px = target_m_per_px
        self.class_mapping = class_mapping or LANDCOVERAI_CLASS_MAP
        
        self.native_gsd = 0.25
        self.scale_factor = self.native_gsd / self.target_m_per_px

        self.files = []
        img_dir = self.root_dir / f"{split}_images"
        mask_dir = self.root_dir / f"{split}_masks"
        if img_dir.exists() and mask_dir.exists():
            for img_path in sorted(img_dir.glob("*.jpg")):
                mask_name = img_path.stem + "_m.png"
                mask_path = mask_dir / mask_name
                if mask_path.exists():
                    self.files.append((img_path, mask_path))

        logger.info("Инициализация LandCoverAIData (%s): найдено %d пар.", split, len(self.files))
        
        if self.files:
            with Image.open(self.files[0][0]) as tmp:
                w, h = tmp.size
            sw, sh = max(1, int(w * self.scale_factor)), max(1, int(h * self.scale_factor))
            starts = compute_patch_starts(sw, sh, self.resolution)
            for i in range(len(self.files)):
                for px, py in starts:
                    self.patches.append((i, px, py))

    def _get_scaled_image_and_mask(self, idx_file):
        if self._cached_idx == idx_file:
            return self._cached_img, self._cached_mask

        img_path, mask_path = self.files[idx_file]
        
        img = Image.open(img_path).convert('RGB')
        sw, sh = max(1, int(img.size[0] * self.scale_factor)), max(1, int(img.size[1] * self.scale_factor))
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)

        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
            
        idx_mask_np = apply_index_map(mask_np, self.class_mapping)
        idx_mask = Image.fromarray(idx_mask_np, mode='L')
        idx_mask = idx_mask.resize((sw, sh), Image.Resampling.NEAREST)

        self._cached_idx = idx_file
        self._cached_img = img
        self._cached_mask = idx_mask
        return img, idx_mask


class GIDData(BasePatchDataset):
    def __init__(self, root_dir, split='train', transforms=None, resolution=512, target_m_per_px=10.0, class_mapping=None):
        super().__init__(transforms, resolution)
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.target_m_per_px = target_m_per_px
        self.class_mapping = class_mapping or GID_COLOR_MAP
        
        self.native_gsd = 4.0
        self.scale_factor = self.native_gsd / self.target_m_per_px

        self.files = []
        if split == 'train':
            img_dir = self.root_dir / "Image_8bit_NirRGB"
            mask_dir = self.root_dir / "Annotation"
        else:
            img_dir = self.root_dir / "test_images"
            mask_dir = self.root_dir / "test_Annotation"

        if img_dir.exists() and mask_dir.exists():
            for img_path in sorted(img_dir.glob("*.tif")):
                mask_name = img_path.name.replace(".tif", "_5label.tif")
                mask_path = mask_dir / mask_name
                if mask_path.exists():
                    self.files.append((img_path, mask_path))

        logger.info("Инициализация GIDData (%s): найдено %d пар.", split, len(self.files))
        
        for i, (img_path, _) in enumerate(self.files):
            with tifffile.TiffFile(img_path) as tif:
                shape = tif.pages[0].shape
                if len(shape) >= 3 and shape[2] in (3, 4):
                    h, w = shape[0], shape[1]
                elif len(shape) >= 3 and shape[0] in (3, 4):
                    h, w = shape[1], shape[2]
                else:
                    h, w = shape[0], shape[1]

            sw, sh = max(1, int(w * self.scale_factor)), max(1, int(h * self.scale_factor))
            starts = compute_patch_starts(sw, sh, self.resolution)
            for px, py in starts:
                self.patches.append((i, px, py))

    def _get_scaled_image_and_mask(self, idx_file):
        if self._cached_idx == idx_file:
            return self._cached_img, self._cached_mask

        img_path, mask_path = self.files[idx_file]
        
        img_arr = tifffile.imread(img_path)
        if img_arr.shape[0] == 4:
            img_arr = np.transpose(img_arr, (1, 2, 0))
            
        rgb_arr = img_arr[:, :, [1, 2, 0]]
        img = Image.fromarray(rgb_arr, mode='RGB')
        sw, sh = max(1, int(img.size[0] * self.scale_factor)), max(1, int(img.size[1] * self.scale_factor))
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)

        mask_arr = tifffile.imread(mask_path)
        if mask_arr.shape[0] == 3:
            mask_arr = np.transpose(mask_arr, (1, 2, 0))
        idx_mask_np = apply_color_map(mask_arr, self.class_mapping)
        idx_mask = Image.fromarray(idx_mask_np, mode='L')
        idx_mask = idx_mask.resize((sw, sh), Image.Resampling.NEAREST)

        self._cached_idx = idx_file
        self._cached_img = img
        self._cached_mask = idx_mask
        return img, idx_mask


class WHUOptSarData(BasePatchDataset):
    def __init__(self, root_dir, split='train', transforms=None, resolution=512, target_m_per_px=10.0, class_mapping=None):
        super().__init__(transforms, resolution)
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.target_m_per_px = target_m_per_px
        self.class_mapping = class_mapping or WHU_CLASS_MAP
        
        self.native_gsd = 5.0
        self.scale_factor = self.native_gsd / self.target_m_per_px

        self.files = []
        if split == 'train':
            img_dir = self.root_dir / "optical"
            mask_dir = self.root_dir / "lbl"
        else:
            img_dir = self.root_dir / "test_optical"
            mask_dir = self.root_dir / "test_lbl"

        if img_dir.exists() and mask_dir.exists():
            for img_path in sorted(img_dir.glob("*.tif")):
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    self.files.append((img_path, mask_path))

        logger.info("Инициализация WHUOptSarData (%s): найдено %d пар.", split, len(self.files))
        
        for i, (img_path, _) in enumerate(self.files):
            with tifffile.TiffFile(img_path) as tif:
                shape = tif.pages[0].shape
                if len(shape) >= 3 and shape[0] in (3, 4):
                    h, w = shape[1], shape[2]
                else:
                    h, w = shape[0], shape[1]

            sw, sh = max(1, int(w * self.scale_factor)), max(1, int(h * self.scale_factor))
            starts = compute_patch_starts(sw, sh, self.resolution)
            for px, py in starts:
                self.patches.append((i, px, py))

    def _get_scaled_image_and_mask(self, idx_file):
        if self._cached_idx == idx_file:
            return self._cached_img, self._cached_mask

        img_path, mask_path = self.files[idx_file]
        
        img_arr = tifffile.imread(img_path)
        if len(img_arr.shape) == 3 and img_arr.shape[0] == 4:
            img_arr = np.transpose(img_arr, (1, 2, 0))
            
        rgb_arr = img_arr[:, :, [2, 1, 0]]
        img = Image.fromarray(rgb_arr, mode='RGB')
        sw, sh = max(1, int(img.size[0] * self.scale_factor)), max(1, int(img.size[1] * self.scale_factor))
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)

        mask_arr = tifffile.imread(mask_path)
        if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
            mask_arr = np.squeeze(mask_arr, axis=-1)
        elif mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
            mask_arr = np.squeeze(mask_arr, axis=0)
            
        idx_mask_np = apply_index_map(mask_arr, self.class_mapping)
        idx_mask = Image.fromarray(idx_mask_np, mode='L')
        idx_mask = idx_mask.resize((sw, sh), Image.Resampling.NEAREST)

        self._cached_idx = idx_file
        self._cached_img = img
        self._cached_mask = idx_mask
        return img, idx_mask


class DODWData(BasePatchDataset):
    def __init__(self, root_dir, split='train', transforms=None, resolution=512, target_m_per_px=10.0, class_mapping=None):
        super().__init__(transforms, resolution)
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.target_m_per_px = target_m_per_px
        self.class_mapping = class_mapping or DODW_CLASS_MAP
        
        self.native_gsd = 10.0
        self.scale_factor = self.native_gsd / self.target_m_per_px

        all_files = []
        img_dir = self.root_dir / "s2_images"
        mask_dir = self.root_dir / "dw_test_zenodo"
        
        if img_dir.exists() and mask_dir.exists():
            for img_path in sorted(img_dir.glob("s2_dw_*_B02-B03-B04-B08.tif")):
                common_part = img_path.name.replace("s2_dw_", "").replace("_B02-B03-B04-B08.tif", "")
                mask_name = f"label_dw_{common_part}.tif"
                mask_path = mask_dir / mask_name
                if mask_path.exists():
                    all_files.append((img_path, mask_path))

        num_train = int(len(all_files) * 0.8)
        if split == 'train':
            self.files = all_files[:num_train]
        else:
            self.files = all_files[num_train:]

        logger.info("Инициализация DODWData (%s): найдено %d пар. Фильтрация пустых...", split, len(self.files))
        
        valid_files = []
        for i, (img_path, mask_path) in enumerate(self.files):
            img_arr = tifffile.imread(img_path)
            if img_arr.max() == 0:
                continue
            valid_files.append((img_path, mask_path))
            
            shape = img_arr.shape
            h, w = (shape[1], shape[2]) if shape[0] == 4 else (shape[0], shape[1])
            
            sw, sh = max(1, int(w * self.scale_factor)), max(1, int(h * self.scale_factor))
            starts = compute_patch_starts(sw, sh, self.resolution)
            for px, py in starts:
                self.patches.append((len(valid_files)-1, px, py))
                
        self.files = valid_files

    def _get_scaled_image_and_mask(self, idx_file):
        if self._cached_idx == idx_file:
            return self._cached_img, self._cached_mask

        img_path, mask_path = self.files[idx_file]
        
        img_arr = tifffile.imread(img_path)
        if len(img_arr.shape) == 3 and img_arr.shape[0] == 4:
            img_arr = np.transpose(img_arr, (1, 2, 0))
        rgb_arr = img_arr[:, :, [2, 1, 0]]
        rgb_arr_uint8 = np.clip(rgb_arr / 3000.0 * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb_arr_uint8, mode='RGB')
        sw, sh = max(1, int(img.size[0] * self.scale_factor)), max(1, int(img.size[1] * self.scale_factor))
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)

        mask_arr = tifffile.imread(mask_path)
        if len(mask_arr.shape) == 3 and mask_arr.shape[0] == 2:
            mask_arr = np.transpose(mask_arr, (1, 2, 0))
        gt_channel = mask_arr[:, :, 1]
        
        idx_mask_np = apply_index_map(gt_channel, self.class_mapping)
        idx_mask = Image.fromarray(idx_mask_np, mode='L')
        idx_mask = idx_mask.resize((sw, sh), Image.Resampling.NEAREST)

        self._cached_idx = idx_file
        self._cached_img = img
        self._cached_mask = idx_mask
        return img, idx_mask


def prepare_datasets(resolution=512, target_m_per_px=10.0,
                     deepglobe=True, landcoverai=True, gid=True, whu=True, dodw=True):
    """
    Функция для подготовки тренировочной и валидационной выборок.
    """
    # ---- Пути к датасетам (заглушки — указать реальные пути) ----
    DEEPGLOBE_ROOT = Path('/path/to/DeepGlobe_Land')                       # DeepGlobe Land Cover
    LANDCOVERAI_ROOT = Path('/path/to/landcoverai')                        # LandCover.ai
    GID_ROOT = Path('/path/to/GID')                                        # GID (Gaofen-2)
    WHU_ROOT = Path('/path/to/WHU-OPT-SAR dataset')                        # WHU-OPT-SAR
    DODW_ROOT = Path('/path/to/Dataset_Open_Dynamic_World_Test_Tiles')     # Open Dynamic World

    from .semantic_sync_transforms import (
        SyncRotate360_plus, SyncRandomHorizontalFlip, SyncRandomVerticalFlip,
        SyncRandomBrightnessContrast
    )
    
    train_transforms = [
        SyncRandomHorizontalFlip(p=0.5),
        SyncRandomVerticalFlip(p=0.5),
        SyncRotate360_plus(p=0.5, interpolation=Image.Resampling.BILINEAR),
        SyncRandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ]
    val_transforms = []

    train_datasets = []
    val_datasets = []

    if deepglobe:
        train_datasets.append(DeepGlobeData(DEEPGLOBE_ROOT, split='train', transforms=train_transforms, resolution=resolution, target_m_per_px=target_m_per_px))
        val_datasets.append(DeepGlobeData(DEEPGLOBE_ROOT, split='valid', transforms=val_transforms, resolution=resolution, target_m_per_px=target_m_per_px))
        
    if landcoverai:
        train_datasets.append(LandCoverAIData(LANDCOVERAI_ROOT, split='train', transforms=train_transforms, resolution=resolution, target_m_per_px=target_m_per_px))
        val_datasets.append(LandCoverAIData(LANDCOVERAI_ROOT, split='test', transforms=val_transforms, resolution=resolution, target_m_per_px=target_m_per_px))

    if gid:
        train_datasets.append(GIDData(GID_ROOT, split='train', transforms=train_transforms, resolution=resolution, target_m_per_px=target_m_per_px))
        val_datasets.append(GIDData(GID_ROOT, split='test', transforms=val_transforms, resolution=resolution, target_m_per_px=target_m_per_px))

    if whu:
        train_datasets.append(WHUOptSarData(WHU_ROOT, split='train', transforms=train_transforms, resolution=resolution, target_m_per_px=target_m_per_px))
        val_datasets.append(WHUOptSarData(WHU_ROOT, split='test', transforms=val_transforms, resolution=resolution, target_m_per_px=target_m_per_px))

    if dodw:
        train_datasets.append(DODWData(DODW_ROOT, split='train', transforms=train_transforms, resolution=resolution, target_m_per_px=target_m_per_px))
        val_datasets.append(DODWData(DODW_ROOT, split='val', transforms=val_transforms, resolution=resolution, target_m_per_px=target_m_per_px))

    train_dataset = ConcatDataset(train_datasets) if train_datasets else None
    val_dataset = ConcatDataset(val_datasets) if val_datasets else None

    return train_dataset, val_dataset


def segmentation_collate_fn(batch):
    """Формирует батч для семантической сегментации.
    
    Returns:
        images: [B, 3, H, W] float tensor
        masks: [B, H, W] long tensor with class indices 0..6
    """
    images = []
    masks = []
    for img, mask in batch:
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3]
        
        images.append(img)
        masks.append(mask)

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks
