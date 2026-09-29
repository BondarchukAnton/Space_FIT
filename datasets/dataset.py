import os
import math
import random
import logging
import pathlib
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import tifffile

from .semantic_sync_transforms import (
    SyncCompose,
    SyncRandomHorizontalFlip,
    SyncRotate360_plus,
    SyncToTensor,
    SyncRandomVerticalFlip,
    TrickyResize_UpDwn,
    SyncResize,
    RandomNoiseSP,
    AffineAugmentation,
    SyncRandomBrightnessContrastTarget,
    RandomElasticTransform,
    RandomGridDistortion,
)

logger = logging.getLogger(__name__)

NUM_TARGET_CLASSES = 7
TARGET_CLASS_NAMES = {
    0: 'Фон',                    # Background
    1: 'Лесной массив',          # Forest
    2: 'Поле',                   # Field / Agriculture
    3: 'Водоём',                 # Water
    4: 'Городская территория',   # Urban
    5: 'Горный район',           # Mountain
    6: 'Прочее',                 # Other
}

# Маппинги классов для каждого датасета в единое пространство (0..6)

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
    60: 6,   # Road → Прочее (согласно проверенной эталонной спецификации)
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
    Перемешивает ранние элементы списка аугментаций,
    сохраняя фиксированный порядок последних |n| элементов (SyncResize, SyncToTensor, RandomNoiseSP).
    """
    augs_list = list(augs_list)
    if len(augs_list) > abs(n):
        s = 1 if skip_first else 0
        first_part = augs_list[s:n]
        random.shuffle(first_part)
        augs_list[s:n] = first_part
    return augs_list


def pad_image_centered(img: Image.Image, target_size: int) -> Image.Image:
    """Центрированный паддинг изображения нулями до target_size."""
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
    """Центрированный паддинг маски фоновым классом (0) до target_size."""
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
    """
    Вычисляет начальные координаты (x, y) для нарезки скользящим окном
    с равномерным перекрытием по формуле step = (size - res) // ceil((size - res) / res).
    """
    if w <= patch_size:
        x_starts = [0]
    else:
        num_steps_x = math.ceil((w - patch_size) / patch_size)
        step_x = (w - patch_size) // num_steps_x if num_steps_x > 0 else patch_size
        x_starts = list(range(0, w - patch_size + 1, step_x))
        if x_starts[-1] != w - patch_size:
            x_starts.append(w - patch_size)

    if h <= patch_size:
        y_starts = [0]
    else:
        num_steps_y = math.ceil((h - patch_size) / patch_size)
        step_y = (h - patch_size) // num_steps_y if num_steps_y > 0 else patch_size
        y_starts = list(range(0, h - patch_size + 1, step_y))
        if y_starts[-1] != h - patch_size:
            y_starts.append(h - patch_size)

    return [(x, y) for x in x_starts for y in y_starts]


def apply_color_map(mask_np: np.ndarray, color_map: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    """Векторизованное преобразование RGB-маски в одноканальные индексы классов (0..6)."""
    out = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
    for color, class_idx in color_map.items():
        match = (mask_np[:, :, 0] == color[0]) & \
                (mask_np[:, :, 1] == color[1]) & \
                (mask_np[:, :, 2] == color[2])
        out[match] = class_idx
    return out


def apply_index_map(mask_np: np.ndarray, index_map: Dict[int, int]) -> np.ndarray:
    """Векторизованный перемаппинг исходных индексов в единые индексы классов (0..6)."""
    out = np.zeros_like(mask_np, dtype=np.uint8)
    for in_idx, out_idx in index_map.items():
        out[mask_np == in_idx] = out_idx
    return out


class LRUImageCache:
    """LRU-кэш смасштабированных полноразмерных изображений и масок."""
    def __init__(self, maxsize: int = 8):
        self.maxsize = maxsize
        self.cache: OrderedDict[int, Tuple[Image.Image, Image.Image]] = OrderedDict()

    def get(self, key: int) -> Optional[Tuple[Image.Image, Image.Image]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: int, value: Tuple[Image.Image, Image.Image]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)


class BasePatchDataset(Dataset):
    """Базовый класс для датасетов патчей с LRU-кэшированием исходных снимков."""
    def __init__(self, transforms=None, resolution=512):
        self.transforms = transforms or []
        self.resolution = resolution
        self.patches: List[Tuple[int, int, int]] = []
        self.cache = LRUImageCache(maxsize=8)

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
            shuffled_transforms = n_e_augs_shuffle(self.transforms)
            sync_transforms = SyncCompose(shuffled_transforms)
            names = list(range(len(self.patches)))
            res = sync_transforms(
                img=img_patch,
                mask=mask_patch,
                img_path=file_idx,
                mask_path=file_idx,
                names=names
            )
            if isinstance(res, (tuple, list)):
                return res[0], res[1]
            return res
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
        
        for i, (img_path, _) in enumerate(self.files):
            with Image.open(img_path) as tmp:
                w, h = tmp.size
            sw, sh = max(1, int(w * self.scale_factor)), max(1, int(h * self.scale_factor))
            starts = compute_patch_starts(sw, sh, self.resolution)
            for px, py in starts:
                self.patches.append((i, px, py))

    def _get_scaled_image_and_mask(self, idx_file: int) -> Tuple[Image.Image, Image.Image]:
        cached = self.cache.get(idx_file)
        if cached is not None:
            return cached

        img_path, mask_path = self.files[idx_file]
        
        img = Image.open(img_path).convert('RGB')
        sw, sh = max(1, int(img.size[0] * self.scale_factor)), max(1, int(img.size[1] * self.scale_factor))
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)

        mask = Image.open(mask_path).convert('RGB')
        mask_np = np.array(mask)
        idx_mask_np = apply_color_map(mask_np, self.class_mapping)
        idx_mask = Image.fromarray(idx_mask_np, mode='L')
        idx_mask = idx_mask.resize((sw, sh), Image.Resampling.NEAREST)

        res = (img, idx_mask)
        self.cache.put(idx_file, res)
        return res


class LandCoverAIData(BasePatchDataset):
    """
    Dataset для LandCover.ai. Загружает полноразмерные исходные ортофотопланы из images/ и маски из masks/
    (формата GeoTIFF/TIF), масштабируя их с исходного разрешения (0.25 или 0.50 м/пикс) до целевого target_m_per_px.
    """
    def __init__(self, root_dir, split='train', transforms=None, resolution=512, target_m_per_px=10.0, class_mapping=None):
        super().__init__(transforms, resolution)
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.target_m_per_px = target_m_per_px
        self.class_mapping = class_mapping or LANDCOVERAI_CLASS_MAP

        self.files = []
        
        # 1. Поиск оригинальных полноразмерных снимков (images/ и masks/)
        img_dir = self.root_dir / "images"
        mask_dir = self.root_dir / "masks"
        
        if not img_dir.exists():
            img_dir = self.root_dir
        if not mask_dir.exists():
            mask_dir = self.root_dir

        all_pairs = []
        if img_dir.exists() and mask_dir.exists():
            for ext in ("*.tif", "*.tiff", "*.png", "*.jpg"):
                for img_path in sorted(img_dir.glob(ext)):
                    mask_path = None
                    for m_ext in (".tif", ".tiff", ".png"):
                        cand = mask_dir / f"{img_path.stem}{m_ext}"
                        if cand.exists():
                            mask_path = cand
                            break
                    if mask_path is not None:
                        all_pairs.append((img_path, mask_path))

        # 2. Разделение по сплитам
        if all_pairs:
            txt_file = self.root_dir / f"{split}.txt"
            if not txt_file.exists() and split in ('val', 'valid', 'test'):
                for alt_name in ('val.txt', 'test.txt'):
                    cand = self.root_dir / alt_name
                    if cand.exists():
                        txt_file = cand
                        break

            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    tile_names = [line.strip() for line in f if line.strip()]
                sheet_stems = set()
                for name in tile_names:
                    parts = name.rsplit('_', 1)
                    sheet_stems.add(parts[0])

                self.files = [(i, m) for i, m in all_pairs if i.stem in sheet_stems]
                if not self.files:
                    # Если имена файлов не совпали с перфиксом, используем процентный сплит
                    num_train = int(len(all_pairs) * 0.8)
                    self.files = all_pairs[:num_train] if split == 'train' else all_pairs[num_train:]
            else:
                num_train = int(len(all_pairs) * 0.8)
                self.files = all_pairs[:num_train] if split == 'train' else all_pairs[num_train:]
        else:
            # Резервный вариант для локального тестирования с нарезанными тайлами
            split_img_dir = self.root_dir / f"{split}_images"
            split_mask_dir = self.root_dir / f"{split}_masks"
            if split_img_dir.exists() and split_mask_dir.exists():
                logger.warning(
                    "Используются нарезанные тайлы LandCoverAI из %s вместо оригинальных GeoTIFF снимков.",
                    split_img_dir
                )
                for img_path in sorted(split_img_dir.glob("*.jpg")):
                    mask_name = img_path.stem + "_m.png"
                    mask_path = split_mask_dir / mask_name
                    if mask_path.exists():
                        self.files.append((img_path, mask_path))

        logger.info("Инициализация LandCoverAIData (%s): найдено %d полноразмерных снимков.", split, len(self.files))

        for i, (img_path, _) in enumerate(self.files):
            w, h = self._get_image_size(img_path)
            # 33 листа: ~8351..9243 × 9289..9715 (0.25 м/пикс), 8 листов: ~4113..4352 × 4672..4750 (0.50 м/пикс)
            native_gsd = 0.25 if max(w, h) > 6000 else 0.50
            scale_factor = native_gsd / self.target_m_per_px

            sw, sh = max(1, int(w * scale_factor)), max(1, int(h * scale_factor))
            starts = compute_patch_starts(sw, sh, self.resolution)
            for px, py in starts:
                self.patches.append((i, px, py))

    def _get_image_size(self, img_path: pathlib.Path) -> Tuple[int, int]:
        if img_path.suffix.lower() in ('.tif', '.tiff'):
            try:
                with tifffile.TiffFile(img_path) as tif:
                    shape = tif.pages[0].shape
                    if len(shape) >= 3 and shape[0] in (3, 4):
                        return shape[2], shape[1]
                    return shape[1], shape[0]
            except Exception:
                pass
        with Image.open(img_path) as tmp:
            return tmp.size

    def _get_scaled_image_and_mask(self, idx_file: int) -> Tuple[Image.Image, Image.Image]:
        cached = self.cache.get(idx_file)
        if cached is not None:
            return cached

        img_path, mask_path = self.files[idx_file]

        if img_path.suffix.lower() in ('.tif', '.tiff'):
            img_arr = tifffile.imread(img_path)
            if img_arr.ndim == 3 and img_arr.shape[0] in (3, 4):
                img_arr = np.transpose(img_arr, (1, 2, 0))
            if img_arr.ndim == 3 and img_arr.shape[-1] >= 3:
                img_arr = img_arr[:, :, :3]
            img = Image.fromarray(img_arr, mode='RGB')
        else:
            img = Image.open(img_path).convert('RGB')

        w, h = img.size
        native_gsd = 0.25 if max(w, h) > 6000 else 0.50
        scale_factor = native_gsd / self.target_m_per_px

        sw, sh = max(1, int(w * scale_factor)), max(1, int(h * scale_factor))
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)

        if mask_path.suffix.lower() in ('.tif', '.tiff'):
            mask_arr = tifffile.imread(mask_path)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0] if mask_arr.shape[-1] < mask_arr.shape[0] else mask_arr[0, :, :]
        else:
            mask = Image.open(mask_path)
            mask_arr = np.array(mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]

        idx_mask_np = apply_index_map(mask_arr, self.class_mapping)
        idx_mask = Image.fromarray(idx_mask_np, mode='L')
        idx_mask = idx_mask.resize((sw, sh), Image.Resampling.NEAREST)

        res = (img, idx_mask)
        self.cache.put(idx_file, res)
        return res



class GIDData(BasePatchDataset):
    # Обрезка черных полей снимков Gaofen-2 (left, top, right, bottom)
    CROP_BORDER = (60, 54, 7260, 6854)

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
        
        l, t, r, b = self.CROP_BORDER
        crop_w, crop_h = r - l, b - t
        sw, sh = max(1, int(crop_w * self.scale_factor)), max(1, int(crop_h * self.scale_factor))
        starts = compute_patch_starts(sw, sh, self.resolution)

        for i in range(len(self.files)):
            for px, py in starts:
                self.patches.append((i, px, py))

    def _get_scaled_image_and_mask(self, idx_file: int) -> Tuple[Image.Image, Image.Image]:
        cached = self.cache.get(idx_file)
        if cached is not None:
            return cached

        img_path, mask_path = self.files[idx_file]
        
        img_arr = tifffile.imread(img_path)
        if img_arr.ndim == 3 and img_arr.shape[0] in (3, 4):
            img_arr = np.transpose(img_arr, (1, 2, 0))
            
        # Формат NirRGB: каналы [NIR(0), R(1), G(2), B(3)]. Берём каналы [1, 2, 3] для формирования RGB.
        rgb_arr = img_arr[:, :, [1, 2, 3]]
        
        l, t, r, b = self.CROP_BORDER
        rgb_arr = rgb_arr[t:b, l:r]

        img = Image.fromarray(rgb_arr, mode='RGB')
        sw, sh = max(1, int(img.size[0] * self.scale_factor)), max(1, int(img.size[1] * self.scale_factor))
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)

        mask_arr = tifffile.imread(mask_path)
        if mask_arr.ndim == 3 and mask_arr.shape[0] == 3:
            mask_arr = np.transpose(mask_arr, (1, 2, 0))
        mask_arr = mask_arr[t:b, l:r]

        idx_mask_np = apply_color_map(mask_arr, self.class_mapping)
        idx_mask = Image.fromarray(idx_mask_np, mode='L')
        idx_mask = idx_mask.resize((sw, sh), Image.Resampling.NEAREST)

        res = (img, idx_mask)
        self.cache.put(idx_file, res)
        return res


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

    def _get_scaled_image_and_mask(self, idx_file: int) -> Tuple[Image.Image, Image.Image]:
        cached = self.cache.get(idx_file)
        if cached is not None:
            return cached

        img_path, mask_path = self.files[idx_file]
        
        img_arr = tifffile.imread(img_path)
        if len(img_arr.shape) == 3 and img_arr.shape[0] in (3, 4):
            img_arr = np.transpose(img_arr, (1, 2, 0))
            
        rgb_arr = img_arr[:, :, [0, 1, 2]]
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

        res = (img, idx_mask)
        self.cache.put(idx_file, res)
        return res


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

        logger.info("Инициализация DODWData (%s): найдено %d пар. Фильтрация пустых снимков...", split, len(self.files))
        
        valid_files = []
        for img_path, mask_path in self.files:
            img_arr = tifffile.imread(img_path)
            if img_arr.max() == 0:
                continue
            valid_idx = len(valid_files)
            valid_files.append((img_path, mask_path))
            
            shape = img_arr.shape
            h, w = (shape[1], shape[2]) if shape[0] == 4 else (shape[0], shape[1])
            
            sw, sh = max(1, int(w * self.scale_factor)), max(1, int(h * self.scale_factor))
            starts = compute_patch_starts(sw, sh, self.resolution)
            for px, py in starts:
                self.patches.append((valid_idx, px, py))
                
        self.files = valid_files

    def _get_scaled_image_and_mask(self, idx_file: int) -> Tuple[Image.Image, Image.Image]:
        cached = self.cache.get(idx_file)
        if cached is not None:
            return cached

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
        # Канал 0 — исходные разметки (Ground Truth), канал 1 — предсказания модели Dynamic World
        gt_channel = mask_arr[:, :, 0]
        
        idx_mask_np = apply_index_map(gt_channel, self.class_mapping)
        idx_mask = Image.fromarray(idx_mask_np, mode='L')
        idx_mask = idx_mask.resize((sw, sh), Image.Resampling.NEAREST)

        res = (img, idx_mask)
        self.cache.put(idx_file, res)
        return res


def prepare_datasets(resolution=512, target_m_per_px=10.0,
                     deepglobe=True, landcoverai=True, gid=True, whu=True, dodw=True,
                     root_dirs: Optional[Dict[str, str]] = None):
    """
    Создаёт обучающую и валидационную выборки с настроенным пайплайном синхронных аугментаций.
    """
    root_dirs = root_dirs or {}
    
    DEEPGLOBE_ROOT = pathlib.Path(root_dirs.get('deepglobe', '/path/to/DeepGlobe_Land'))
    LANDCOVERAI_ROOT = pathlib.Path(root_dirs.get('landcoverai', '/path/to/landcoverai'))
    GID_ROOT = pathlib.Path(root_dirs.get('gid', '/path/to/GID'))
    WHU_ROOT = pathlib.Path(root_dirs.get('whu', '/path/to/WHU-OPT-SAR dataset'))
    DODW_ROOT = pathlib.Path(root_dirs.get('dodw', '/path/to/Dataset_Open_Dynamic_World_Test_Tiles'))

    # Пайплайн аугментаций строго в установленном порядке
    s_rbct = SyncRandomBrightnessContrastTarget()
    sync_hor = SyncRandomHorizontalFlip()
    sync_v = SyncRandomVerticalFlip()
    sync_rot = SyncRotate360_plus(resolution=resolution)
    sync_r_g_d = RandomGridDistortion()
    sync_r_e_t = RandomElasticTransform()
    sync_r_up_dwn = TrickyResize_UpDwn(resolution=resolution)
    sync_aff_aug = AffineAugmentation(resolution=resolution)
    sync_res = SyncResize(resolution=resolution)
    sync_to_tens = SyncToTensor()
    sync_noise = RandomNoiseSP()

    train_transforms = [
        s_rbct,
        sync_hor,
        sync_v,
        sync_rot,
        sync_r_g_d,
        sync_r_e_t,
        sync_r_up_dwn,
        sync_aff_aug,
        sync_res,
        sync_to_tens,
        sync_noise,
    ]

    val_transforms = [
        sync_res,
        sync_to_tens,
    ]

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
    """
    Формирует батч для семантической сегментации.
    
    Returns:
        images: [B, 3, H, W] float tensor
        masks: [B, H, W] long tensor с индексами классов (0..6)
    """
    images = []
    masks = []
    for img, mask in batch:
        if isinstance(img, torch.Tensor):
            if img.ndim == 2:
                img = img.unsqueeze(0).repeat(3, 1, 1)
            elif img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] > 3:
                img = img[:3]
        
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            mask = mask.long()
            
        images.append(img)
        masks.append(mask)

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks
