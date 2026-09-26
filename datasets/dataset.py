import math
import pathlib
import copy
from PIL.Image import Image
from torch.utils.data import Dataset, ConcatDataset
import xml.etree.ElementTree as ET
import numpy as np
import torch
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageOps
from typing import List, Dict, Tuple, Optional, Union, Any


from .semantic_sync_transforms import SyncCompose, SyncRandomHorizontalFlip, SyncRotate360_plus, Test_error, \
    SyncToTensor, SyncRandomVerticalFlip, TrickyResize_UpDwn, SyncResize, RandomGaussianBlur, \
    RandomNoiseSP, AffineAugmentation, SyncRandomBrightnessContrastTarget, RandomFilter, RandomMirror, \
    RandomPaste, RandomMedianFilter, RandomElasticTransform, RandomGridDistortion, RandomGrayMaker, \
    SyncTimeMosaicStitching

from urllib.parse import unquote

from utils import n_e_augs_shuffle


def pad_image_centered(img: Image.Image, resolution: int, fill=(0, 0, 0)) -> Image.Image:
    """
    Дополняет изображение паддингом до размера resolution x resolution,
    размещая исходное изображение в центре результирующего.

    Если изображение уже не меньше resolution по обеим осям, возвращается как есть.

    :param img: исходное PIL-изображение
    :param resolution: целевой размер стороны (квадрат)
    :param fill: цвет/значение паддинга
    :return: PIL-изображение размера (resolution, resolution)
    """
    w, h = img.size
    if w >= resolution and h >= resolution:
        return img

    padded = Image.new(img.mode, (resolution, resolution), fill)
    # Смещение, при котором исходное изображение оказывается по центру
    offset_x = (resolution - w) // 2
    offset_y = (resolution - h) // 2
    padded.paste(img, (offset_x, offset_y))
    return padded


def pad_mask_centered(mask: Image.Image, resolution: int, fill=0) -> Image.Image:
    """
    Дополняет маску паддингом до размера resolution x resolution,
    размещая исходную маску в центре (аналогично pad_image_centered).

    Паддинг заполняется значением fill (по умолчанию 0 — фон),
    поэтому аннотации автоматически остаются согласованными с изображением:
    смещение маски и изображения одинаковое.

    :param mask: исходная PIL-маска (обычно режим 'L')
    :param resolution: целевой размер стороны (квадрат)
    :param fill: значение паддинга (0 — фон)
    :return: PIL-маска размера (resolution, resolution)
    """
    w, h = mask.size
    if w >= resolution and h >= resolution:
        return mask

    padded = Image.new(mask.mode, (resolution, resolution), fill)
    offset_x = (resolution - w) // 2
    offset_y = (resolution - h) // 2
    padded.paste(mask, (offset_x, offset_y))
    return padded


def compute_patch_starts(size: int, resolution: int) -> List[int]:
    """
    Вычисляет координаты правых/нижних границ патчей (как в исходной логике
    range(resolution, size + 1, step)) для нарезки изображения размера `size`
    на патчи размера `resolution`.

    Если size <= resolution, нарезка на несколько патчей невозможна —
    изображение должно быть дополнено паддингом (см. pad_image_centered /
    pad_mask_centered), и возвращается единственная точка [resolution],
    что соответствует одному патчу на всё изображение.
    """
    if size <= resolution:
        return [resolution]

    step = (size - resolution) // math.ceil((size - resolution) / resolution)
    return list(range(resolution, size + 1, step))


class OptINSInstaSeg(Dataset):
    def __init__(self, images_dir: Path, annotations_path: Path,
                 class_names: List[str] = None, transforms=[],
                 less_data=False, resolution=512, max_objects_per_patch=10000,
                 overlap=0.90, size_scale=1.0, skip_first=False):
        """
        Dataset для обучения Mask R-CNN с использованием XML-аннотаций

        Args:
            images_dir: Путь к папке с изображениями
            annotations_path: Путь к XML-файлу с аннотациями
            class_names: Список имен классов для использования (None для всех)
            transforms: Список аугментаций
            less_data: Процент данных для использования (False для 100%)
            resolution: Размер патчей для разрезания изображений
            max_objects_per_patch: Максимальное количество объектов в одном патче
        """
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.class_names = class_names or ["Building", "Building_base", "Shadow",
                                           "Shadow_with object", "Forest", "Water"]
        self.class_to_idx = {name: idx + 1 for idx, name in enumerate(self.class_names)}  # 0 - фон
        self.idx_to_class = {idx + 1: name for idx, name in enumerate(self.class_names)}
        self.transforms = transforms
        self.less_data = less_data
        self.resolution = resolution
        self.max_objects_per_patch = max_objects_per_patch
        self.aug = True
        self.overlap = overlap
        self.size_scale = size_scale
        self.skip_first = skip_first


        print("Загрузка аннотаций...")
        self.annotations = self._parse_annotations()
        print(f"Загружено аннотаций для {len(self.annotations)} изображений")

        print("Формирование патчей...")
        self.patches = self._create_patches()
        print(f"Создано {len(self.patches)} патчей")

        self.names = list(range(len(self.patches)))
        # Группировка изображений по схожести имен (выделяем подстроку до даты снимка)
        # Пример: "46°44'49"N 71°17'29"W_08_11_2022.png" -> "46°44'49"N 71°17'29"W"
        self.image_groups = {}
        for img_name in self.annotations.keys():
            # Находим разделитель перед датой (предполагаем формат с нижним подчеркиванием перед ДД_ММ_ГГГГ)
            # Если формат фиксирован, можно отрезать последние 14 символов (_ДД_ММ_ГГГГ.png)
            base_key = img_name
            if '_' in img_name:
                parts = img_name.split('_')
                if len(parts) >= 3:  # Проверка, что в конце действительно дата
                    base_key = '_'.join(parts[:-3]) if parts[-1].split('.')[0].isdigit() else img_name

            if base_key not in self.image_groups:
                self.image_groups[base_key] = []
            self.image_groups[base_key].append(img_name)

    def _parse_annotations(self) -> dict[str, dict[str, int | list[Any]]]:
        """
        Парсит XML-файл аннотаций и возвращает словарь:
        {
            "image_name": [
                {
                    "type": "polygon"|"box",
                    "label": "Building",
                    "points": [(x1, y1), (x2, y2), ...]  # для polygon
                    "bbox": [x_min, y_min, x_max, y_max]  # для box
                },
                ...
            ],
            ...
        }
        """
        tree = ET.parse(self.annotations_path)
        root = tree.getroot()

        annotations_dict = {}

        # Сначала собираем все изображения
        for image_elem in root.findall('.//image'):
            image_name = image_elem.get('name')
            # Нормализуем имя файла (удаляем %-кодирование и специальные символы)
            clean_name = unquote(image_name)

            image_path_decoded = self.images_dir / clean_name
            image_path_original = self.images_dir / image_name
            if image_path_original.exists():
                clean_name = image_name
            elif not image_path_decoded.exists():
                print(
                    f"Предупреждение: изображение не найдено ни с именем '{image_name}', ни с именем '{clean_name}'")
                continue
            width = int(int(image_elem.get('width')) * self.size_scale)
            height = int(int(image_elem.get('height')) * self.size_scale)

            annotations_dict[clean_name] = {
                'width': width,
                'height': height,
                'annotations': []
            }

            # Собираем все аннотации для этого изображения
            for polygon_elem in image_elem.findall('.//polygon'):
                label = polygon_elem.get('label')
                if label not in self.class_names:
                    # print('label not in class_names')
                    continue

                points_str = polygon_elem.get('points')
                points = []
                for point_str in points_str.split(';'):
                    x_str, y_str = point_str.split(',')
                    points.append((float(x_str) * self.size_scale, float(y_str) * self.size_scale))

                annotations_dict[clean_name]['annotations'].append({
                    'type': 'polygon',
                    'label': label,
                    'points': points
                })

            for box_elem in image_elem.findall('.//box'):
                label = box_elem.get('label')
                if label not in self.class_names:
                    # print('label not in class_names')
                    continue

                xtl = float(box_elem.get('xtl')) * self.size_scale
                ytl = float(box_elem.get('ytl')) * self.size_scale
                xbr = float(box_elem.get('xbr')) * self.size_scale
                ybr = float(box_elem.get('ybr')) * self.size_scale

                annotations_dict[clean_name]['annotations'].append({
                    'type': 'box',
                    'label': label,
                    'bbox': [xtl, ytl, xbr, ybr]
                })

        return annotations_dict

    def _create_patches(self) -> List[Dict]:
        """
        Создает патчи из изображений с аннотациями
        Возвращает список словарей:
        [
            {
                'image_path': Path,
                'image_name': str,
                'patch_coords': (left, top, right, bottom),
                'annotations': [...]  # аннотации, попадающие в этот патч
            },
            ...
        ]
        """
        patches = []

        for image_name, image_data in tqdm(self.annotations.items(), desc='Создание патчей'):

            image_path = self.images_dir / image_name

            if not image_path.exists():
                print(f"Предупреждение: изображение не найдено: {image_path}")
                continue

            img = Image.open(image_path).convert('RGB')
            if self.size_scale != 1.0:
                new_w = int(img.width * self.size_scale)
                new_h = int(img.height * self.size_scale)
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            width, height = img.size

            # Если разрешение не задано, используем всё изображение как один патч
            if not self.resolution:
                patches.append({
                    'image_path': image_path,
                    'image_name': image_name,
                    'patch_coords': (0, 0, width, height),
                    'annotations': image_data['annotations']
                })
                continue

            # Вычисляем шаг для разрезания с перекрытием (90%)
            overlap = int(self.resolution * self.overlap)
            step = self.resolution - overlap

            # Создаем сетку патчей
            for top in range(0, height, step):
                for left in range(0, width, step):
                    right = min(left + self.resolution, width)
                    bottom = min(top + self.resolution, height)
                    patch_width = right - left
                    patch_height = bottom - top
                    # Фильтруем аннотации, попадающие в этот патч
                    patch_annotations = []
                    for ann in image_data['annotations']:
                        if ann['type'] == 'polygon':
                            # Проверяем, пересекается ли полигон с патчем
                            if self._polygon_intersects_patch(ann['points'],
                                                              (left, top, right, bottom)):
                                patch_annotations.append(ann)
                        elif ann['type'] == 'box':
                            # Проверяем, пересекается ли бокс с патчем
                            if self._box_intersects_patch(ann['bbox'],
                                                          (left, top, right, bottom)):
                                patch_annotations.append(ann)

                    # Сохраняем патч, даже если в нем нет аннотаций (может быть полезно для фона)
                    # Но если патч слишком маленький и мы не используем padding, пропускаем его
                    if patch_width < self.resolution * 0.1 or patch_height < self.resolution * 0.1:
                        continue

                    needs_padding = (patch_width < self.resolution or patch_height < self.resolution)

                    patches.append({
                        'image_path': image_path,
                        'image_name': image_name,
                        'patch_coords': (left, top, right, bottom),
                        'annotations': patch_annotations,
                        'needs_padding': needs_padding,
                        'original_size': (patch_width, patch_height)
                        # сохраняем оригинальный размер для корректного padding
                    })
        return patches

    def _pad_image(self, img: Image.Image,
                            original_size: Tuple[int, int]) -> Image:
        """
        Дополняет изображение паддингом до размера self.resolution x self.resolution,
        размещая исходное изображение в центре (аналогично pad_image_centered).
        """

        orig_w, orig_h = original_size
        target_size = self.resolution

        # Если изображение уже нужного размера, возвращаем как есть
        if orig_w >= target_size and orig_h >= target_size:
            return img

        # Создаем новое изображение с черным фоном (0 для всех каналов)
        padded_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))

        # Смещение, при котором исходное изображение оказывается по центру
        offset_x = (target_size - orig_w) // 2
        offset_y = (target_size - orig_h) // 2
        padded_img.paste(img, (offset_x, offset_y))

        return padded_img

    def _pad_target(self, target: Dict[str, torch.Tensor], target_size: int) -> Dict[str, torch.Tensor]:
        """
        Дополняет маску (3, H, W) до target_size x target_size,
        размещая исходную маску в центре (аналогично _pad_image).

        Паддинг заполняется: канал фона = 1 (фон), остальные = 0.
        Смещение маски точно совпадает со смещением изображения в _pad_image,
        поэтому аннотации остаются согласованными с изображением.
        """
        masks = target['masks']
        # Ожидаем размер (3, H, W)
        _, curr_h, curr_w = masks.shape

        if curr_w == target_size and curr_h == target_size:
            return target

        # Смещение, при котором исходная маска оказывается по центру.
        # ВАЖНО: должно совпадать со смещением в _pad_image, иначе маска
        # не будет соответствовать изображению.
        offset_x = (target_size - curr_w) // 2
        offset_y = (target_size - curr_h) // 2

        # Создаем новый тензор из нулей
        padded_masks = torch.zeros((3, target_size, target_size), dtype=masks.dtype)

        # Заполняем канал фона (индекс 0) единицами по всему холсту:
        # добавленные черные пиксели изображения — это тоже фон.
        padded_masks[0, :, :] = 1

        # Копируем оригинальную маску в центр (перезаписывает фон
        # в центральной области значениями исходной маски)
        padded_masks[:, offset_y:offset_y + curr_h, offset_x:offset_x + curr_w] = masks

        return {
            'masks': padded_masks,
            'labels': target['labels'], # просто прокидываем пустые тензоры
            'boxes': target['boxes']
        }

    def _polygon_intersects_patch(self, polygon_points: List[Tuple[float, float]],
                                  patch_coords: Tuple[int, int, int, int]) -> bool:
        """
        Проверяет, пересекается ли полигон с патчем
        """
        left, top, right, bottom = patch_coords

        # Быстрая проверка по bounding box полигона
        poly_xs = [p[0] for p in polygon_points]
        poly_ys = [p[1] for p in polygon_points]
        poly_min_x, poly_max_x = min(poly_xs), max(poly_xs)
        poly_min_y, poly_max_y = min(poly_ys), max(poly_ys)

        # Если bounding box не пересекается с патчем - полигон точно не пересекается
        if poly_max_x < left or poly_min_x > right or poly_max_y < top or poly_min_y > bottom:
            return False

        # Подробная проверка: хотя бы одна точка полигона внутри патча
        for x, y in polygon_points:
            if left <= x <= right and top <= y <= bottom:
                return True

        return False

    def _box_intersects_patch(self, box: List[float],
                              patch_coords: Tuple[int, int, int, int]) -> bool:
        """
        Проверяет, пересекается ли bounding box с патчем
        """
        box_x1, box_y1, box_x2, box_y2 = box
        patch_x1, patch_y1, patch_x2, patch_y2 = patch_coords

        # Проверка пересечения прямоугольников
        return not (box_x2 < patch_x1 or box_x1 > patch_x2 or
                    box_y2 < patch_y1 or box_y1 > patch_y2)

    def _create_target_for_patch(self, annotations: List[Dict],
                                 patch_coords: Tuple[int, int, int, int]) -> Dict[str, torch.Tensor]:
        """
        Создает target для Semantic Segmentation.
        Возвращает словарь с ключом 'masks' размера (3, H, W):
        - канал 0: Фон
        - канал 1: Forest
        - канал 2: Water
        """
        left, top, right, bottom = patch_coords
        patch_width = right - left
        patch_height = bottom - top

        # 1. Создаем однослойную маску (L mode), куда будем рисовать классы.
        # Изначально заполнена нулями (фон).
        combined_mask = Image.new('L', (patch_width, patch_height), 0)
        draw = ImageDraw.Draw(combined_mask)

        for ann in annotations:
            label = ann['label']
            # Получаем числовой индекс класса (1 или 2)
            class_idx = self.class_to_idx.get(label)
            if class_idx is None:
                continue

            if ann['type'] == 'polygon':
                # Сдвигаем координаты полигона к началу патча
                shifted_points = [(x - left, y - top) for x, y in ann['points']]
                draw.polygon(shifted_points, fill=class_idx)

            elif ann['type'] == 'box':
                # Сдвигаем координаты бокса
                x1 = max(0, float(ann['bbox'][0] - left))
                y1 = max(0, float(ann['bbox'][1] - top))
                x2 = min(patch_width, float(ann['bbox'][2] - left))
                y2 = min(patch_height, float(ann['bbox'][3] - top))
                draw.rectangle([x1, y1, x2, y2], fill=class_idx)

        # 2. Конвертируем в numpy array (H, W) со значениями 0, 1, 2
        mask_np = np.array(combined_mask, dtype=np.uint8)

        # 3. Создаем 3-канальную маску (C, H, W)
        # Используем булевы операции для распределения по каналам
        final_masks = np.zeros((3, patch_height, patch_width), dtype=np.uint8)

        final_masks[0] = (mask_np == 0) # Канал 0: True там, где фон
        final_masks[1] = (mask_np == 1) # Канал 1: True там, где Forest
        final_masks[2] = (mask_np == 2) # Канал 2: True там, где Water

        masks_tensor = torch.from_numpy(final_masks).to(dtype=torch.float32)

        # Возвращаем словарь.
        # boxes и labels оставляем пустыми заглушками, чтобы не ломать пайплайн трансформаций.
        return {
            'masks': masks_tensor,
            'labels': torch.empty((0,), dtype=torch.int64),
            'boxes': torch.empty((0, 4), dtype=torch.float32)
        }

    def __len__(self):
        if self.less_data and self.less_data < 100:
            return int(len(self.patches) * (self.less_data / 100))
        return len(self.patches)

    def __getitem__(self, idx):

        if self.less_data and self.less_data < 100:
            idx = random.randint(0, len(self.patches) - 1)

        patch_info = self.patches[idx]

        # Загружаем изображение и обрезаем патч
        img = Image.open(patch_info['image_path']).convert('RGB')
        if self.size_scale != 1.0:
            new_w = int(img.width * self.size_scale)
            new_h = int(img.height * self.size_scale)
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        patch = img.crop(patch_info['patch_coords'])

        # Создаем target в формате Mask R-CNN
        target = self._create_target_for_patch(
            patch_info['annotations'],
            patch_info['patch_coords']
        )

        # Если патч маленький и требуется дополнение - дополняем
        if patch_info.get('needs_padding', False):
            patch = self._pad_image(patch, patch_info.get('original_size', (patch.width, patch.height)))
            target = self._pad_target(target, self.resolution)

        # 1. Переводим в numpy (и переносим на CPU, если тензор был на GPU)
        mask_np = target['masks'].cpu().numpy()

        # 2. Меняем оси (C, H, W) -> (H, W, C)
        mask_np = np.transpose(mask_np, (1, 2, 0))

        # 3. Приводим к типу uint8 (PIL требует именно этот тип)
        # Значения будут 0 и 1. Если нужно "увидеть" маску, можно умножить на 255: (mask_np * 255).astype(np.uint8)
        mask_np = mask_np.astype(np.uint8) * 255

        # 4. Создаем PIL изображение
        pil_mask = Image.fromarray(mask_np, mode='RGB')

        if self.aug:
            # Находим имя базовой группы для текущего изображения
            img_name = patch_info['image_name']
            base_key = img_name
            if '_' in img_name:
                parts = img_name.split('_')
                if len(parts) >= 3 and parts[-1].split('.')[0].isdigit():
                    base_key = '_'.join(parts[:-3])

            group_images = self.image_groups.get(base_key, [img_name])

            # Собираем аналогичные патчи и аннотации с других изображений группы
            group_patches_data = []
            for alt_img_name in group_images:
                if alt_img_name == img_name:
                    continue
                # Находим аннотации для этого альтернативного изображения
                alt_img_data = self.annotations.get(alt_img_name)
                if alt_img_data:
                    group_patches_data.append({
                        'image_path': self.images_dir / alt_img_name,
                        'annotations': alt_img_data['annotations']
                    })

            # Передаем собранную группу и метаинформацию в пайплайн трансформаций
            sync_transforms = SyncCompose(n_e_augs_shuffle(self.transforms, skip_first=self.skip_first))
            patch, target = sync_transforms(
                img=patch,
                mask=pil_mask,
                names=self.names,
                img_path=patch_info['image_path'],
                patch_coords=patch_info['patch_coords'],
                needs_padding=patch_info.get('needs_padding', False),
                original_size=patch_info.get('original_size', (patch.width, patch.height)),
                group_patches_data=group_patches_data,
                dataset_obj=self
            )

        return patch, target


class PotsdamData(Dataset):
    def __init__(self, masks_paths, images_paths, tensor_classes, transforms=[], less_data=False, resolution=512):
        self.less_data = less_data
        self.masks = list()
        self.imgs = list()
        self.resolution = resolution
        # Вычисляем коэффициент масштабирования
        # Исходный масштаб: 1.8 метра на 35 пикселей = 0.0514 м/пикс
        # Целевой масштаб: 0.5 м/пикс
        current_m_per_px = 1.8 / 35
        target_m_per_px = 0.5
        scale_factor = current_m_per_px / target_m_per_px  # Примерно 0.103 (уменьшение размера)
        # Классы Potsdam (для справки и обработки значений)
        # 255 - непроницаемые, 226 - транспорт, 179 - низкая растительность,
        # 150 - деревья, 76 - фон, 29 - здания
        # Если классы не переданы, используем стандартный набор Potsdam
        if tensor_classes is None:
            tensor_classes = [0, 29, 76, 150, 179, 226, 255]
        # Если передан numpy массив, конвертируем его в список
        if isinstance(tensor_classes, np.ndarray):
            tensor_classes = tensor_classes.tolist()
        # Убедимся, что 0 есть (нужен для паддинга/фона)
        if 0 not in tensor_classes:
            tensor_classes.insert(0, 0)

        self.ten_cls = sorted(tensor_classes)

        # Разницы для восстановления классов после трансформаций
        self.ten_cls_diff = np.zeros(len(self.ten_cls) - 1)
        for i in range(len(self.ten_cls) - 1):
            self.ten_cls_diff[i] = self.ten_cls[i + 1] - self.ten_cls[i]

        for i in tqdm(range(len(masks_paths)), desc='Формирование изображений'):
            img = Image.open(images_paths / masks_paths[i].name.replace('label.', 'RGB.'))
            mask = Image.open(masks_paths[i]).convert('L')
            w, h = img.size
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            # Защита от нулевых размеров
            if new_w < 1: new_w = 1
            if new_h < 1: new_h = 1
            # Для изображения используем качественное уменьшение (LANCZOS)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Для маски используем метод ближайшего соседа (NEAREST), чтобы не смешивать классы!
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

            if self.resolution:
                width, height = img.size
                # Случай A: Изображение МЕНЬШЕ нужного размера -> Делаем паддинг
                if width < self.resolution or height < self.resolution:
                    delta_w = self.resolution - width
                    delta_h = self.resolution - height

                    # Паддинг равномерно со всех сторон
                    pad_left = delta_w // 2
                    pad_right = delta_w - pad_left
                    pad_top = delta_h // 2
                    pad_bottom = delta_h - pad_top

                    # Паддинг изображения (черный цвет)
                    img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
                    # Паддинг маски (значение 0, которое будет фоном)
                    mask = ImageOps.expand(mask, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

                    self.imgs.append(img)
                    self.masks.append(mask)

                # Случай B: Изображение БОЛЬШЕ или РАВНО -> Скользящее окно (как было)
                else:
                    step_h = (height - self.resolution) // math.ceil((height - self.resolution) / self.resolution)
                    step_w = (width - self.resolution) // math.ceil((width - self.resolution) / self.resolution)

                    # Защита от нулевого шага, если размер чуть больше resolution
                    step_h = max(1, step_h)
                    step_w = max(1, step_w)

                    for y in range(self.resolution, height + 1, step_h):
                        for x in range(self.resolution, width + 1, step_w):
                            left = x - self.resolution
                            top = y - self.resolution
                            right = x
                            bottom = y

                            new_img = img.crop((left, top, right, bottom))
                            new_mask = mask.crop((left, top, right, bottom))

                            self.imgs.append(new_img)
                            self.masks.append(new_mask)
            else:
                # Если resolution не задан, просто сохраняем уменьшенное изображение
                self.imgs.append(img)
                self.masks.append(mask)

        self.names = list(range(len(self.imgs)))
        self.transforms = transforms
        self.aug = True

        print(f'Количество изображений = {len(self.imgs)}')

    def __len__(self):
        if self.less_data:
            return int((len(self.imgs) * self.less_data) // 100)
        else:
            return len(self.imgs)

    def __getitem__(self, item):
        if self.less_data:
            item = np.random.randint(len(self.imgs))

        img = self.imgs[item]
        mask = self.masks[item]

        mask_np = np.array(mask)
        h, w = mask_np.shape
        target_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Значение класса "Деревья/Лес" (сырое значение до нормализации)
        # В списке ten_cls: [0, 29, 76, 150, 179, 226, 255] -> индекс 3 это 150
        forest_val_raw = 150

        is_forest = (mask_np == forest_val_raw)
        # Формируем каналы:
        # Канал 0: Фон (все, что не лес). Заполняем 255 там, где НЕ лес.
        # Канал 1: Лес. Заполняем 255 там, где лес.
        # Канал 2: Вода (в Potsdam отсутствует, оставляем 0).

        target_mask[:, :, 0] = (~is_forest) * 255  # Фон
        target_mask[:, :, 1] = is_forest * 255  # Лес

        # Конвертируем обратно в PIL для трансформаций
        mask = Image.fromarray(target_mask, mode='RGB')


        if self.aug:
            sync_transforms = SyncCompose(n_e_augs_shuffle(self.transforms))
            pic, msk = sync_transforms(img=img, mask=mask,
                                       names=self.names, img_path=self.imgs, mask_path=self.masks)
            # После трансформаций значения могут быть не строго 0/1 (из-за интерполяции или шума),
            # поэтому делаем пороговое отсечение (thresholding).
            msk = (msk > 0.5).float()

            return pic, msk

        # Если аугментации выключены
        return img, mask


class BAMFORESTS(Dataset):
    def __init__(self, data_path, transforms=[], less_data=False, resolution=512, train_mode=True):
        """
        Args:
            data_path: Путь к папке reconstructed_train
            transforms: Список трансформаций (SyncCompose)
            less_data: Процент данных для использования (если < 100)
            resolution: Размер патча (по умолчанию 512)
        """
        self.less_data = less_data
        self.resolution = resolution
        self.imgs = list()
        self.masks = list()
        self.train_mode = train_mode

        # --- НАСТРОЙКА МАСШТАБА ---
        # Текущий: 0.017 м/пиксель
        # Целевой: 0.5 м/пиксель
        current_gsd = 0.017
        target_gsd = 0.5
        self.scale_factor = current_gsd / target_gsd  # ~0.034

        data_path = Path(data_path)

        # Ищем все изображения (по шаблону имени reconstructed.tif)
        # Ваши файлы названы: prefix_reconstructed.tif
        img_files = sorted(list(data_path.glob("*_reconstructed.tif")))

        if not img_files:
            print(f"Предупреждение: в папке {data_path} не найдено файлов по шаблону '*_reconstructed.tif'")

        for img_path in tqdm(img_files, desc='Обработка датасета'):
            # Формируем путь к маске: prefix_mask.png
            mask_path = data_path / img_path.name.replace("_reconstructed.tif", "_mask.png")

            if not mask_path.exists():
                print(f"Пропуск {img_path.name}: не найдена маска {mask_path.name}")
                continue

            # 1. Загрузка
            try:
                img = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')
            except Exception as e:
                print(f"Ошибка чтения {img_path.name}: {e}")
                continue

            # 2. Ресайз (Downscaling)
            w, h = img.size
            new_w = max(1, int(w * self.scale_factor))
            new_h = max(1, int(h * self.scale_factor))

            # LANCZOS для изображений (качество), NEAREST для масок (сохранение меток)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

            # 3. Нарезка на патчи или Паддинг
            if self.resolution:
                width, height = img.size

                # СЛУЧАЙ А: Изображение МЕНЬШЕ resolution -> Паддинг
                if width < self.resolution or height < self.resolution:

                    self.imgs.append(img)
                    self.masks.append(mask)

                # СЛУЧАЙ Б: Изображение БОЛЬШЕ или РАВНО -> Скользящее окно
                else:
                    # Вычисляем шаг (чтобы покрыть всё изображение)
                    step_h = (height - self.resolution) // math.ceil((height - self.resolution) / self.resolution)
                    step_w = (width - self.resolution) // math.ceil((width - self.resolution) / self.resolution)

                    step_h = max(1, step_h)
                    step_w = max(1, step_w)

                    for y in range(self.resolution, height + 1, step_h):
                        for x in range(self.resolution, width + 1, step_w):
                            left = x - self.resolution
                            top = y - self.resolution
                            right = x
                            bottom = y

                            crop_img = img.crop((left, top, right, bottom))
                            crop_mask = mask.crop((left, top, right, bottom))

                            self.imgs.append(crop_img)
                            self.masks.append(crop_mask)
            else:
                # Если resolution не задан
                self.imgs.append(img)
                self.masks.append(mask)

        self.names = list(range(len(self.imgs)))
        self.transforms = transforms
        self.aug = True

        print(f'Количество патчей = {len(self.imgs)}')

    def __len__(self):
        if self.less_data and self.less_data < 100:
            return int(len(self.imgs) * (self.less_data / 100))
        return len(self.imgs)

    def __getitem__(self, item):
        if self.less_data and self.less_data < 100:
            item = np.random.randint(len(self.imgs))

        img_tl = self.imgs[item]
        mask_tl = self.masks[item]

        if self.train_mode:

            # Случайные изображения для остальных позиций
            item1 = np.random.randint(len(self.imgs))
            item2 = np.random.randint(len(self.imgs))
            item3 = np.random.randint(len(self.imgs))

            img_tr = self.imgs[item1]  # Top-Right
            mask_tr = self.masks[item1]

            img_bl = self.imgs[item2]  # Bottom-Left
            mask_bl = self.masks[item2]

            img_br = self.imgs[item3]  # Bottom-Right
            mask_br = self.masks[item3]

            # 1. Получаем размеры каждого изображения
            w_tl, h_tl = img_tl.size
            w_tr, h_tr = img_tr.size
            w_bl, h_bl = img_bl.size
            w_br, h_br = img_br.size

            # 2. Вычисляем размеры строк и общие размеры итоговой мозаики

            # Высота верхней строки определяется самым высоким изображением в ней (TL или TR)
            h_top_row = max(h_tl, h_tr)
            # Высота нижней строки определяется самым высоким изображением в ней (BL или BR)
            h_bottom_row = max(h_bl, h_br)

            # Общая ширина мозаики определяется самой широкой строкой
            # Верхняя строка: ширина TL + ширина TR
            # Нижняя строка: ширина BL + ширина BR
            total_width = max(w_tl + w_tr, w_bl + w_br)
            total_height = h_top_row + h_bottom_row

            # 3. Создаем пустые холсты (Canvas)
            # Image.new заполняет изображение черным цветом (0), что и есть паддинг фона
            mosaic_img = Image.new('RGB', (total_width, total_height), (0, 0, 0))
            mosaic_mask = Image.new('L', (total_width, total_height), 0)  # 0 для маски = фон

            # 4. Вставляем изображения на холст (координаты paste - левый верхний угол)

            # --- Верхняя строка ---
            # Top-Left: координаты (0, 0)
            mosaic_img.paste(img_tl, (0, 0))
            mosaic_mask.paste(mask_tl, (0, 0))

            # Top-Right: координаты (ширина TL, 0)
            # Если TR меньше по высоте, чем TL, снизу останется черный паддинг
            mosaic_img.paste(img_tr, (w_tl, 0))
            mosaic_mask.paste(mask_tr, (w_tl, 0))

            # --- Нижняя строка ---
            # Bottom-Left: координаты (0, высота верхней строки)
            # Если BL меньше по ширине, чем TL, справа останется паддинг (или стык с BR)
            mosaic_img.paste(img_bl, (0, h_top_row))
            mosaic_mask.paste(mask_bl, (0, h_top_row))

            # Bottom-Right: координаты (ширина BL, высота верхней строки)
            mosaic_img.paste(img_br, (w_bl, h_top_row))
            mosaic_mask.paste(mask_br, (w_bl, h_top_row))
        else:
            mosaic_img = img_tl
            mosaic_mask = mask_tl

        width, height = mosaic_img.size

        # СЛУЧАЙ А: Изображение МЕНЬШЕ resolution -> Паддинг (центрированный)
        if width < self.resolution or height < self.resolution:
            mosaic_img = pad_image_centered(mosaic_img, self.resolution)
            mosaic_mask = pad_mask_centered(mosaic_mask, self.resolution)

        # Проверяем, если изображение больше заданного разрешения
        if width > self.resolution or height > self.resolution:
            # Координаты для обрезки: (left, top, right, bottom)
            # Мы берем квадрат self.resolution x self.resolution от точки (0, 0)
            crop_box = (0, 0, self.resolution, self.resolution)

            mosaic_img = mosaic_img.crop(crop_box)
            mosaic_mask = mosaic_mask.crop(crop_box)

        m_np = np.array(mosaic_mask)
        target_np = np.zeros((m_np.shape[0], m_np.shape[1], 3), dtype=np.uint8)

        is_forest = m_np > 0

        # Формируем каналы (как в других датасетах)
        target_np[:, :, 0] = (~is_forest) * 255  # Канал 0: Фон
        target_np[:, :, 1] = is_forest * 255  # Канал 1: Лес
        # Канал 2 (Вода) остается нулями

        mosaic_mask_rgb = Image.fromarray(target_np, mode='RGB')

        if self.aug:
            # Применяем трансформации
            # Если ваш SyncCompose требует img_path/mask_path, передаем заглушки или пустые строки
            sync_transforms = SyncCompose(n_e_augs_shuffle(self.transforms))
            pic, msk = sync_transforms(img=mosaic_img, mask=mosaic_mask_rgb,
                                       names=self.names, img_path="", mask_path="")
        else:
            pic, msk = mosaic_img, mosaic_mask_rgb


        return pic, msk


class ChesapeakeData(Dataset):
    def __init__(self, root_dir, states=['de', 'md', 'ny', 'pa', 'va', 'wv'],
                 splits=['train'], transforms=[], resolution=512):
        """
        Dataset для Chesapeake Land Cover (Semantic Segmentation).

        Args:
            root_dir: Путь к корневой папке (cvpr_chesapeake_landcover)
            states: Список штатов для использования (коды: de, md, ny, pa, va, wv)
            splits: Список выборок ('train', 'val', 'test')
            transforms: Аугментации (SyncCompose)
            resolution: Размер патча для нарезки. Если None, возвращаются целые тайлы.
        """
        self.root_dir = pathlib.Path(root_dir)
        self.resolution = resolution
        self.transforms = transforms
        self.aug = True if transforms else False

        # Маппинг классов датасета Chesapeake
        # 1: Вода, 2: Деревья/Лес, 3: Низкая растительность,
        # 4: Голая земля, 5: Застройка, 6: Дороги, 15: Нет данных
        self.class_water = 1
        self.class_forest = 2

        self.patches = []
        print("Сканирование датасета Chesapeake...")

        # 1. Сбор всех папок, соответствующих выбранным штатам и сплитам
        folder_paths = []
        for state in states:
            for split in splits:
                # Паттерн поиска папок: root/state_*_split_tiles
                # Пример: de_1m_2013_extended-debuffered-train_tiles
                print('Поиск ', f"{state}_*-{split}_tiles")
                pattern = f"{state}_*-{split}_tiles"
                found = list(self.root_dir.glob(pattern))
                folder_paths.extend(found)

        if not folder_paths:
            print(f"Предупреждение: папки не найдены по паттернам в {self.root_dir}")
            return

        # 2. Обработка каждой папки
        for folder in tqdm(folder_paths, desc="Обработка папок"):
            # Ищем все маски (_lc.tif) в папке
            mask_files = list(folder.glob("*_lc.tif"))

            for mask_path in mask_files:
                # Формируем имя изображения: заменяем суффикс _lc.tif на _naip-new.tif
                img_name = mask_path.name.replace("_lc.tif", "_naip-new.tif")
                img_path = folder / img_name

                if not img_path.exists():
                    print('отсутствует', img_path)

                # Если включена нарезка, вычисляем координаты патчей
                if self.resolution:
                    try:
                        # Быстро получаем размеры изображения без загрузки пикселей
                        with Image.open(img_path) as temp_img:
                            w, h = temp_img.size

                        # Проходим сеткой по изображению
                        # Шаг равен разрешению (без перекрытия)
                        for top in range(0, h, self.resolution):
                            for left in range(0, w, self.resolution):
                                right = left + self.resolution
                                bottom = top + self.resolution

                                # Строгое условие: если патч выходит за границы - пропускаем (NO PADDING)

                                self.patches.append({
                                    'img_path': str(img_path),
                                    'mask_path': str(mask_path),
                                    'coords': (left, top, right, bottom)
                                })
                    except Exception as e:
                        print(f"Ошибка при чтении {img_path}: {e}")
                else:
                    # Если нарезка не нужна, храним инфо о полном файле
                    self.patches.append({
                        'img_path': str(img_path),
                        'mask_path': str(mask_path),
                        'coords': None
                    })
        self.names = list(range(len(self.patches)))
        print(f"Загружено {len(self.patches)} сэмплов.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]

        # Загрузка изображений
        # Используем Pillow для открытия TIFF
        img = Image.open(patch_info['img_path'])
        mask = Image.open(patch_info['mask_path'])

        # Обработка изображения (NAIP: 4 канала -> берем первые 3 RGB)
        # Если изображение 4-канальное, берем RGB для совместимости с моделями
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode == 'L':
            img = img.convert('RGB')
        # Если mode='CMYK' или что-то еще, конвертируем в RGB
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Нарезка (Crop), если задан resolution
        if patch_info['coords']:
            left, top, right, bottom = patch_info['coords']
            img = img.crop((left, top, right, bottom))
            mask = mask.crop((left, top, right, bottom))
            # Проверяем, нужно ли дополнять (паддинг)
            current_w, current_h = img.size

            if current_w < self.resolution or current_h < self.resolution:
                # Используем центрированный паддинг для консистентности
                img = pad_image_centered(img, self.resolution)
                mask = pad_mask_centered(mask, self.resolution)

        # Перевод маски в numpy для быстрой обработки
        mask_np = np.array(mask, dtype=np.uint8)

        # --- Формирование 3-канальной маски (Фон, Лес, Вода) ---
        # Создаем заготовку (3, H, W)
        h, w = mask_np.shape
        target_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # Канал 1: Лес (класс 2) -> в RGB это канал G (индекс 1)
        target_mask[:, :, 1] = (mask_np == self.class_forest).astype(
            np.uint8) * 255  # Умножаем на 255 для наглядности, хотя для сети можно и 1

        # Канал 2: Вода (класс 1) -> в RGB это канал B (индекс 2)
        target_mask[:, :, 2] = (mask_np == self.class_water).astype(np.uint8) * 255

        # Канал 0: Фон -> в RGB это канал R (индекс 0)
        # Фон = всё, что не Лес и не Вода
        not_forest_or_water = (target_mask[:, :, 1] + target_mask[:, :, 2]) == 0
        target_mask[:, :, 0] = not_forest_or_water.astype(np.uint8) * 255

        # Создаем PIL изображение
        mask_pil = Image.fromarray(target_mask, mode='RGB')
        # Аугментации
        if self.aug and self.transforms:
            sync_transforms = SyncCompose(n_e_augs_shuffle(self.transforms))
            # Если вы используете простой вариант без словаря:
            pic, mask_tensor = sync_transforms(img=img, mask=mask_pil, names=self.names, img_path=patch_info['img_path'],
                                               mask_path=patch_info['mask_path'])

            return pic, mask_tensor

        return img, mask_pil





# def create_testaug(prev_name):
#     return testdata(prev_name)

def prepare_forest_water_datasets(resolution=512, image_size = None, chb_mode=False, our=True, potsdam=True, bam=True, ches=True, less_opensar=False):
    main_transforms_list = []

    s_rbct = SyncRandomBrightnessContrastTarget()
    sync_hor = SyncRandomHorizontalFlip()
    sync_v = SyncRandomVerticalFlip()
    syns_rp = SyncRotate360_plus(resolution=resolution)
    sync_rs = SyncResize(resolution)
    rs_up_dwn = TrickyResize_UpDwn(resolution=resolution, minmax_size_up=[117, 200])
    aa = AffineAugmentation(p=0.75, translate_percent=(-0.5, 0.5), scale=(0.7, 1.1))
    # r_m = RandomMirror(p=0.7, resolution=resolution)
    sync_totensor = SyncToTensor()
    rn = RandomNoiseSP()
    ret = RandomElasticTransform()
    rgd = RandomGridDistortion()
    # ВАЖНО: передаём resolution, иначе мозаика будет падить альтернативные
    # маски до 512 (дефолт), а изображения — до self.resolution датасета (1024),
    # что приведёт к рассинхрону размеров и ошибке broadcast при сшивке краевых патчей
    stms = SyncTimeMosaicStitching(p=0.5, resolution=resolution)

    if chb_mode:
        rgm = RandomGrayMaker(p=1.0)
        main_transforms_list.append(rgm)

    main_transforms_list.append(s_rbct)
    main_transforms_list.append(sync_hor)
    main_transforms_list.append(sync_v)
    main_transforms_list.append(syns_rp)
    main_transforms_list.append(rgd)
    main_transforms_list.append(ret)
    # main_transforms_list.append(r_m)
    main_transforms_list.append(rs_up_dwn)
    # main_transforms_list.append(srsmain)
    main_transforms_list.append(aa)
    main_transforms_list.append(sync_rs)
    main_transforms_list.append(sync_totensor)
    main_transforms_list.append(rn)

    our_transform_list = copy.deepcopy(main_transforms_list)
    our_transform_list.insert(0, stms)


    train_dataset_dir = Path('/mnt/980EAB530EAB2968/Segmentation_Dataset/Potsdam/images')
    val_dataset_dir = Path('/mnt/980EAB530EAB2968/Segmentation_Dataset/Potsdam/images_test')
    train_mask_dataset_dir = Path('/mnt/980EAB530EAB2968/Segmentation_Dataset/Potsdam/labels')
    val_mask_dataset_dir = Path('/mnt/980EAB530EAB2968/Segmentation_Dataset/Potsdam/labels_test')
    train_mask_glob = train_mask_dataset_dir.glob("*")
    val_mask_glob = val_mask_dataset_dir.glob("*")
    t_masks = [x for x in train_mask_glob]
    v_masks = [x for x in val_mask_glob]
    train_arr = np.array(t_masks)
    val_arr = np.array(v_masks)
    testlist = []
    for v in v_masks:
        find_cls = Image.open(v).convert('L')
        all_cls = np.array(find_cls)
        testlist.append(all_cls[:3704, :5555])

    t_n_cls = np.unique(testlist)

    data_train = []
    val_train = []

    if potsdam:
        print('Датасет Potsdam')
        print('Для обучения')
        if less_opensar:
            train_dataset = PotsdamData(train_arr, train_dataset_dir, t_n_cls,
                                        transforms=main_transforms_list, less_data=less_opensar, resolution=resolution)

        else:
            train_dataset = PotsdamData(train_arr, train_dataset_dir, t_n_cls,
                                        transforms=main_transforms_list, resolution=resolution)

        data_train.append(train_dataset)

        print('Для тестирования')
        val_dataset = PotsdamData(val_arr, val_dataset_dir, t_n_cls,
                                  transforms=[sync_rs, sync_totensor], resolution=resolution)
        val_train.append(val_dataset)
        print('Potsdam датасет сформирован')

    train_dataset_dir = Path('/mnt/980EAB530EAB2968/Segmentation_Dataset/coco1024/reconstructed_train')
    val_dataset_dir = Path('/mnt/980EAB530EAB2968/Segmentation_Dataset/coco1024/reconstructed_val')
    if bam:
        print('Датасет BAMFORESTS')
        print('Для обучения')
        if less_opensar:
            train_dataset = BAMFORESTS(train_dataset_dir,
                                        transforms=main_transforms_list, less_data=less_opensar, resolution=resolution)

        else:
            train_dataset = BAMFORESTS(train_dataset_dir,
                                        transforms=main_transforms_list, resolution=resolution)

        data_train.append(train_dataset)

        print('Для тестирования')
        val_dataset = BAMFORESTS(val_dataset_dir,
                                  transforms=[sync_rs, sync_totensor], resolution=resolution, train_mode=False)
        val_train.append(val_dataset)
        print('BAMFORESTS датасет сформирован')

    train_dataset_dir = Path('/mnt/980EAB530EAB2968/Segmentation_Dataset/cvpr_chesapeake_landcover')
    if ches:
        print('Датасет ChesapeakeData')
        print('Для обучения')

        train_dataset = ChesapeakeData(train_dataset_dir,
                                   transforms=main_transforms_list, resolution=resolution)
        data_train.append(train_dataset)
        train_dataset = ChesapeakeData(train_dataset_dir, splits=['test'],
                                       transforms=main_transforms_list, resolution=resolution)
        data_train.append(train_dataset)

        print('Для тестирования')
        val_dataset = ChesapeakeData(train_dataset_dir, splits=['val'],
                                  transforms=[sync_rs, sync_totensor], resolution=resolution)
        val_train.append(val_dataset)
        print('ChesapeakeData датасет сформирован')

#11111111111111111111111111
    train_dataset_dir = Path('/mnt/980EAB530EAB2968/OptoinsDatasets/OptINSDatasetv2t/train')
    val_dataset_dir = Path('/mnt/980EAB530EAB2968/OptoinsDatasets/OptINSDatasetv2/test')
    annotations_path = Path('/mnt/980EAB530EAB2968/OptoinsDatasets/OptINSDatasetv2/annotations.xml')
    test_annotations_path = Path('/mnt/980EAB530EAB2968/OptoinsDatasets/OptINSDatasetv2/test_annotations.xml')

    if image_size is None:
        image_size = resolution

    print('Датасет Radar')
    print('Для обучения')
    if our:
        train_dataset = OptINSInstaSeg(train_dataset_dir, annotations_path, class_names=["Forest", "Water"],
                                      transforms=our_transform_list, resolution=image_size, size_scale=0.65,
                                       skip_first=True, overlap=0.5)

        data_train.append(train_dataset)
        train_dataset = OptINSInstaSeg(train_dataset_dir, annotations_path, class_names=["Forest", "Water"],
                                       transforms=our_transform_list, resolution=image_size, skip_first=True,
                                       overlap=0.5)

        data_train.append(train_dataset)
    print('Для тестирования')

    data_train = ConcatDataset(data_train)
    if chb_mode:
        val_dataset_chb = OptINSInstaSeg(val_dataset_dir, test_annotations_path, class_names=["Forest", "Water"],
                                         transforms=[rgm, sync_rs, sync_totensor], resolution=image_size, overlap=0.0)
        print('Radar датасет сформирован (chb_mode)')
        return data_train, None, val_dataset_chb

    val_dataset = OptINSInstaSeg(val_dataset_dir, test_annotations_path, class_names=["Forest", "Water"],
                                 transforms=[sync_rs, sync_totensor], resolution=image_size, overlap=0.0)
    print('Radar датасет сформирован')

    if len(val_train) > 0:
        val_train = ConcatDataset(val_train)
        return data_train, val_train, val_dataset
    else:
        return data_train, val_dataset


def collate_fn(items):
    images = [item[0][None, ...] for item in items]
    targets = [item[1] for item in items]



    batch_images = torch.cat(images)


    return batch_images, targets


def forest_collate_fn(batch):
    """
    Формирует батч и преобразует 3-канальную маску в 2-канальную,
    объединяя классы 'Вода' и 'Фон'.
    """
    images = []
    targets = []

    for X, y in batch:
        # Проверяем количество каналов у изображения X [C, H, W]
        if X.shape[0] == 1:
            # Если аугментация вернула ч/б (1 канал), дублируем его до 3 каналов
            X = X.repeat(3, 1, 1)
        elif X.shape[0] == 4:
            # На случай, если где-то проскочил альфа-канал (RGBA), отрезаем его
            X = X[:3, :, :]
        images.append(X.unsqueeze(0))

        # На входе y имеет размер [3, H, W]
        # y[0] - Фон
        # y[1] - Лес
        # y[2] - Вода

        # Сливаем канал 0 (фон) и канал 2 (вода).
        # clamp нужен для подстраховки, чтобы значения не превысили 1.0
        new_bg = torch.clamp(y[0] + y[2], 0.0, 1.0).unsqueeze(0)
        new_forest = y[1].unsqueeze(0)

        # Собираем новую маску [2, H, W]
        y_new = torch.cat([new_bg, new_forest], dim=0)
        targets.append(y_new.unsqueeze(0))

    batch_images = torch.cat(images, dim=0)
    batch_targets = torch.cat(targets, dim=0)

    return batch_images, batch_targets