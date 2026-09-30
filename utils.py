"""
Вспомогательные функции проекта сегментации.

Содержит метрики качества (IoU), функции визуализации масок
и утилиты для работы с цветовой палитрой.
"""

import logging
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datasets.dataset import NUM_TARGET_CLASSES, TARGET_CLASS_NAMES

logger = logging.getLogger(__name__)

TARGET_COLORS = {
    0: (0, 0, 0),         # Фон
    1: (34, 139, 34),     # Лесной массив
    2: (255, 215, 0),     # Поле
    3: (30, 144, 255),    # Водоём
    4: (220, 20, 60),     # Городская территория
    5: (139, 90, 43),     # Горный район
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


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Вычисление матрицы ошибок размера (num_classes, num_classes).
    """
    pred = pred.flatten().long()
    target = target.flatten().long()

    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]

    indices = target * num_classes + pred
    cm = torch.bincount(indices, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


def compute_iou_from_cm(cm: torch.Tensor) -> Tuple[Dict[int, float], float]:
    """
    Вычисление IoU по классам из матрицы ошибок.
    """
    intersection = cm.diag()
    union = cm.sum(dim=1) + cm.sum(dim=0) - intersection

    iou_per_class = {}
    valid_ious = []

    for cls_idx in range(cm.shape[0]):
        if union[cls_idx] == 0:
            continue
        iou = (intersection[cls_idx].float() / union[cls_idx].float()).item()
        iou_per_class[cls_idx] = iou
        valid_ious.append(iou)

    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return iou_per_class, mean_iou


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> Tuple[Dict[int, float], float]:
    """
    Вычисление IoU по классам для пары предсказание/истина.
    """
    cm = compute_confusion_matrix(pred, target, num_classes)
    return compute_iou_from_cm(cm)


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def mask_to_rgb(
    mask: np.ndarray,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Преобразование одноканальной маски индексов в RGB-изображение.
    """
    if colors is None:
        colors = TARGET_COLORS

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in colors.items():
        rgb[mask == idx] = color
    return rgb


def create_legend_patches(
    class_indices: Optional[List[int]] = None,
) -> List[mpatches.Patch]:
    """
    Формирование элементов легенды для визуализации маски.
    """
    names = TARGET_CLASS_NAMES
    colors = TARGET_COLORS

    if class_indices is None:
        class_indices = sorted(colors.keys())

    patches = []
    for idx in class_indices:
        if idx not in colors or idx not in names:
            continue
        r, g, b = colors[idx]
        color_norm = (r / 255.0, g / 255.0, b / 255.0)
        patch = mpatches.Patch(color=color_norm, label=f"{idx}: {names[idx]}")
        patches.append(patch)
    return patches


def visualize_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    visible_classes: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (18, 6),
) -> plt.Figure:
    """
    Визуализация результата сегментации.
    """
    # Подготовка изображения: извлечение RGB
    if image.ndim == 3 and image.shape[0] <= image.shape[2]:
        # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
    rgb = image[:, :, :3].copy()

    # Нормализация в [0, 1] для отображения
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 1).astype(np.float32)
    else:
        rgb = rgb.astype(np.float32) / 255.0

    # Фильтрация классов
    filtered_mask = pred_mask.copy()
    if visible_classes is not None:
        for idx in np.unique(filtered_mask):
            if idx not in visible_classes and idx != 0:
                filtered_mask[filtered_mask == idx] = 0

    # Формирование RGB-масок
    pred_rgb = mask_to_rgb(filtered_mask)
    pred_rgb_norm = pred_rgb.astype(np.float32) / 255.0

    num_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=figsize)

    # Исходное изображение
    axes[0].imshow(rgb)
    axes[0].set_title("Исходное изображение (RGB)")
    axes[0].axis("off")

    # Предсказание с наложением
    overlay = rgb * (1 - alpha) + pred_rgb_norm * alpha
    overlay = np.clip(overlay, 0, 1)
    axes[1].imshow(overlay)
    axes[1].set_title("Предсказание сегментации")
    axes[1].axis("off")

    # Легенда
    display_classes = visible_classes if visible_classes else None
    patches = create_legend_patches(display_classes)
    axes[1].legend(
        handles=patches, loc="upper right", fontsize=7, framealpha=0.8,
    )

    # Ground truth
    if gt_mask is not None:
        gt_filtered = gt_mask.copy()
        if visible_classes is not None:
            for idx in np.unique(gt_filtered):
                if idx not in visible_classes and idx != 0:
                    gt_filtered[gt_filtered == idx] = 0
        gt_rgb = mask_to_rgb(gt_filtered)
        gt_rgb_norm = gt_rgb.astype(np.float32) / 255.0
        gt_overlay = rgb * (1 - alpha) + gt_rgb_norm * alpha
        gt_overlay = np.clip(gt_overlay, 0, 1)
        axes[2].imshow(gt_overlay)
        axes[2].set_title("Эталонная разметка")
        axes[2].axis("off")
        axes[2].legend(
            handles=patches, loc="upper right", fontsize=7, framealpha=0.8,
        )

    plt.tight_layout()
    return fig


def setup_logging(level: int = logging.INFO) -> None:
    """
    Настройка формата логирования для всех модулей проекта.
    """
    fmt = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
