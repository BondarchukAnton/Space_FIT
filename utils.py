"""
Вспомогательные функции проекта сегментации.

Содержит метрики качества (IoU), функции визуализации масок
и утилиты для работы с цветовой палитрой.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    get_class_names,
    get_class_remap,
    get_selected_colors,
    SELECTED_CLASSES,
)

logger = logging.getLogger(__name__)


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

    Args:
        pred: предсказанные индексы классов, shape (N,) или (H, W).
        target: истинные индексы классов, shape (N,) или (H, W).
        num_classes: общее число классов.

    Returns:
        Матрица ошибок (num_classes, num_classes), dtype int64.
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

    Args:
        cm: матрица ошибок (num_classes, num_classes).

    Returns:
        Кортеж (iou_per_class, mean_iou):
            - iou_per_class: словарь {индекс класса: IoU}.
            - mean_iou: среднее IoU по классам с ненулевой площадью.
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

    Args:
        pred: предсказанные индексы, shape произвольный.
        target: истинные индексы, shape произвольный.
        num_classes: общее число классов.

    Returns:
        Кортеж (iou_per_class, mean_iou).
    """
    cm = compute_confusion_matrix(pred, target, num_classes)
    return compute_iou_from_cm(cm)


# ---------------------------------------------------------------------------
# Ремаппинг масок
# ---------------------------------------------------------------------------
_LUT = np.zeros(256, dtype=np.int64)
for _orig_val, _target_idx in get_class_remap().items():
    _LUT[_orig_val] = _target_idx


def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Быстрый попиксельный ремаппинг через Look-Up Table (LUT).

    Args:
        mask: одноканальная маска, dtype uint8.

    Returns:
        Маска с непрерывными индексами классов (0..5), dtype int64.
    """
    return _LUT[mask]
# def remap_mask(mask: np.ndarray) -> np.ndarray:
#     """
#     Ремаппинг пиксельных значений маски в непрерывные индексы.
#
#     Пиксели выбранных классов получают индексы 1..N.
#     Все прочие пиксели (включая фон, значение 0) -> индекс 0.
#
#     Args:
#         mask: одноканальная маска, dtype uint8.
#
#     Returns:
#         Маска с непрерывными индексами классов, dtype int64.
#     """
#     remap = get_class_remap()
#     remapped = np.zeros_like(mask, dtype=np.int64)
#     for pixel_val, idx in remap.items():
#         remapped[mask == pixel_val] = idx
#     return remapped


# def inverse_remap_mask(mask: np.ndarray) -> np.ndarray:
#     """
#     Обратный ремаппинг: непрерывные индексы -> пиксельные значения.
#
#     Args:
#         mask: маска с непрерывными индексами, dtype int.
#
#     Returns:
#         Маска с пиксельными значениями датасета, dtype uint8.
#     """
#     from config import get_inverse_remap
#     inv_remap = get_inverse_remap()
#     result = np.zeros_like(mask, dtype=np.uint8)
#     for idx, pixel_val in inv_remap.items():
#         result[mask == idx] = pixel_val
#     return result


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def mask_to_rgb(
    mask: np.ndarray,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Преобразование одноканальной маски индексов в RGB-изображение.

    Args:
        mask: маска с непрерывными индексами классов, shape (H, W).
        colors: словарь {индекс: (R, G, B)}. По умолчанию — палитра
                из конфигурации.

    Returns:
        RGB-изображение, shape (H, W, 3), dtype uint8.
    """
    if colors is None:
        colors = get_selected_colors()

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

    Args:
        class_indices: список индексов классов для отображения.
                       По умолчанию — все выбранные классы + фон.

    Returns:
        Список matplotlib.patches.Patch для добавления в легенду.
    """
    names = get_class_names()
    colors = get_selected_colors()

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

    Отображает исходное изображение, маску предсказания
    и (при наличии) маску ground truth.

    Args:
        image: исходное изображение, shape (H, W, C) или (C, H, W).
               Отображаются первые 3 канала (RGB).
        pred_mask: предсказанная маска, shape (H, W), непрерывные индексы.
        gt_mask: маска ground truth (опционально), shape (H, W).
        alpha: прозрачность наложения маски.
        visible_classes: список классов для отображения. Если None — все.
        figsize: размер фигуры matplotlib.

    Returns:
        Объект matplotlib.Figure.
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

    Формат: [ГГГГ-ММ-ДД ЧЧ:ММ:СС] УРОВЕНЬ имя_модуля — сообщение.
    """
    fmt = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
