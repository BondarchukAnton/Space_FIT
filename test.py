import argparse
import logging
import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

from config import (
    PathConfig, ModelConfig, get_class_names, get_selected_colors,
    CHANNEL_MEAN, CHANNEL_STD, SELECTED_CLASSES
)
from model import load_model, create_model
from utils import (
    remap_mask, mask_to_rgb, create_legend_patches, compute_iou, setup_logging
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Скрипт визуального тестирования обученной модели сегментации.")
    parser.add_argument('--image', type=str, required=True, help="Путь к тестовому изображению (.tif, 5 каналов)")
    parser.add_argument('--checkpoint', type=str, default="checkpoints/best_model.pth", help="Путь к чекпоинту обученной модели")
    parser.add_argument('--gt-mask', type=str, default=None, help="Путь к ground-truth маске (опционально)")
    parser.add_argument('--alpha', type=float, default=0.5, help="Прозрачность наложения маски")
    parser.add_argument('--save', type=str, default=None, help="Путь для сохранения результата вместо интерактивного отображения")
    parser.add_argument('--device', type=str, default="cuda", help="Устройство (cuda/cpu)")
    return parser.parse_args()


def main() -> None:
    """Основная функция для тестирования и визуализации."""
    setup_logging()
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используемое устройство: {device}")

    # Загрузка чекпоинта и модели
    logger.info(f"Загрузка модели из {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model_config_dict = checkpoint.get('model_config', {})
    if isinstance(model_config_dict, ModelConfig):
        model_config = model_config_dict
    else:
        model_config = ModelConfig(**model_config_dict) if model_config_dict else ModelConfig()
        
    model = create_model(model_config)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Чтение изображения
    logger.info(f"Загрузка изображения {args.image}")
    image = tifffile.imread(args.image)
    if image.shape[2] == 5:
        image_chw = np.transpose(image, (2, 0, 1))
    else:
        image_chw = image

    if image_chw.dtype != np.float32:
        image_chw = image_chw.astype(np.float32)

    if image_chw.max() > 1.0:
        image_chw /= 255.0

    mean = np.array(CHANNEL_MEAN, dtype=np.float32).reshape(5, 1, 1)
    std = np.array(CHANNEL_STD, dtype=np.float32).reshape(5, 1, 1)
    image_chw = (image_chw - mean) / std

    # Предсказание маски
    logger.info("Выполнение сегментации...")
    with torch.no_grad():
        input_tensor = torch.from_numpy(image_chw).unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_mask = np.argmax(probs, axis=0).astype(np.uint8)

    # Подготовка RGB для фона
    if image.shape[2] == 5:
        rgb_image = image[:, :, :3].copy()
    else:
        rgb_image = np.transpose(image[:3], (1, 2, 0)).copy()
        
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 1).astype(np.float32)
    else:
        rgb_image = rgb_image.astype(np.float32) / 255.0

    class_names = get_class_names()
    colors = get_selected_colors()
    
    unique_classes = np.unique(pred_mask)
    
    gt_mask = None
    gt_remapped = None
    mean_iou = 0.0
    if args.gt_mask:
        logger.info(f"Загрузка ground-truth маски {args.gt_mask}")
        gt_mask = tifffile.imread(args.gt_mask)
        gt_remapped = remap_mask(gt_mask)
        unique_classes = np.unique(np.concatenate([unique_classes, np.unique(gt_remapped)]))
        
        iou_per_class, mean_iou = compute_iou(
            torch.from_numpy(pred_mask), 
            torch.from_numpy(gt_remapped), 
            len(class_names)
        )
        logger.info(f"Среднее IoU: {mean_iou:.4f}")
        for cls_idx, iou in iou_per_class.items():
            logger.info(f"  {class_names[cls_idx]}: {iou:.4f}")

    visible_classes = {cls_idx: True for cls_idx in class_names.keys()}

    fig, ax = plt.subplots(1, 2 if gt_mask is not None else 1, figsize=(15, 8))
    plt.subplots_adjust(left=0.25, bottom=0.1)

    axes = [ax] if not isinstance(ax, np.ndarray) else ax.flatten()

    def update_plot() -> None:
        """Обновление отображения при переключении классов."""
        for i, a in enumerate(axes):
            a.clear()
            a.imshow(rgb_image)
            a.axis('off')
            
            mask_to_show = pred_mask if i == 0 else gt_remapped
            title = "Предсказание" if i == 0 else "Эталонная разметка"
            if args.gt_mask and i == 0:
                title += f" (mIoU: {mean_iou:.4f})"
                
            filtered_mask = mask_to_show.copy()
            for cls_idx in class_names.keys():
                if not visible_classes[cls_idx] and cls_idx != 0:
                    filtered_mask[filtered_mask == cls_idx] = 0
                    
            overlay_rgb = mask_to_rgb(filtered_mask, colors)
            overlay_rgb_norm = overlay_rgb.astype(np.float32) / 255.0
            
            # Фоновые пиксели маски оставляем прозрачными
            mask_alpha = np.where(filtered_mask > 0, args.alpha, 0.0)[..., np.newaxis]
            overlay = rgb_image * (1 - mask_alpha) + overlay_rgb_norm * mask_alpha
            overlay = np.clip(overlay, 0, 1)
            
            a.imshow(overlay)
            a.set_title(title)
            
        plt.draw()

    update_plot()

    # Настройка интерактивных чекбоксов для фильтрации классов
    ax_check = plt.axes([0.02, 0.1, 0.2, 0.8])
    labels = [f"{k}: {v}" for k, v in class_names.items()]
    visibility = [visible_classes[k] for k in class_names.keys()]
    
    check = CheckButtons(ax_check, labels, visibility)
    
    # Раскраска чекбоксов цветами классов
    for i, p in enumerate(check.rectangles):
        idx = list(class_names.keys())[i]
        c = colors.get(idx, (0, 0, 0))
        p.set_facecolor(tuple(c_val / 255.0 for c_val in c))
        p.set_edgecolor('black')
        p.set_alpha(0.8)

    def on_check(label: str) -> None:
        idx = int(label.split(':')[0])
        visible_classes[idx] = not visible_classes[idx]
        update_plot()

    check.on_clicked(on_check)

    if args.save:
        plt.savefig(args.save, bbox_inches='tight')
        logger.info(f"Результат визуализации сохранен в {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
