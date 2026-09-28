import argparse
import logging
import tifffile
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

from datasets.dataset import NUM_TARGET_CLASSES, TARGET_CLASS_NAMES
from model import load_model
from utils import (
    mask_to_rgb, create_legend_patches, compute_iou, setup_logging, TARGET_COLORS
)
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Скрипт визуального тестирования обученной модели сегментации.")
    parser.add_argument('--image', type=str, required=True, help="Путь к тестовому изображению (.tif, 3 канала)")
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

    logger.info(f"Загрузка модели из {args.checkpoint}")
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        architecture='UnetPlusPlus',
        encoder_name='tu-maxvit_base_tf_512',
        in_channels=3,
        num_classes=NUM_TARGET_CLASSES
    )

    logger.info(f"Загрузка изображения {args.image}")
    image = tifffile.imread(args.image)
    if image.ndim == 3 and image.shape[2] == 3:
        image_chw = np.transpose(image, (2, 0, 1))
    else:
        image_chw = image

    if image_chw.dtype != np.float32:
        image_chw = image_chw.astype(np.float32)

    if image_chw.max() > 1.0:
        image_chw /= 255.0

    preprocess_params = smp.encoders.get_preprocessing_params('tu-maxvit_base_tf_512')
    mean = np.array(preprocess_params['mean'], dtype=np.float32).reshape(3, 1, 1)
    std = np.array(preprocess_params['std'], dtype=np.float32).reshape(3, 1, 1)
    image_chw = (image_chw - mean) / std

    logger.info("Выполнение сегментации...")
    with torch.no_grad():
        input_tensor = torch.from_numpy(image_chw).unsqueeze(0).to(device)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_mask = np.argmax(probs, axis=0).astype(np.uint8)

    if image.shape[2] == 3:
        rgb_image = image[:, :, :3].copy()
    else:
        rgb_image = np.transpose(image[:3], (1, 2, 0)).copy()
        
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 1).astype(np.float32)
    else:
        rgb_image = rgb_image.astype(np.float32) / 255.0

    class_names = TARGET_CLASS_NAMES
    colors = TARGET_COLORS
    
    unique_classes = np.unique(pred_mask)
    
    gt_mask = None
    mean_iou = 0.0
    if args.gt_mask:
        logger.info(f"Загрузка ground-truth маски {args.gt_mask}")
        gt_mask = tifffile.imread(args.gt_mask)
        unique_classes = np.unique(np.concatenate([unique_classes, np.unique(gt_mask)]))
        
        iou_per_class, mean_iou = compute_iou(
            torch.from_numpy(pred_mask), 
            torch.from_numpy(gt_mask), 
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
        for i, a in enumerate(axes):
            a.clear()
            a.imshow(rgb_image)
            a.axis('off')
            
            mask_to_show = pred_mask if i == 0 else gt_mask
            title = "Предсказание" if i == 0 else "Эталонная разметка"
            if args.gt_mask and i == 0:
                title += f" (mIoU: {mean_iou:.4f})"
                
            filtered_mask = mask_to_show.copy()
            for cls_idx in class_names.keys():
                if not visible_classes[cls_idx] and cls_idx != 0:
                    filtered_mask[filtered_mask == cls_idx] = 0
                    
            overlay_rgb = mask_to_rgb(filtered_mask, colors)
            overlay_rgb_norm = overlay_rgb.astype(np.float32) / 255.0
            
            mask_alpha = np.where(filtered_mask > 0, args.alpha, 0.0)[..., np.newaxis]
            overlay = rgb_image * (1 - mask_alpha) + overlay_rgb_norm * mask_alpha
            overlay = np.clip(overlay, 0, 1)
            
            a.imshow(overlay)
            a.set_title(title)
            
        plt.draw()

    update_plot()

    ax_check = plt.axes([0.02, 0.1, 0.2, 0.8])
    labels = [f"{k}: {v}" for k, v in class_names.items()]
    visibility = [visible_classes[k] for k in class_names.keys()]
    
    check = CheckButtons(ax_check, labels, visibility)
    
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
