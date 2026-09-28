import logging
from typing import Dict, Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


def get_model_info(model: nn.Module) -> Dict[str, int]:
    """
    Получение информации о количестве параметров модели.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': trainable_params + non_trainable_params
    }


def create_model(
    architecture: str = 'UnetPlusPlus',
    encoder_name: str = 'tu-maxvit_base_tf_512',
    encoder_weights: str = 'imagenet',
    in_channels: int = 3,
    num_classes: int = 7,
    activation: Optional[str] = None,
) -> nn.Module:
    """
    Создание модели семантической сегментации на основе segmentation_models_pytorch.
    """
    kwargs = {
        'encoder_name': encoder_name,
        'encoder_weights': encoder_weights,
        'in_channels': in_channels,
        'classes': num_classes,
        'activation': activation,
    }
    
    if architecture == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(**kwargs)
    elif architecture == "UnetPlusPlus":
        model = smp.UnetPlusPlus(**kwargs)
    elif architecture == "Unet":
        model = smp.Unet(**kwargs)
    else:
        raise ValueError(f"Неизвестная архитектура: {architecture}")

    info = get_model_info(model)
    logger.info(
        f"Создана модель {architecture} (энкодер: {encoder_name}). "
        f"Обучаемые параметры: {info['trainable_params']:,} | "
        f"Необучаемые параметры: {info['non_trainable_params']:,}."
    )
    return model


def load_model(
    checkpoint_path: str,
    device: torch.device,
    architecture: str = 'UnetPlusPlus',
    encoder_name: str = 'tu-maxvit_base_tf_512',
    in_channels: int = 3,
    num_classes: int = 7,
) -> nn.Module:
    """
    Загрузка модели с весами из указанного файла (чекпоинта).
    """
    model = create_model(
        architecture=architecture,
        encoder_name=encoder_name,
        in_channels=in_channels,
        num_classes=num_classes
    )
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Ошибка при загрузке весов из {checkpoint_path}: {e}")
        raise
        
    model.to(device)
    model.eval()
    
    logger.info(f"Веса успешно загружены из {checkpoint_path} на устройство {device}.")
    return model
