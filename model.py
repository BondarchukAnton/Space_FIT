import logging
from typing import Dict

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import ModelConfig

logger = logging.getLogger(__name__)


def get_model_info(model: nn.Module) -> Dict[str, int]:
    """
    Получение информации о количестве параметров модели.

    Args:
        model: объект нейронной сети.

    Returns:
        Словарь с числом обучаемых, необучаемых и общим числом параметров.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': trainable_params + non_trainable_params
    }


def create_model(model_cfg: ModelConfig) -> nn.Module:
    """
    Создание модели семантической сегментации на основе segmentation_models_pytorch.

    Поддерживаемые архитектуры задаются в конфигурации (DeepLabV3Plus, UnetPlusPlus, Unet).

    Args:
        model_cfg: конфигурация архитектуры модели.

    Returns:
        Необученная (или предобученная на ImageNet) модель nn.Module.
    """
    arch = model_cfg.architecture
    kwargs = {
        'encoder_name': model_cfg.encoder_name,
        'encoder_weights': model_cfg.encoder_weights,
        'in_channels': model_cfg.in_channels,
        'classes': model_cfg.num_classes,
        'activation': model_cfg.activation,
    }
    
    if arch == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(**kwargs)
    elif arch == "UnetPlusPlus":
        model = smp.UnetPlusPlus(**kwargs)
    elif arch == "Unet":
        model = smp.Unet(**kwargs)
    else:
        raise ValueError(f"Неизвестная архитектура: {arch}")

    info = get_model_info(model)
    logger.info(
        f"Создана модель {arch} (энкодер: {model_cfg.encoder_name}). "
        f"Обучаемые параметры: {info['trainable_params']:,} | "
        f"Необучаемые параметры: {info['non_trainable_params']:,}."
    )
    return model


def load_model(checkpoint_path: str, model_cfg: ModelConfig, device: torch.device) -> nn.Module:
    """
    Загрузка модели с весами из указанного файла (чекпоинта).

    Args:
        checkpoint_path: путь к файлу с весами модели.
        model_cfg: конфигурация архитектуры модели.
        device: устройство (CPU или GPU), на которое будет загружена модель.

    Returns:
        Модель nn.Module в режиме оценки (eval).
    """
    model = create_model(model_cfg)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Поддержка чекпоинтов, содержащих полный словарь или только веса
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
