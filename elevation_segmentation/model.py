"""Создание модели, нормализация и начальная загрузка совместимых весов."""
import logging
import torch
from torch import nn
import segmentation_models_pytorch as smp

LOG = logging.getLogger(__name__)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def create_model(config, pretrained=True):
    """Создаёт SMP-модель с одним каналом логитов; активация применяется отдельно."""
    architecture = config['architecture']
    if architecture not in ('DeepLabV3Plus', 'Unet', 'UnetPlusPlus'):
        raise ValueError(f'Неизвестная архитектура: {architecture}')
    return getattr(smp,architecture)(encoder_name=config['encoder_name'],
            encoder_weights=config['encoder_weights'] if pretrained else None,
            in_channels=3, classes=1, activation=None)


def normalize(images, mean=MEAN, std=STD):
    """Нормализует готовые RGB 0..1 параметрами ImageNet."""
    mean = images.new_tensor(mean)[None,:,None,None]
    std = images.new_tensor(std)[None,:,None,None]
    return (images-mean)/std


def freeze_batchnorm(model):
    """Фиксирует статистики BatchNorm при малом батче, сохраняя обучение аффинных параметров."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def initialize_weights(model, path):
    """Загружает совместимые общие слои; всегда исключает старую голову классов."""
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    state = checkpoint.get('model_state_dict',checkpoint.get('state_dict',checkpoint))
    own = model.state_dict(); loaded = {}
    for key,value in state.items():
        key = key.removeprefix('module.')
        if key.startswith(('segmentation_head.','classification_head.')):
            continue
        if key in own and isinstance(value,torch.Tensor) and value.shape == own[key].shape:
            loaded[key] = value
    if not any(k.startswith('encoder.') for k in loaded):
        raise ValueError('Нет совместимых весов энкодера: проверьте архитектуру и encoder_name')
    model.load_state_dict(loaded,strict=False)
    LOG.info('Начальная загрузка: %d/%d тензоров; голова сегментации новая',len(loaded),len(own))
