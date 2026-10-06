"""Создание модели, нормализация и начальная загрузка совместимых весов."""
import logging
from pathlib import Path
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


def initialize_weights(model, path, same_num_classes=True):
    """Загружает веса модели из файла чекпоинта.

    Args:
        model: Модель нейронной сети для инициализации весов.
        path: Путь к файлу чекпоинта.
        same_num_classes: Флаг совпадения структуры классов. При значении True
            выполняется полная строгая загрузка всех слоев, включая классификатор.
            При значении False загружаются только совместимые общие слои
            (энкодер и декодер), а слой классификации инициализируется заново.

    Raises:
        FileNotFoundError: Если файл чекпоинта не найден.
        ValueError: Если в чекпоинте отсутствует словарь весов, отсутствуют
            совместимые веса энкодера или нарушена структура при строгой загрузке.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Файл чекпоинта не найден: {path}')

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    if not isinstance(state, dict):
        raise ValueError(f'Некорректный формат чекпоинта в {path}: отсутствует state_dict')

    state = {k.removeprefix('module.'): v for k, v in state.items()}
    own = model.state_dict()

    if same_num_classes:
        missing_keys = [k for k in own if k not in state]
        unexpected_keys = [k for k in state if k not in own]
        mismatched_shapes = [
            f'{k} (в чекпоинте: {tuple(state[k].shape)}, в модели: {tuple(own[k].shape)})'
            for k in own if k in state and isinstance(state[k], torch.Tensor) and state[k].shape != own[k].shape
        ]
        if missing_keys or unexpected_keys or mismatched_shapes:
            reasons = []
            if missing_keys:
                reasons.append(f'отсутствующие ключи ({len(missing_keys)}): {missing_keys[:5]}')
            if unexpected_keys:
                reasons.append(f'неожиданные ключи ({len(unexpected_keys)}): {unexpected_keys[:5]}')
            if mismatched_shapes:
                reasons.append(f'несовпадение размеров ({len(mismatched_shapes)}): {mismatched_shapes[:5]}')
            raise ValueError(f'Строгая загрузка модели не удалась для {path}: {"; ".join(reasons)}')

        model.load_state_dict(state, strict=True)
        LOG.info('Полная загрузка модели: успешно загружены все %d тензоров, включая классификатор', len(state))
    else:
        loaded = {}
        skipped = {}
        for key, value in state.items():
            if key.startswith(('segmentation_head.', 'classification_head.')):
                skipped.setdefault('Слой классификатора исключен по настройке same_num_classes=False', []).append(key)
                continue
            if key not in own:
                skipped.setdefault('Ключ отсутствует в целевой архитектуре модели', []).append(key)
                continue
            if not isinstance(value, torch.Tensor):
                skipped.setdefault('Значение не является тензором', []).append(key)
                continue
            if value.shape != own[key].shape:
                skipped.setdefault(
                    f'Несовпадение размера (чекпоинт {tuple(value.shape)} != модель {tuple(own[key].shape)})', []
                ).append(key)
                continue
            loaded[key] = value

        if not any(k.startswith('encoder.') for k in loaded):
            raise ValueError('Нет совместимых весов энкодера: проверьте архитектуру и encoder_name')

        model.load_state_dict(loaded, strict=False)
        LOG.info('Загрузка совместимых слоев: загружено %d/%d тензоров модели; слой классификации инициализирован заново',
                 len(loaded), len(own))
        for reason, keys in skipped.items():
            LOG.info('  Пропущено (%d тензоров): %s. Примеры: %s', len(keys), reason, keys[:3])
