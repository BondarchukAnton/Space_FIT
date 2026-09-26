import numpy as np
import torch
import math
import logging
from typing import List, Dict, Optional, Union

from config import (
    ModelConfig, CHANNEL_MEAN, CHANNEL_STD, SELECTED_CLASSES,
    get_class_names, get_class_names_en, get_class_remap
)
from model import create_model

logger = logging.getLogger(__name__)


class SegmentationModel:
    """
    Класс для выполнения логического вывода (инференса) модели сегментации.

    Обеспечивает загрузку весов, предобработку входных изображений (включая
    тайлинг для изображений произвольного размера), выполнение предсказания
    и объединение результатов.

    Формат входных данных: изображения в виде np.ndarray, форма (H, W, 5)
    или (5, H, W).
    Формат выходных данных: маски в виде np.ndarray, форма (H, W),
    dtype uint8 (непрерывные индексы классов).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        selected_classes: Optional[List[int]] = None
    ) -> None:
        """
        Инициализация модели сегментации.

        Args:
            checkpoint_path: Путь к файлу чекпоинта (.pth).
            device: Устройство для вычислений ('cuda' или 'cpu').
            selected_classes: Список выбранных классов. Если None,
                              используется значение из конфигурации.
        """
        self.device = torch.device(device)
        self.selected_classes = selected_classes if selected_classes is not None else SELECTED_CLASSES
        self.num_classes = len(self.selected_classes) + 1

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Если в чекпоинте сохранен конфиг, можно его извлечь
        model_config_dict = checkpoint.get('model_config', {})
        if isinstance(model_config_dict, ModelConfig):
            self.model_config = model_config_dict
        else:
            self.model_config = ModelConfig(**model_config_dict) if model_config_dict else ModelConfig()

        self.model = create_model(self.model_config).to(self.device)
        
        # Поддержка разных форматов сохранения чекпоинта
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        self.model.load_state_dict(state_dict)
        
        self.model.eval()

        self.mean = np.array(CHANNEL_MEAN, dtype=np.float32)
        self.std = np.array(CHANNEL_STD, dtype=np.float32)
        
        logger.info(f"Модель успешно загружена из {checkpoint_path} на {self.device}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения: приведение осей, нормализация.

        Args:
            image: Входное изображение (H, W, 5) или (5, H, W).

        Returns:
            Нормализованное изображение формы (5, H, W), dtype float32.
        """
        if image.ndim != 3:
            raise ValueError(f"Ожидается трехмерный массив, получено {image.ndim} измерений")

        # Приведение к формату (C, H, W)
        if image.shape[2] == 5:
            image = np.transpose(image, (2, 0, 1))
        elif image.shape[0] != 5:
            raise ValueError(f"Ожидается 5 спектральных каналов, получена форма {image.shape}")

        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Нормализация
        if image.max() > 1.0:
            image /= 255.0

        for i in range(5):
            image[i] = (image[i] - self.mean[i]) / self.std[i]

        return image

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятностей классов для изображения произвольного размера.

        Изображение нарезается на тайлы 512x512 с перекрытием 64 пикселя.
        Результаты усредняются в зонах перекрытия.

        Args:
            image: Входное изображение (H, W, 5) или (5, H, W).

        Returns:
            Массив вероятностей формы (num_classes, H, W), dtype float32.
        """
        image_proc = self._preprocess(image)
        c, h, w = image_proc.shape
        
        tile_size = 512
        overlap = 64
        stride = tile_size - overlap

        pad_h = (tile_size - h % tile_size) % tile_size
        pad_w = (tile_size - w % tile_size) % tile_size
        
        if pad_h > 0 or pad_w > 0:
            image_proc = np.pad(image_proc, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            
        padded_h, padded_w = image_proc.shape[1:]
        
        probs = np.zeros((self.num_classes, padded_h, padded_w), dtype=np.float32)
        weight_map = np.zeros((1, padded_h, padded_w), dtype=np.float32)
        
        y_steps = range(0, padded_h - tile_size + 1, stride) if padded_h >= tile_size else [0]
        x_steps = range(0, padded_w - tile_size + 1, stride) if padded_w >= tile_size else [0]
        
        # Если размер меньше размера тайла (что редкость, но возможно)
        if padded_h < tile_size or padded_w < tile_size:
            y_steps, x_steps = [0], [0]
            # Дополнительный паддинг до размера тайла
            target_h = max(tile_size, padded_h)
            target_w = max(tile_size, padded_w)
            image_proc = np.pad(image_proc, ((0, 0), (0, target_h - padded_h), (0, target_w - padded_w)), mode='reflect')
            probs = np.zeros((self.num_classes, target_h, target_w), dtype=np.float32)
            weight_map = np.zeros((1, target_h, target_w), dtype=np.float32)

        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    tile = image_proc[:, y:y+tile_size, x:x+tile_size]
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(self.device)
                    
                    logits = self.model(tile_tensor)
                    tile_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    
                    probs[:, y:y+tile_size, x:x+tile_size] += tile_probs
                    weight_map[:, y:y+tile_size, x:x+tile_size] += 1.0

        probs /= weight_map
        probs = probs[:, :h, :w]
        
        return probs

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Предсказание маски сегментации для изображения.

        Args:
            image: Входное изображение (H, W, 5) или (5, H, W).

        Returns:
            Маска индексов классов формы (H, W), dtype uint8.
        """
        probs = self.predict_proba(image)
        mask = np.argmax(probs, axis=0).astype(np.uint8)
        return mask

    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Пакетная обработка списка изображений.

        Args:
            images: Список изображений.

        Returns:
            Список предсказанных масок.
        """
        return [self.predict(img) for img in images]

    def get_class_names(self) -> Dict[int, str]:
        """
        Получение словаря имен классов на русском языке.

        Returns:
            Словарь вида {индекс_класса: "Название RU"}.
        """
        return get_class_names()

    def get_class_names_en(self) -> Dict[int, str]:
        """
        Получение словаря имен классов на английском языке.

        Returns:
            Словарь вида {индекс_класса: "Название EN"}.
        """
        return get_class_names_en()
