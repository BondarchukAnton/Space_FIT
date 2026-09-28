import numpy as np
import torch
import math
import logging
from typing import List, Dict, Optional, Union

from datasets.dataset import NUM_TARGET_CLASSES, TARGET_CLASS_NAMES
from model import create_model
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


class SegmentationModel:
    """
    Класс для выполнения логического вывода (инференса) модели сегментации.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
    ) -> None:
        """
        Инициализация модели сегментации.
        """
        self.device = torch.device(device)
        self.num_classes = NUM_TARGET_CLASSES

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model = create_model(
            architecture='UnetPlusPlus',
            encoder_name='tu-maxvit_base_tf_512',
            in_channels=3,
            num_classes=self.num_classes
        ).to(self.device)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        self.model.load_state_dict(state_dict)
        
        self.model.eval()

        preprocess_params = smp.encoders.get_preprocessing_params('tu-maxvit_base_tf_512')
        self.mean = np.array(preprocess_params['mean'], dtype=np.float32)
        self.std = np.array(preprocess_params['std'], dtype=np.float32)
        
        logger.info(f"Модель успешно загружена из {checkpoint_path} на {self.device}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения: приведение осей, нормализация.
        """
        if image.ndim != 3:
            raise ValueError(f"Ожидается трехмерный массив, получено {image.ndim} измерений")

        if image.shape[2] == 3:
            image = np.transpose(image, (2, 0, 1))
        elif image.shape[0] != 3:
            raise ValueError(f"Ожидается 3 спектральных каналов, получена форма {image.shape}")

        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if image.max() > 1.0:
            image /= 255.0

        for i in range(3):
            image[i] = (image[i] - self.mean[i]) / self.std[i]

        return image

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """
        Предсказание вероятностей классов для изображения произвольного размера.
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
        
        if padded_h < tile_size or padded_w < tile_size:
            y_steps, x_steps = [0], [0]
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
        """
        probs = self.predict_proba(image)
        mask = np.argmax(probs, axis=0).astype(np.uint8)
        return mask

    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Пакетная обработка списка изображений.
        """
        return [self.predict(img) for img in images]

    def get_class_names(self) -> Dict[int, str]:
        """
        Получение словаря имен классов.
        """
        return TARGET_CLASS_NAMES
