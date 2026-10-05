import os
import logging
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import torch
import segmentation_models_pytorch as smp
from elevation_segmentation.inference import ElevationPredictor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SegmentationInference")

# ---------------------------------------------------------------------------
# Конфигурация по умолчанию
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT_PATH = "/mnt/980EAB530EAB2968/hdd_logs/OtherProject/SPACE/UnetPP_maxvit_6cls_512/top_model.pt"

NUM_TARGET_CLASSES = 6
TARGET_CLASS_NAMES = {
    0: 'Фон',                    # Background (no scene)
    1: 'Лесной массив',          # Forest
    2: 'Поле',                   # Field/Agriculture
    3: 'Водоём',                 # Water
    4: 'Городская территория',   # Urban
    5: 'Горный район',           # Mountain
}

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),         # Фон — чёрный
    1: (0, 255, 0),     # Лесной массив — зелёный
    2: (255, 255, 0),     # Поле — золотой
    3: (0, 0, 255),    # Водоём — синий
    4: (255, 0, 0),     # Городская территория — красный
    5: (255, 0, 255),     # Горный район — фиолетовый
}


# Индекс класса «Горный район» в выходном тензоре основной модели
MOUNTAIN_CLASS_INDEX = 5


class SegmentationModel:
    """
    Автономный класс для выполнения инференса модели сегментации.
    Не требует сторонних модулей проекта и подключения к интернету.

    Опционально интегрирует модель сегментации возвышенностей (ElevationPredictor):
    перед аргмакс её вероятностная карта (float32 H×W) умножается на
    elevation_weight и прибавляется к каналу «Горный район» (индекс
    MOUNTAIN_CLASS_INDEX), корректируя финальную маску в пользу горного класса
    там, где рельеф высокий.
    """

    def __init__(
            self,
            checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
            device: Optional[str] = None,
            num_classes: int = NUM_TARGET_CLASSES,
            tile_size: int = 512,
            overlap: int = 64,
            elevation_checkpoint: Optional[str] = None,
            elevation_weight: float = 1.0,
    ) -> None:
        """
        Инициализация и загрузка весов.

        Args:
            checkpoint_path: Путь к файлу весов (.pt / .pth).
            device: 'cuda', 'cpu' или None (автовыбор).
            num_classes: Количество выходных классов.
            tile_size: Размер окна тайлинга (по умолчанию 512).
            overlap: Перекрытие между соседними окнами.
            elevation_checkpoint: Путь к чекпоинту ElevationPredictor (.pt).
                Если None — коррекция по возвышенностям не применяется.
            elevation_weight: Коэффициент масштабирования вклада elevation-карты
                при добавлении к каналу «Горный район» перед argmax.
                По умолчанию 1.0 (без масштабирования).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.num_classes = num_classes
        self.tile_size = tile_size
        self.overlap = overlap
        self.elevation_weight = elevation_weight

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Файл весов не найден по пути: {checkpoint_path}")

        # Создание архитектуры БЕЗ обращения к интернету (encoder_weights=None)
        self.model = smp.UnetPlusPlus(
            encoder_name='tu-maxvit_base_tf_512',
            encoder_weights=None,  # Исключает попытки скачать imagenet-веса из сети
            in_channels=3,
            classes=self.num_classes,
            activation=None,
        ).to(self.device)

        # Загрузка обученных весов
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

        # Очистка префикса 'module.' если модель сохранялась через DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Статистики нормализации (стандарт ImageNet для maxvit_base_tf_512)
        try:
            params = smp.encoders.get_preprocessing_params('tu-maxvit_base_tf_512')
            self.mean = np.array(params['mean'], dtype=np.float32)
            self.std = np.array(params['std'], dtype=np.float32)
        except Exception:
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        logger.info(f"Модель успешно загружена из {checkpoint_path} на устройство {self.device}")

        # Инициализация elevation-модели (опционально)
        self.elevation_predictor: Optional[ElevationPredictor] = None
        if elevation_checkpoint is not None:
            if not os.path.isfile(elevation_checkpoint):
                raise FileNotFoundError(
                    f"Файл весов elevation-модели не найден: {elevation_checkpoint}"
                )
            self.elevation_predictor = ElevationPredictor(
                checkpoint=elevation_checkpoint,
                device=str(self.device),
            )
            logger.info(
                f"Elevation-модель загружена из {elevation_checkpoint} "
                f"(weight={elevation_weight})"
            )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка: приведение к формату float32, диапазону [0, 1], CHW и нормализация.
        """
        if image.ndim != 3:
            raise ValueError(f"Ожидается трехмерный массив (H, W, C) или (C, H, W), получено: {image.shape}")

        img = image.copy()

        # Приведение к порядку каналов CHW
        if img.shape[2] == 3:
            img = np.transpose(img, (2, 0, 1))
        elif img.shape[0] != 3:
            raise ValueError(f"Ожидается 3 спектральных канала (RGB), получена форма: {img.shape}")

        if img.dtype != np.float32:
            img = img.astype(np.float32)

        if img.max() > 1.0:
            img /= 255.0

        for i in range(3):
            img[i] = (img[i] - self.mean[i]) / self.std[i]

        return img

    @staticmethod
    def _compute_steps(length: int, tile: int, stride: int) -> List[int]:
        """Генерация координат скользящего окна без выхода за границы."""
        if length <= tile:
            return [0]
        steps = list(range(0, length - tile + 1, stride))
        if steps[-1] + tile < length:
            steps.append(length - tile)
        return steps

    def _apply_elevation_correction(self, probs: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
        """
        Прибавляет взвешенную карту вероятностей возвышенностей к каналу
        «Горный район» (MOUNTAIN_CLASS_INDEX) в массиве вероятностей.

        Args:
            probs: Карта вероятностей (num_classes, H, W) float32 после тайлинга.
            image_rgb: Исходное RGB-изображение (H, W, 3) uint8, передаётся в
                ElevationPredictor напрямую (без предобработки основной модели).

        Returns:
            Скорректированный массив вероятностей той же формы и dtype.
        """
        elevation_proba = self.elevation_predictor.predict_proba(image_rgb)  # float32 (H, W)
        probs = probs.copy()
        probs[MOUNTAIN_CLASS_INDEX] += elevation_proba * self.elevation_weight
        return probs

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """
        Предсказание карты вероятностей классов для изображения произвольного размера.

        Если была инициализирована elevation-модель, вероятностная карта
        возвышенностей (умноженная на elevation_weight) прибавляется к каналу
        «Горный район» перед возвратом результата.

        Args:
            image: массив (H, W, 3) или (3, H, W) в RGB.

        Returns:
            Карта вероятностей shape (num_classes, H, W), dtype float32.
        """
        image_proc = self._preprocess(image)
        _, h, w = image_proc.shape

        stride = self.tile_size - self.overlap

        # Паддинг только если изображение меньше минимального размера окна сети
        pad_h = max(0, self.tile_size - h)
        pad_w = max(0, self.tile_size - w)

        if pad_h > 0 or pad_w > 0:
            image_proc = np.pad(image_proc, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

        cur_h, cur_w = image_proc.shape[1], image_proc.shape[2]

        y_steps = self._compute_steps(cur_h, self.tile_size, stride)
        x_steps = self._compute_steps(cur_w, self.tile_size, stride)

        probs = np.zeros((self.num_classes, cur_h, cur_w), dtype=np.float32)
        weight_map = np.zeros((1, cur_h, cur_w), dtype=np.float32)

        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    tile = image_proc[:, y:y + self.tile_size, x:x + self.tile_size]
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(self.device)

                    logits = self.model(tile_tensor)
                    tile_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                    probs[:, y:y + self.tile_size, x:x + self.tile_size] += tile_probs
                    weight_map[:, y:y + self.tile_size, x:x + self.tile_size] += 1.0

        probs /= weight_map
        probs = probs[:, :h, :w]

        # Коррекция по возвышенностям: прибавляем elevation-вероятность к каналу гор
        if self.elevation_predictor is not None:
            # Восстанавливаем исходный RGB uint8 для ElevationPredictor
            image_rgb = image if image.ndim == 3 and image.shape[2] == 3 else np.transpose(image, (1, 2, 0))
            if image_rgb.dtype != np.uint8:
                image_rgb = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)
            probs = self._apply_elevation_correction(probs, image_rgb)

        return probs

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Получение дискретной маски сегментации (индексы классов 0..N-1).

        Args:
            image: массив (H, W, 3) или (3, H, W).

        Returns:
            Маска shape (H, W), dtype uint8.
        """
        probs = self.predict_proba(image)
        return np.argmax(probs, axis=0).astype(np.uint8)

    def predict_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Получение цветной RGB-маски для визуализации.

        Returns:
            RGB-изображение shape (H, W, 3), dtype uint8.
        """
        mask = self.predict(image)
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx, color in CLASS_COLORS.items():
            rgb[mask == cls_idx] = color
        return rgb

    def predict_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Пакетный инференс для списка изображений."""
        return [self.predict(img) for img in images]

    @staticmethod
    def get_class_names() -> Dict[int, str]:
        """Словарь названий классов."""
        return TARGET_CLASS_NAMES


# ---------------------------------------------------------------------------
# Пример вызова в стороннем скрипте
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     # 1. Инициализация модели
#     model = SegmentationModel()
#
#     # 2. Пример прогона тестового массива (например, изображение 1024x1024)
#     dummy_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
#
#     # 3. Получение маски классов (0..5)
#     mask = model.predict(dummy_image)
#     print("Форма маски:", mask.shape, "| Уникальные классы:", np.unique(mask))
#
#     # 4. Получение цветной маски
#     color_mask = model.predict_rgb(dummy_image)
#     print("Форма цветной визуализации:", color_mask.shape)