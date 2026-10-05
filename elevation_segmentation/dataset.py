"""Чтение пар RGB/маска и воспроизводимые вырезки без изменения GSD."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset


def read_rgb(path):
    """Возвращает RGB uint8 без повторной радиометрической калибровки."""
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def read_mask(path):
    """Сохраняет исходные индексы маски, включая 255."""
    with Image.open(path) as image:
        mask = np.array(image)
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError(f'Неверная индексная маска: {path}')
    return mask.astype(np.uint8)


def load_records(root):
    """Читает manifest или пары из train/val; проверяет разделение локаций."""
    root = Path(root).resolve()
    manifest = root / 'manifest.jsonl'
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        rows = [{'split': split, 'name': p.stem, 'image': str(p.relative_to(root)),
                 'mask': f'{split}/masks/{p.name}', 'location_id': p.stem.split('__')[0]}
                for split in ('train', 'val') for p in sorted((root/split/'images').glob('*.png'))]
    result = {'train': [], 'val': []}; seen = set()
    for row in rows:
        split = row['split']
        if split not in result:
            continue
        row = dict(row)
        for key in ('image', 'mask'):
            path = (root / row[key]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'Недоступный файл: {path}')
            row[key] = str(path)
        if row['image'] in seen:
            raise ValueError(f'Повтор изображения в manifest: {row["image"]}')
        seen.add(row['image'])
        row.setdefault('name', Path(row['image']).stem)
        row.setdefault('location_id', row['name'].split('__')[0])
        result[split].append(row)
    if not all(result.values()):
        raise ValueError('Нужны непустые train и val')
    shared = {r['location_id'] for r in result['train']} & {r['location_id'] for r in result['val']}
    if shared:
        raise ValueError(f'Общие location_id между train и val: {sorted(shared)[:5]}')
    return result


def inspect_dataset(records):
    """Проверяет пары и вычисляет отпечаток содержимого для восстановления обучения."""
    digest = hashlib.sha256(); report = {}; identities = {}
    for split, rows in records.items():
        counts = {str(k): 0 for k in (0, 1, 255)}; shapes = set()
        for row in rows:
            image = read_rgb(row['image']); mask = read_mask(row['mask'])
            if image.shape[:2] != mask.shape:
                raise ValueError(f'Размеры не совпадают: {row["name"]}')
            if not (mask != 255).any():
                raise ValueError(f'Вся маска исключена: {row["name"]}')
            shapes.add(tuple(mask.shape))
            for key in counts:
                counts[key] += int((mask == int(key)).sum())
            rgb_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if rgb_hash in identities and identities[rgb_hash] != split:
                raise ValueError('Одинаковое RGB-изображение обнаружено в train и val')
            identities[rgb_hash] = split
            digest.update(json.dumps([split, row['name'], row['location_id'], image.shape]).encode())
            digest.update(image.tobytes()); digest.update(mask.tobytes())
        report[split] = {'images': len(rows), 'locations': len({r['location_id'] for r in rows}),
                         'shapes_hw': sorted(shapes), 'pixels': counts,
                         'positive_fraction': counts['1']/(counts['0']+counts['1'])}
    return report, digest.hexdigest()


class ElevationDataset(Dataset):
    """Создаёт квадратные вырезки с синхронными преобразованиями RGB и маски."""
    def __init__(self, records, config):
        self.records = records; self.config = config; self.epoch = 0

    def __len__(self):
        return len(self.records) * self.config['crops_per_image']

    def __getitem__(self, index):
        cfg = self.config; row = self.records[index % len(self.records)]
        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], self.epoch, index]))
        image = read_rgb(row['image']); mask = read_mask(row['mask']); size = cfg['crop_size']
        h, w = mask.shape
        dh, dw = max(0, size-h), max(0, size-w)
        padding = ((dh//2, dh-dh//2), (dw//2, dw-dw//2))
        image = np.pad(image, (*padding, (0,0)), mode='edge')
        mask = np.pad(mask, padding, mode='constant', constant_values=255)
        h,w = mask.shape
        points = np.argwhere(mask == 1) if rng.random() < cfg['positive_crop_probability'] else []
        selected = None
        for attempt in range(12):
            if len(points):
                py,px = points[rng.integers(len(points))]
                y = int(rng.integers(max(0,py-size+1), min(py,h-size)+1))
                x = int(rng.integers(max(0,px-size+1), min(px,w-size)+1))
            else:
                y = int(rng.integers(h-size+1)); x = int(rng.integers(w-size+1))
            if (mask[y:y+size,x:x+size] != 255).any():
                selected = (y,x); break
        if selected is None:
            py,px = np.argwhere(mask != 255)[0]
            selected = (min(max(int(py)-size//2,0),h-size), min(max(int(px)-size//2,0),w-size))
        y,x = selected
        image = image[y:y+size,x:x+size]; mask = mask[y:y+size,x:x+size]
        rotation = int(rng.integers(4)); image = np.rot90(image,rotation); mask = np.rot90(mask,rotation)
        if rng.random() < .5:
            image = np.fliplr(image); mask = np.fliplr(mask)
        pil = Image.fromarray(np.ascontiguousarray(image))
        pil = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(.9,1.1)))
        pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(.9,1.1)))
        image = np.array(pil,dtype=np.float32)/255.
        return torch.from_numpy(image.transpose(2,0,1).copy()), torch.from_numpy(mask.copy()).long()
