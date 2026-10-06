"""Обучение RGB-модели с проверкой на полных изображениях валидационной выборки."""
import argparse
import json
import logging
import math
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Обеспечиваем корректный импорт при прямом запуске скрипта из PyCharm
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

if __package__ is None or __package__ == "":
    from elevation_segmentation.dataset import load_records, inspect_dataset, ElevationDataset, read_rgb, read_mask
    from elevation_segmentation.model import create_model, normalize, freeze_batchnorm, initialize_weights
    from elevation_segmentation.losses import ElevationLoss
    from elevation_segmentation.inference import predict_probability, choose_device
    from elevation_segmentation.utils import setup_logging, atomic_save, confusion, scores, write_json
else:
    from .dataset import load_records, inspect_dataset, ElevationDataset, read_rgb, read_mask
    from .model import create_model, normalize, freeze_batchnorm, initialize_weights
    from .losses import ElevationLoss
    from .inference import predict_probability, choose_device
    from .utils import setup_logging, atomic_save, confusion, scores, write_json

LOG = logging.getLogger(__name__)


def resolve_relative_path(path_str, base_dir):
    """Преобразует относительный путь в абсолютный относительно корня проекта.

    Args:
        path_str: Строка пути или None.
        base_dir: Базовая директория (корень проекта).

    Returns:
        Абсолютный путь Path или None.
    """
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p).resolve()


def validate_config(cfg):
    """Проверяет параметры конфигурации, влияющие на размеры тензоров и цикл обучения.

    Args:
        cfg: Словарь настроек конфигурации.

    Raises:
        ValueError: При наличии некорректных значений параметров.
    """
    for key in ('normalization_mean', 'normalization_std'):
        if len(cfg[key]) != 3 or not all(math.isfinite(x) for x in cfg[key]):
            raise ValueError(f'Неверный {key}')
    if any(x <= 0 for x in cfg['normalization_std']):
        raise ValueError('Параметры std должны быть строго положительными')
    for key in ('crop_size', 'tile_size'):
        if cfg[key] < 32 or cfg[key] % 32 != 0:
            raise ValueError(f'{key}: требуется положительное число, кратное 32')
    for key in ('batch_size', 'accumulation_steps', 'epochs', 'crops_per_image', 'save_every_steps'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'Неверное значение {key}: ожидается положительное целое число')
    if not (0 <= cfg['overlap'] < cfg['tile_size']):
        raise ValueError('Неверное перекрытие окон overlap: должно быть неотрицательным и меньше tile_size')
    if not (0 <= cfg['positive_crop_probability'] <= 1) or not (0 <= cfg['dice_weight'] <= 1):
        raise ValueError('Доли вероятностей должны находиться в диапазоне [0, 1]')
    if not (0 < cfg['threshold'] < 1) or cfg['pos_weight'] <= 0:
        raise ValueError('Неверный порог классификации или вес положительного класса')
    if cfg['num_workers'] < 0:
        raise ValueError('Параметр num_workers должен быть неотрицательным')
    for key in ('encoder_lr', 'decoder_lr', 'weight_decay'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'Неверное значение {key}')
    if cfg['batch_size'] == 1 and not cfg['freeze_batchnorm']:
        raise ValueError('При batch_size=1 используйте freeze_batchnorm=true')

    # Проверка параметров загрузки модели
    if cfg.get('resume_training', False):
        if not cfg.get('pretrained_path'):
            raise ValueError('Ошибочная конфигурация: resume_training=true требует указания pretrained_path')
        if not cfg.get('same_num_classes', True):
            raise ValueError('Ошибочная конфигурация: resume_training=true невозможно при same_num_classes=false')

    # Проверка параметров расписания скорости обучения
    scheduler_type = cfg.get('scheduler', 'onecycle')
    if scheduler_type not in ('onecycle', 'none'):
        raise ValueError(f'Неподдерживаемый тип планировщика: {scheduler_type}. Допустимы "onecycle" или "none"')
    if scheduler_type == 'onecycle':
        for key in ('encoder_max_lr', 'decoder_max_lr', 'onecycle_div_factor', 'onecycle_final_div_factor'):
            if key not in cfg or not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f'Неверный параметр расписания OneCycleLR: {key}')
        if not (0 < cfg.get('onecycle_pct_start', 0.1) < 1):
            raise ValueError('Параметр onecycle_pct_start должен находиться в интервале (0, 1)')


def capture_rng():
    """Сохраняет состояния генераторов случайных чисел."""
    return {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'python': random.getstate(),
    }


def restore_rng(state):
    """Восстанавливает сохраненные состояния генераторов случайных чисел."""
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def compatible_config(old, new):
    """Проверяет совместимость конфигурации при возобновлении обучения."""
    ignored = {
        'dataset_root', 'output_dir', 'device', 'num_workers',
        'pretrained_path', 'resume_training', 'same_num_classes'
    }
    changed = [key for key in set(old) | set(new) if key not in ignored and old.get(key) != new.get(key)]
    if changed:
        raise ValueError(f'Для resume изменены параметры {changed}. Для нового обучения используйте resume_training=false.')


def evaluate(model, records, cfg, device, output, epoch):
    """Вычисляет метрики валидационной выборки после объединения скользящих окон."""
    from PIL import Image
    matrix = np.zeros((2, 2), np.int64)
    groups = {}
    per_image = []
    preview = output / 'previews'
    preview.mkdir(exist_ok=True)
    for i, row in enumerate(tqdm(records, desc='val', mininterval=2)):
        rgb = read_rgb(row['image'])
        mask = read_mask(row['mask'])
        probability = predict_probability(
            model, rgb, device, cfg['tile_size'], cfg['overlap'],
            cfg['amp'], cfg['normalization_mean'], cfg['normalization_std']
        )
        cm = confusion(probability, mask, cfg['threshold'])
        matrix += cm
        group = row.get('group', 'unknown')
        groups.setdefault(group, np.zeros((2, 2), np.int64))
        groups[group] += cm
        per_image.append({'name': row['name'], 'group': group, **scores(cm)})
        if i < 3:
            overlay = rgb.copy()
            hit = probability >= cfg['threshold']
            overlay[hit] = (.55 * overlay[hit] + .45 * np.array([255, 40, 20])).astype(np.uint8)
            gt = np.zeros_like(rgb)
            gt[mask == 1] = [255, 255, 255]
            gt[mask == 255] = [180, 0, 180]
            canvas = np.concatenate([rgb, gt, overlay], axis=1)
            image = Image.fromarray(canvas)
            image.thumbnail((1440, 480))
            image.save(preview / f'val_{i:02d}.jpg')
    result = scores(matrix)
    result['groups'] = {g: scores(m) for g, m in groups.items()}
    write_json(output / 'val_metrics.json', {'epoch': epoch + 1, **result, 'images': per_image})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).resolve().parent / 'config.json'
    parser.add_argument('--config', default=str(default_config), help='Путь к конфигурационному файлу')
    parser.add_argument('--data', help='Переопределение корня датасета (dataset_root)')
    parser.add_argument('--output', help='Переопределение папки результатов (output_dir)')
    parser.add_argument('--device', help='Переопределение вычислительного устройства')
    parser.add_argument('--pretrained-path', help='Переопределение пути к чекпоинту предобученной модели')
    parser.add_argument('--same-num-classes', type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                        help='Флаг одинакового числа классов (true/false)')
    parser.add_argument('--resume-training', action='store_true',
                        help='Флаг продолжения обучения (resume)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', help='Устаревший флаг: путь к чекпоинту для продолжения обучения')
    group.add_argument('--init-from', help='Устаревший флаг: путь к чекпоинту для переноса весов с новой головой')
    parser.add_argument('--check-data', action='store_true', help='Только проверить целостность данных')
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Конфигурационный файл не найден: {config_path}')
    cfg = json.loads(config_path.read_text(encoding='utf-8'))

    # Применение аргументов командной строки
    if args.data:
        cfg['dataset_root'] = args.data
    if args.output:
        cfg['output_dir'] = args.output
    if args.device:
        cfg['device'] = args.device
    if args.pretrained_path:
        cfg['pretrained_path'] = args.pretrained_path
    if args.same_num_classes is not None:
        cfg['same_num_classes'] = args.same_num_classes
    if args.resume_training:
        cfg['resume_training'] = True

    # Совместимость со старыми аргументами
    if args.resume:
        cfg['pretrained_path'] = args.resume
        cfg['resume_training'] = True
        cfg['same_num_classes'] = True
    elif args.init_from:
        cfg['pretrained_path'] = args.init_from
        cfg['resume_training'] = False
        cfg['same_num_classes'] = False

    # Разрешение относительных путей относительно корня проекта Space_FIT
    project_root = Path(__file__).resolve().parents[1]
    cfg['dataset_root'] = str(resolve_relative_path(cfg['dataset_root'], project_root))
    cfg['output_dir'] = str(resolve_relative_path(cfg['output_dir'], project_root))
    if cfg.get('pretrained_path') is not None:
        cfg['pretrained_path'] = str(resolve_relative_path(cfg['pretrained_path'], project_root))

    validate_config(cfg)
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(output)

    # Определение режима работы для информационного вывода
    pretrained_path = cfg.get('pretrained_path')
    resume_training = cfg.get('resume_training', False)
    same_num_classes = cfg.get('same_num_classes', True)

    if resume_training:
        mode_name = 'Продолжение обучения (resume)'
        mode_desc = 'Восстановление весов модели, оптимизатора, планировщика, скейлера и номера эпохи'
    elif pretrained_path:
        if same_num_classes:
            mode_name = 'Новое обучение с полной загрузкой модели'
            mode_desc = 'Загрузка всех слоев, включая классификатор; новое обучение с 1-й эпохи с новым оптимизатором'
        else:
            mode_name = 'Новое обучение с загрузкой общих слоев'
            mode_desc = 'Загрузка совместимых слоев энкодера/декодера с новой головой; новое обучение с 1-й эпохи'
    else:
        mode_name = 'Новое обучение с нуля'
        mode_desc = f'Инициализация энкодера весами {cfg.get("encoder_weights")}; новое обучение с 1-й эпохи'

    LOG.info('================================================================================')
    LOG.info('Выбранный режим: %s', mode_name)
    LOG.info('Описание режима: %s', mode_desc)
    LOG.info('Абсолютный путь к конфигу: %s', config_path)
    LOG.info('Абсолютный путь к датасету: %s', cfg['dataset_root'])
    LOG.info('Абсолютный путь к чекпоинту: %s', cfg['pretrained_path'] if cfg['pretrained_path'] else 'Не используется (None)')
    LOG.info('Абсолютный путь к папке результатов: %s', output)
    LOG.info('================================================================================')

    records = load_records(cfg['dataset_root'])
    report, fingerprint = inspect_dataset(records)
    write_json(output / 'dataset_report.json', report)
    LOG.info('Датасет проверен: %s', report)
    if args.check_data:
        return

    if not resume_training and any((output / p).exists() for p in ('last.pt', 'best.pt')):
        raise ValueError(f'В output_dir ({output}) уже есть сохраненные чекпоинты. Для продолжения обучения установите resume_training: true либо укажите другую директорию output_dir.')

    checkpoint = None
    if resume_training:
        checkpoint = torch.load(cfg['pretrained_path'], map_location='cpu', weights_only=False)
        if checkpoint.get('format_version') != 1:
            raise ValueError(f'Неподдерживаемый формат чекпоинта для resume: {checkpoint.get("format_version")}')
        compatible_config(checkpoint['config'], cfg)
        if checkpoint.get('dataset_fingerprint') != fingerprint:
            raise ValueError('Датасет изменился с момента сохранения чекпоинта (отпечаток не совпадает)')

    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = choose_device(cfg['device'])

    if resume_training:
        model = create_model(cfg, pretrained=False).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        LOG.info('Веса модели для resume успешно восстановлены из %s', cfg['pretrained_path'])
    else:
        pretrained_encoder = bool(cfg.get('encoder_weights'))
        model = create_model(cfg, pretrained=pretrained_encoder).to(device)
        if pretrained_path:
            initialize_weights(model, pretrained_path, same_num_classes=same_num_classes)

    dataset = ElevationDataset(records['train'], cfg, is_train=True)
    num_train_examples = len(dataset)
    num_batches = math.ceil(num_train_examples / cfg['batch_size'])
    steps_per_epoch = max(1, math.ceil(num_batches / cfg['accumulation_steps']))

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if n.startswith('encoder.')], 'lr': cfg['encoder_lr']},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.')], 'lr': cfg['decoder_lr']}
    ], weight_decay=cfg['weight_decay'])

    scheduler_type = cfg.get('scheduler', 'onecycle')
    if scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[cfg['encoder_max_lr'], cfg['decoder_max_lr']],
            epochs=cfg['epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg['onecycle_pct_start'],
            div_factor=cfg['onecycle_div_factor'],
            final_div_factor=cfg['onecycle_final_div_factor']
        )
    else:
        scheduler = None

    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'] and device.type == 'cuda')
    loss_fn = ElevationLoss(cfg['pos_weight'], cfg['dice_weight'])

    epoch = 0
    next_batch = 0
    global_step = 0
    best = -1.0
    loss_sum = 0.0
    loss_batches = 0

    if resume_training:
        old_sched_type = checkpoint.get('scheduler_type')
        if old_sched_type == 'ReduceLROnPlateau' or (
            isinstance(checkpoint.get('scheduler'), dict) and 'mode' in checkpoint.get('scheduler')
        ):
            raise ValueError(
                'Возобновление обучения (resume_training: true) с прежним планировщиком ReduceLROnPlateau не реализовано. '
                'Используйте данный чекпоинт как источник весов (resume_training: false).'
            )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            if checkpoint.get('scheduler') is None:
                raise ValueError('В чекпоинте отсутствует сохраненное состояние планировщика OneCycleLR')
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
                LOG.info('Состояние планировщика OneCycleLR успешно восстановлено из чекпоинта')
            except Exception as e:
                raise ValueError(
                    f'Не удалось восстановить состояние планировщика OneCycleLR: {e}. '
                    'Убедитесь, что параметры epochs, batch_size и accumulation_steps не изменялись.'
                )

        if scaler.is_enabled() and checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        elif scaler.is_enabled():
            LOG.warning('В чекпоинте отсутствует состояние AMP; GradScaler инициализирован заново')

        epoch = checkpoint['epoch']
        next_batch = checkpoint['next_batch']
        global_step = checkpoint['global_step']
        best = checkpoint['best_iou']
        loss_sum = checkpoint['loss_sum']
        loss_batches = checkpoint['loss_batches']
        restore_rng(checkpoint['rng'])
        LOG.info('Продолжение обучения: эпоха %d, следующий батч %d, шаг оптимизатора %d, лучший IoU %.4f',
                 epoch + 1, next_batch, global_step, best)

    write_json(output / 'config.json', cfg)
    writer = SummaryWriter(str(output / 'tensorboard'), purge_step=epoch + 1 if resume_training else None)

    def snapshot():
        return {
            'format_version': 1,
            'config': cfg,
            'dataset_fingerprint': fingerprint,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'scheduler_type': cfg.get('scheduler', 'onecycle'),
            'scaler': scaler.state_dict() if scaler.is_enabled() else None,
            'rng': capture_rng(),
            'epoch': epoch,
            'next_batch': next_batch,
            'global_step': global_step,
            'best_iou': best,
            'loss_sum': loss_sum,
            'loss_batches': loss_batches,
        }

    if not resume_training:
        atomic_save(snapshot(), output / 'last.pt')

    history_path = output / 'history.jsonl'
    if resume_training and history_path.exists():
        lines = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        history_path.write_text(''.join(json.dumps(x) + '\n' for x in lines if x['epoch'] <= epoch), encoding='utf-8')

    try:
        while epoch < cfg['epochs']:
            dataset.epoch = epoch
            generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
            order = torch.randperm(len(dataset), generator=generator).tolist()
            batches = [order[i:i + cfg['batch_size']] for i in range(0, len(order), cfg['batch_size'])]
            if next_batch > len(batches):
                raise ValueError('Некорректный номер батча в чекпоинте')
            start = next_batch
            loader = DataLoader(
                dataset,
                batch_sampler=batches[start:],
                num_workers=cfg['num_workers'],
                pin_memory=device.type == 'cuda',
                generator=generator,
                persistent_workers=False
            )
            model.train()
            if cfg['freeze_batchnorm']:
                freeze_batchnorm(model)
            optimizer.zero_grad(set_to_none=True)

            for batch_index, (images, target) in enumerate(
                tqdm(loader, desc=f'train {epoch+1}/{cfg["epochs"]}', mininterval=2),
                start
            ):
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                group_start = (batch_index // cfg['accumulation_steps']) * cfg['accumulation_steps']
                group_end = min(group_start + cfg['accumulation_steps'], len(batches))
                group_examples = sum(len(batches[j]) for j in range(group_start, group_end))

                with torch.autocast(device_type=device.type, enabled=cfg['amp'] and device.type == 'cuda'):
                    logits = model(normalize(images, cfg['normalization_mean'], cfg['normalization_std']))
                    loss = loss_fn(logits, target)

                if not torch.isfinite(loss):
                    raise FloatingPointError('Нечисловая функция потерь')

                scaler.scale(loss * (len(images) / group_examples)).backward()
                loss_sum += float(loss.detach()) * len(images)
                loss_batches += len(images)

                if batch_index + 1 == group_end:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    step_skipped = scaler.is_enabled() and (scaler.get_scale() < scale_before)
                    if not step_skipped:
                        if scheduler is not None:
                            scheduler.step()
                        global_step += 1
                        writer.add_scalar('lr/encoder', optimizer.param_groups[0]['lr'], global_step)
                        writer.add_scalar('lr/decoder', optimizer.param_groups[1]['lr'], global_step)
                    else:
                        LOG.warning('Шаг оптимизатора пропущен из-за некорректных градиентов (AMP scale: %s -> %s)',
                                    scale_before, scaler.get_scale())
                    optimizer.zero_grad(set_to_none=True)
                    next_batch = batch_index + 1
                    if global_step % cfg['save_every_steps'] == 0 or next_batch == len(batches):
                        atomic_save(snapshot(), output / 'last.pt')

            metrics = evaluate(model, records['val'], cfg, device, output, epoch)
            score = metrics['iou_elevation']
            if score is None:
                raise ValueError('В валидационной выборке нет пикселей для вычисления IoU возвышенностей')

            improved = score > best
            if improved:
                best = score

            summary = {
                'epoch': epoch + 1,
                'train_loss': loss_sum / max(loss_batches, 1),
                **metrics,
                'encoder_lr': optimizer.param_groups[0]['lr'],
                'decoder_lr': optimizer.param_groups[1]['lr'],
            }
            LOG.info('Эпоха %d: train_loss=%.4f; IoU возвышенностей=%.4f; IoU фона=%s',
                     epoch + 1, summary['train_loss'], score, metrics['iou_background'])
            with history_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(summary) + '\n')
            for key, value in summary.items():
                if isinstance(value, (float, int)):
                    writer.add_scalar(key, value, epoch + 1)
            writer.flush()
            epoch += 1
            next_batch = 0
            loss_sum = 0.0
            loss_batches = 0
            state = snapshot()
            if improved:
                atomic_save(state, output / 'best.pt')
            atomic_save(state, output / 'last.pt')
    except KeyboardInterrupt:
        LOG.warning('Обучение прервано. Для продолжения используйте last.pt; незаписанные шаги будут повторены.')
    finally:
        writer.close()


if __name__ == '__main__':
    main()
