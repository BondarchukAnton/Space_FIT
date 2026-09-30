import math
import numpy as np
from datasets.dataset import prepare_datasets, segmentation_collate_fn, NUM_TARGET_CLASSES, TARGET_CLASS_NAMES
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
from datetime import datetime


class _Tee:
    """Пишет вывод одновременно в консоль и в лог-файл."""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        # Для лог-файла пропускаем промежуточные '\r' обновления progress-бара
        if '\r' in data and '\n' not in data:
            return
        clean_data = data.replace('\r', '')
        if clean_data:
            self.log_file.write(clean_data)
            self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


# Сохраняем весь вывод в консоль в текстовый файл в папке logs
_LOGS_DIR = Path('logs')
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = open(_LOGS_DIR / f"train_{datetime.now():%Y%m%d_%H%M%S}.txt", 'w', encoding='utf-8')
sys.stdout = _Tee(sys.__stdout__, _LOG_FILE)
sys.stderr = _Tee(sys.__stderr__, _LOG_FILE)


def save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss, experiment_dir, model_name, scheduler_state_dict):
    model_dst = experiment_dir / model_name
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler': scheduler_state_dict,
        'loss': mean_loss,
    }, model_dst)


BATCH_SIZE = 8  # RTX 5090 (32GB) with MaxViT encoder
ACCUMULATION_STEPS = 2  # Effective batch size = 16
WRITER_EPOCH = 1
start_epoch = 1
EPOCHS = 50
resolution = 512
LABEL = 'UnetPP_maxvit_6cls_512'

# ---- Параметры дообучения / возобновления обучения ----
PRETRAINED_PATH = None       # Путь к чекпоинту предобученной модели (None — обучение без чекпоинта)
SAME_NUM_CLASSES = False     # True — одинаковое число классов (полная загрузка модели),
                             # False — разное число классов (загрузка весов только в общие слои до слоя классификации)
RESUME_TRAINING = False      # True — продолжить обучение (восстановить оптимизатор, планировщик и эпоху),
                             # False — только взять веса модели и начать обучение заново с 1-й эпохи

train_dataset, val_dataset = prepare_datasets(resolution=resolution, target_m_per_px=10.0)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                          num_workers=4, persistent_workers=True, pin_memory=True,
                          prefetch_factor=2, collate_fn=segmentation_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                        collate_fn=segmentation_collate_fn)

print('Количество объектов в обучающей выборке ', len(train_dataset))
print('Количество объектов в валидационной выборке ', len(val_dataset))

iou_metric = MulticlassJaccardIndex(
    num_classes=NUM_TARGET_CLASSES,
    average='none',
    ignore_index=None
).cpu()

accuracy_metric = MulticlassAccuracy(
    num_classes=NUM_TARGET_CLASSES,
    average='micro'
).cpu()

model = smp.UnetPlusPlus(
    encoder_name='tu-maxvit_base_tf_512',
    encoder_weights='imagenet',
    in_channels=3,
    classes=NUM_TARGET_CLASSES,
)
model = model.cuda()

preprocess_params = smp.encoders.get_preprocessing_params('tu-maxvit_base_tf_512')
ENCODER_MEAN = torch.tensor(preprocess_params['mean'], device='cuda').view(1, 3, 1, 1)
ENCODER_STD = torch.tensor(preprocess_params['std'], device='cuda').view(1, 3, 1, 1)

def normalize_tensor(tensor):
    tensor = tensor.to('cuda')
    return (tensor - ENCODER_MEAN) / ENCODER_STD

param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad]},
    {"params": [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]},
]

optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)

steps_per_epoch = max(1, math.ceil(len(train_loader) / ACCUMULATION_STEPS))

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-5, 1e-4],  # [encoder, decoder]
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    pct_start=0.1,
    div_factor=10.0,
    final_div_factor=1e4
)

class ComboLoss(nn.Module):
    def __init__(self, num_classes, weights=None):
        super().__init__()
        if weights is None:
            weights = {'dice': 0.5, 'ce': 0.3, 'focal': 0.2}
        self.weights = weights
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', gamma=2.0)

    def forward(self, outputs, targets):
        loss = 0.0
        if self.weights.get('dice', 0) > 0:
            loss += self.weights['dice'] * self.dice_loss(outputs, targets)
        if self.weights.get('ce', 0) > 0:
            loss += self.weights['ce'] * self.ce_loss(outputs, targets)
        if self.weights.get('focal', 0) > 0:
            loss += self.weights['focal'] * self.focal_loss(outputs, targets)
        return loss

criterion = ComboLoss(num_classes=NUM_TARGET_CLASSES, weights={'dice': 0.6, 'ce': 0.3, 'focal': 0.1}).cuda()
scaler = torch.amp.GradScaler('cuda')

if PRETRAINED_PATH:
    print(f'Загрузка предобученной модели из: {PRETRAINED_PATH}')
    checkpoint = torch.load(PRETRAINED_PATH, weights_only=False, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

    # Убираем префикс 'module.', если модель сохранялась через DataParallel/DDP
    state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    if SAME_NUM_CLASSES:
        # Количество классов совпадает: загружаем модель полностью
        model.load_state_dict(state_dict)
        print('Все веса модели (включая классификатор) успешно загружены')
    else:
        # Количество классов отличается: загружаем веса только в общие слои до слоя классификации (encoder, decoder).
        # Слой классификации (segmentation_head) исключается, так как его размерность зависит от числа классов,
        # и он обучается заново под текущее количество классов (NUM_TARGET_CLASSES).
        common_weights = {
            k: v for k, v in state_dict.items()
            if not k.startswith('segmentation_head.')
        }
        msg = model.load_state_dict(common_weights, strict=False)
        print('Веса общих слоев (encoder, decoder) успешно загружены до слоя классификации')
        print(f'Слой классификации (segmentation_head) инициализирован заново под {NUM_TARGET_CLASSES} классов')
        if msg.missing_keys:
            print(f'Пропущенные ключи (классификатор): {msg.missing_keys}')

    if RESUME_TRAINING:
        if not SAME_NUM_CLASSES:
            print('Внимание: RESUME_TRAINING=True, но SAME_NUM_CLASSES=False (число классов отличается).')
            print('Состояние оптимизатора и планировщика несовместимо с новой структурой классификатора.')
            print('Обучение начнется заново с 1-й эпохи с новым оптимизатором и планировщиком.')
            start_epoch = 1
        else:
            missing_items = []
            if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('Состояние оптимизатора успешно восстановлено из чекпоинта')
            else:
                missing_items.append('optimizer_state_dict')

            if isinstance(checkpoint, dict) and 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print('Состояние планировщика (scheduler) успешно восстановлено из чекпоинта')
            else:
                missing_items.append('scheduler')

            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f'Номер эпохи восстановлен: продолжение обучения с эпохи {start_epoch}')
            else:
                missing_items.append('epoch')
                start_epoch = 1

            if missing_items:
                print(f'Информация: в чекпоинте отсутствуют следующие данные: {", ".join(missing_items)}.')
                if 'epoch' in missing_items:
                    print('Обучение начнется с 1-й эпохи.')
    else:
        print('RESUME_TRAINING=False: загружены только веса модели, обучение начинается заново с 1-й эпохи')
        start_epoch = 1

logs_root = Path('/mnt/980EAB530EAB2968/hdd_logs/OtherProject/SPACE')
experiment_dir = logs_root / LABEL
writer = SummaryWriter(experiment_dir)
top_iou = 0.0
top_mean_loss = 200

TARGET_COLORS = {
    0: (0, 0, 0),         # Фон — чёрный
    1: (0, 255, 0),     # Лесной массив — зелёный
    2: (255, 255, 0),     # Поле — золотой
    3: (0, 0, 255),    # Водоём — синий
    4: (255, 0, 0),     # Городская территория — красный
    5: (255, 0, 255),     # Горный район — фиолетовый
}

def build_vis_image(X_vis, y_vis, output_vis, resolution):
    row_N = X_vis.shape[0]
    
    def colorize_mask(idx_mask):
        rgb = torch.zeros(idx_mask.shape[0], 3, resolution, resolution)
        for cls_idx, (r, g, b) in TARGET_COLORS.items():
            match = (idx_mask == cls_idx)
            rgb[:, 0][match] = r / 255.0
            rgb[:, 1][match] = g / 255.0
            rgb[:, 2][match] = b / 255.0
        return rgb
    
    gt_rgb = colorize_mask(y_vis)
    pred_idx = output_vis.argmax(1)
    pred_rgb = colorize_mask(pred_idx)
    
    correct = (pred_idx == y_vis)
    acc_rgb = torch.zeros(row_N, 3, resolution, resolution)
    acc_rgb[:, 0] = (~correct).float()
    acc_rgb[:, 1] = correct.float()
    
    img_col = torch.cat(list(X_vis), dim=1)
    gt_col = torch.cat(list(gt_rgb), dim=1)
    pred_col = torch.cat(list(pred_rgb), dim=1)
    acc_col = torch.cat(list(acc_rgb), dim=1)
    
    return torch.cat([img_col, gt_col, pred_col, acc_col], dim=2)

for epoch in range(start_epoch, EPOCHS + 1):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            torch.set_grad_enabled(True)
            loader = train_loader
            optimizer.zero_grad()
        else:
            model.eval()
            torch.set_grad_enabled(False)
            loader = val_loader

        if phase != 'train':
            iou_metric.reset()
            accuracy_metric.reset()

        running_loss = []
        X_vis, y_vis, output_vis = None, None, None
        step = 0
        
        for batch in tqdm(loader, desc=f'{phase} loader in {epoch} epoch:', mininterval=2.0):
            X, y = batch
            X_normalized = normalize_tensor(X)
            
            with torch.amp.autocast('cuda'):
                output = model(X_normalized)
                loss = criterion(output, y.cuda())
                
            running_loss.append(loss.item())

            if phase == 'train' and ((epoch % WRITER_EPOCH == 0) or epoch == 1) and X_vis is None:
                X_vis = X[:min(6, X.shape[0])]
                y_vis = y[:min(6, X.shape[0])]
                output_vis = output.detach().cpu()[:min(6, X.shape[0])]

            step += 1
            if phase != 'train' and ((epoch % WRITER_EPOCH == 0) or epoch == 1):
                pred_classes = output.argmax(dim=1).cpu().detach()
                target_classes = y.cpu()
                accuracy_metric.update(pred_classes, target_classes)
                iou_metric.update(pred_classes, target_classes)

                if X_vis is None:
                    X_vis = X[:min(6, X.shape[0])]
                    y_vis = y[:min(6, X.shape[0])]
                    output_vis = output.detach().cpu()[:min(6, X.shape[0])]

            if phase == 'train':
                scaler.scale(loss / ACCUMULATION_STEPS).backward()
                is_accum_step = (step % ACCUMULATION_STEPS == 0) or (step == len(train_loader))
                if is_accum_step:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

        mean_loss = sum(running_loss) / len(running_loss) if running_loss else 0.0
        del running_loss

        if phase == 'val' and mean_loss < top_mean_loss:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            scheduler_state_dict = scheduler.state_dict()
            save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss, experiment_dir,
                             'top_loss.pt', scheduler_state_dict)
            top_mean_loss = mean_loss

        if phase != 'train' and ((epoch % WRITER_EPOCH == 0) or epoch == 1):
            class_ious = iou_metric.compute()
            overall_iou = torch.nanmean(class_ious).item()
            acc = accuracy_metric.compute().item()

            print(f'точность {phase}_acc', f'{acc:.4f}')
            print(f'IoU {phase}_iou', f'{overall_iou:.4f}')
            print(f"Имена классов: {list(TARGET_CLASS_NAMES.values())}")
            print(f'  Class IoUs: {[f"{iou:.4f}" for iou in class_ious.cpu().numpy()]}')

            iou_metric.reset()
            accuracy_metric.reset()
            
            if phase == 'val' and overall_iou > top_iou:
                model_state_dict = model.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                scheduler_state_dict = scheduler.state_dict()
                save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss, experiment_dir,
                                 'top_model.pt', scheduler_state_dict)
                top_iou = overall_iou

            if (epoch % WRITER_EPOCH == 0) or epoch == 1:
                if X_vis is not None:
                    out_img = build_vis_image(X_vis, y_vis, output_vis, resolution)
                    writer.add_image(f'{phase}_img', out_img, epoch)
                    writer.add_scalar(f'{phase}_acc', acc, epoch)
                    writer.add_scalar(f'{phase}_iou', overall_iou, epoch)

        if phase == 'train' and ((epoch % WRITER_EPOCH == 0) or epoch == 1) and X_vis is not None:
            out_img = build_vis_image(X_vis, y_vis, output_vis, resolution)
            writer.add_image('train_img', out_img, epoch)

        writer.add_scalar(f'{phase}_loss', mean_loss, epoch)
        torch.cuda.empty_cache()

    if epoch and epoch % WRITER_EPOCH == 0:
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        scheduler_state_dict = scheduler.state_dict()
        save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss, experiment_dir,
                         f'{epoch:05d}.pt', scheduler_state_dict)

    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler_state_dict = scheduler.state_dict()
    save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss, experiment_dir,
                     'last_model.pt', scheduler_state_dict)

    writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch)
    writer.flush()

torch.cuda.empty_cache()
writer.close()
