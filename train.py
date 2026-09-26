import numpy as np
from datasets.forest_dataset import prepare_forest_water_datasets, forest_collate_fn
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from networks.vit_seg_modeling import VisionTransformer, CONFIGS
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
from datetime import datetime


class _Tee:
    """Пишет вывод одновременно в консоль и в лог-файл."""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


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

BATCH_SIZE = 13  # Физический размер батча (при 1024 разрешение, чтобы не было CUDA OOM)
ACCUMULATION_STEPS = 1  #
WRITER_EPOCH = 1
start_epoch = 1
EPOCHS = 20
resolution = 512
LABEL = 'TransUNet_forest_v6_onlyRGB_512'
continue_with = None

# !!!!!!!!!!!!!!1
acc = 0.99
IoU = 0.99

train_dataset, val_dataset_big, val_dataset_target = prepare_forest_water_datasets(resolution=resolution, chb_mode=False)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4,
                          persistent_workers=True, pin_memory=True, prefetch_factor=2, collate_fn=forest_collate_fn)  # при 8 оператива забивается
# val_loader_big = DataLoader(val_dataset_big, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=water_collate_fn)
val_loader_target = DataLoader(val_dataset_target, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, collate_fn=forest_collate_fn)

print('Количество объектов в обучающей выборке ', len(train_dataset))
# print('Количество объектов в val_dataset_big ', len(val_dataset_big))
print('Количество объектов в val_dataset_target ', len(val_dataset_target))
del train_dataset
del val_dataset_big
del val_dataset_target




iou_metric = MulticlassJaccardIndex(
    num_classes=2,
    average='none',
    ignore_index=None
).cpu()

accuracy_metric = MulticlassAccuracy(
    num_classes=2,
    average='micro'
).cpu()

# 1. Настройка конфигурации для нашей задачи (3 класса, размер 512)
config_vit = CONFIGS['R50-ViT-B_16']
config_vit.n_classes = 2      # 0 - Фон, 1 - объект
config_vit.n_skip = 3         # U-Net skip connections
config_vit.patches.grid = (int(resolution / 16), int(resolution / 16)) # Для 1024х1024 сетка будет 64х64

# 2. Создание модели
model = VisionTransformer(config_vit, img_size=resolution, num_classes=2)

# 3. Загрузка предобученных весов ImageNet21k
weights_path =  '/home/user/PycharmProjects/OptINS_etap3/Water_Semantic_Seg/imagenet21k_R50+ViT-B_16.npz'
model.load_from(weights=np.load(weights_path))

# Перенос на видеокарту
model = model.cuda()

param_dicts = [
    # Группа 1: Бэкбоун (ResNet)
    {"params": [p for n, p in model.named_parameters() if "resnet" in n and p.requires_grad]},

    # Группа 2: Всё остальное (Трансформер, Декодер, Голова)
    {"params": [p for n, p in model.named_parameters() if "resnet" not in n and p.requires_grad]},
]

# OneCycleLR сам назначит правильные шаги из списка max_lr
optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[1e-5, 1e-4],  # Пиковые значения: [Бэкбоун, Трансформер]
    steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS,  # Шагов оптимизатора в эпоху (с учётом накопления градиентов)
    epochs=EPOCHS,
    pct_start=0.1,        # 10% от всего времени обучения тратим на плавный разогрев
    div_factor=10.0,      # Стартуем с LR в 10 раз меньше пикового
    final_div_factor=1e4  # В конце спускаемся к микроскопическому значению
)


class ComboLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        import segmentation_models_pytorch as smp
        if weights is None:
            weights = {'dice': 0.5, 'ce': 0.3, 'focal': 0.2}

        self.weights = weights
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', alpha=0.25, gamma=2.0)

    def forward(self, outputs, targets):
        """
        outputs: [B, C, H, W] — логиты от модели
        targets: [B, C, H, W] — one-hot маска
        """
        # Преобразуем one-hot -> индекс классов
        targets_argmax = targets.argmax(dim=1).long()  # [B, H, W]

        loss = 0.0

        if self.weights.get('dice', 0) > 0:
            # DiceLoss ожидает индексную маску при mode='multiclass'
            loss += self.weights['dice'] * self.dice_loss(outputs, targets_argmax)

        if self.weights.get('ce', 0) > 0:
            loss += self.weights['ce'] * self.ce_loss(outputs, targets_argmax)

        if self.weights.get('focal', 0) > 0:
            loss += self.weights['focal'] * self.focal_loss(outputs, targets_argmax)

        return loss

criterion = ComboLoss(weights={'dice': 0.6, 'ce': 0.3, 'focal': 0.1}).cuda()

if continue_with:
    checkpoint = torch.load(continue_with, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Веса загружены успешно')

logs_root = Path('/mnt/980EAB530EAB2968/hdd_logs/Optins3etap') #'/mnt/16TBvolume1/hdd_logs/Optins3etap')
experiment_dir = logs_root / LABEL
writer = SummaryWriter(experiment_dir)
top_acc = 0
top_mean_loss = 200
best_val_iou = 0.0
overall_iou = 0.0
mean_ious = []

# Определяем параметры нормализации один раз
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

def normalize_tensor(tensor):
    """Нормализация тензора на GPU"""
    tensor = tensor.to('cuda')
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def build_vis_image(X_vis, y_vis, output_vis, resolution):
    """Собирает композитное изображение [Img, GT, Pred, Acc] для TensorBoard (один формат для train и val)."""
    row_N = X_vis.shape[0]

    # 2. Цветовая маска GT
    ancmsk = y_vis.argmax(1)
    anc_mask = torch.zeros(row_N, 3, resolution, resolution)
    # Цвета: Фон(0)-Красный, Лес(1)-Зеленый
    anc_mask[:, 0] = (ancmsk == 0)  # R
    anc_mask[:, 1] = (ancmsk == 1)  # G

    # 3. Цветовая маска Pred
    predmsk = output_vis.argmax(1)
    rgb_mask = torch.zeros(row_N, 3, resolution, resolution)
    rgb_mask[:, 0] = (predmsk == 0)  # R
    rgb_mask[:, 1] = (predmsk == 1)  # G

    # 4. Карта точности
    r_pred = (output_vis.argmax(1) == y_vis.argmax(1))
    rgb_acc = torch.zeros(row_N, 3, resolution, resolution)
    rgb_acc[:, 0] = (~r_pred) * 255  # Ошибки - Красный
    rgb_acc[:, 1] = (r_pred) * 255   # Правильно - Зеленый
    rgb_acc = (rgb_acc / 255)

    # Склеиваем сэмплы по вертикали
    img_col = torch.cat(list(X_vis), dim=1)
    y_col = torch.cat(list(anc_mask), dim=1)
    out_col = torch.cat(list(rgb_mask), dim=1)
    acc_col = torch.cat(list(rgb_acc), dim=1)

    # Финальная склейка: [Img, GT, Pred, Acc] по горизонтали (dim=2)
    return torch.cat([img_col, y_col, out_col, acc_col], dim=2)

for epoch in range(start_epoch, EPOCHS + 1):

    for phase in 'train val_t'.split():
    # for phase in 'train val_t'.split():
        if phase == 'train':
            model.train()
            torch.set_grad_enabled(True)
            loader = train_loader
            optimizer.zero_grad()

        # elif phase == 'val_big':
        #     model.eval()
        #     torch.set_grad_enabled(False)
        #     loader = val_loader_big

        elif phase == 'val_t':
            model.eval()
            torch.set_grad_enabled(False)
            loader = val_loader_target

        if phase != 'train':
            iou_metric.reset()
            accuracy_metric.reset()

        running_ans = []
        running_pred = []
        running_loss = []
        running_acc = []

        running_acc_1 = []
        all_ious = []
        # Переменные для визуализации картинок
        X_vis, y_vis, output_vis = None, None, None
        step = 0
        for batch in tqdm(loader, desc=f'{phase} loader in {epoch} epoch:'):
            X, y = batch
            X_normalized = normalize_tensor(X)
            output = model(X_normalized)
            loss = criterion(output, y.cuda())
            running_loss.append(loss.item())

            # Собираем данные для визуализации train (как на валидации)
            if phase == 'train' and ((epoch % WRITER_EPOCH == 0) or epoch == 1) and X_vis is None:
                X_vis = X[:min(6, X.shape[0])]
                y_vis = y[:min(6, X.shape[0])]
                output_vis = output.detach().cpu()[:min(6, X.shape[0])]

            step += 1
            if phase != 'train' and ((epoch % WRITER_EPOCH == 0) or epoch == 1):
                running_ans = y.detach().to(torch.bool).cpu()
                running_pred = output.detach().cpu()

                running_pred = (torch.eq(running_pred, running_pred.max(1)[0].unsqueeze(1).repeat(1, 2, 1, 1)))
                running_acc_1.append(running_pred[running_ans])

                # Вычисляем IoU для каждого класса
                pred_classes = output.argmax(dim=1).cpu().detach()  # [B, H, W]
                target_classes = y.argmax(dim=1).cpu()  # [B, H, W]
                accuracy_metric.update(pred_classes, target_classes)
                iou_metric.update(pred_classes, target_classes)

                if X_vis is None:
                    X_vis = X[:min(6, X.shape[0])]
                    y_vis = y[:min(6, X.shape[0])]
                    output_vis = output.detach().cpu()[:min(6, X.shape[0])]

                del running_ans
                del running_pred

            if phase == 'train':
                # Накопление градиентов: масштабируем loss, чтобы эффективный батч = BATCH_SIZE * ACCUMULATION_STEPS
                (loss / ACCUMULATION_STEPS).backward()
                if step % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()


        mean_loss = sum(running_loss) / len(running_loss)
        del running_loss

        if phase == 'val_t' and mean_loss < top_mean_loss:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            scheduler_state_dict = scheduler.state_dict()
            save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss, experiment_dir,
                             'top_loss.pt', scheduler_state_dict)
            top_mean_loss = mean_loss

        if phase != 'train' and ((epoch % WRITER_EPOCH == 0) or epoch == 1):
            class_ious = iou_metric.compute()  # [6] - IoU для каждого класса
            overall_iou = class_ious.mean().item()
            acc_2 = accuracy_metric.compute().item()

            print(f'точность {phase}_acc_2', f'{acc_2:.4f}')
            print(f'IoU {phase}_iou', f'{overall_iou:.4f}')
            print("Имена классов: ['Фон', 'Лес']")
            print(f'  Class IoUs: {[f"{iou:.4f}" for iou in class_ious.cpu().numpy()]}')

            # Сбрасываем метрики для следующей эпохи
            iou_metric.reset()
            accuracy_metric.reset()
            acc = torch.cat(running_acc_1, dim=0).to(torch.float32).mean().cpu()
            print(f'точность {phase}_acc', acc)
            del running_acc_1
            if phase == 'val_t' and acc > top_acc:
                model_state_dict = model.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                scheduler_state_dict = scheduler.state_dict()
                save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss, experiment_dir,
                                 'top_model.pt', scheduler_state_dict)
                top_acc = acc

            # Визуализация изображений (как в старом коде)
            if (epoch % WRITER_EPOCH == 0) or epoch == 1:
                if X_vis is not None:
                    out_img = build_vis_image(X_vis, y_vis, output_vis, resolution)
                    writer.add_image(f'{phase}_img', out_img, epoch)
                    writer.add_scalar(f'{phase}_acc', acc, epoch)
                    writer.add_scalar(f'{phase}_iou', overall_iou, epoch)

                    print(f'точность {phase}_acc (manual)', acc)
                    print(f'IoU {phase}', overall_iou)

        # Запись изображения с результатами обучения в TensorBoard (формат как на валидации)
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

    # Save last epoch checkpoint (overwritten every epoch)
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    scheduler_state_dict = scheduler.state_dict()
    save_checkpoints(epoch, model_state_dict, optimizer_state_dict, mean_loss, experiment_dir,
                     'last_model.pt', scheduler_state_dict)

    writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch)
    writer.flush()

torch.cuda.empty_cache()
writer.close()

