"""Обучение RGB-модели с проверкой на полных изображениях val."""
import argparse
import json
import logging
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .dataset import load_records,inspect_dataset,ElevationDataset,read_rgb,read_mask
from .model import create_model,normalize,freeze_batchnorm,initialize_weights
from .losses import ElevationLoss
from .inference import predict_probability,choose_device
from .utils import setup_logging,atomic_save,confusion,scores,write_json

LOG=logging.getLogger(__name__)


def validate_config(cfg):
    """Проверяет параметры, влияющие на размеры тензоров и цикл обучения."""
    for key in ('normalization_mean','normalization_std'):
        if len(cfg[key])!=3 or not all(math.isfinite(x) for x in cfg[key]):raise ValueError(f'Неверный {key}')
    if any(x<=0 for x in cfg['normalization_std']):raise ValueError('std должен быть положительным')
    for key in ('crop_size','tile_size'):
        if cfg[key]<32 or cfg[key]%32: raise ValueError(f'{key}: требуется положительное кратное 32')
    for key in ('batch_size','accumulation_steps','epochs','crops_per_image','save_every_steps'):
        if not isinstance(cfg[key],int) or cfg[key]<1: raise ValueError(f'Неверный {key}')
    if not 0<=cfg['overlap']<cfg['tile_size']: raise ValueError('Неверное перекрытие окон')
    if not 0<=cfg['positive_crop_probability']<=1 or not 0<=cfg['dice_weight']<=1: raise ValueError('Неверная доля')
    if not 0<cfg['threshold']<1 or cfg['pos_weight']<=0: raise ValueError('Неверный порог или вес класса')
    if cfg['num_workers']<0: raise ValueError('num_workers должен быть неотрицательным')
    for key in ('encoder_lr','decoder_lr'):
        if not math.isfinite(cfg[key]) or cfg[key]<=0: raise ValueError(f'Неверный {key}')
    if cfg['batch_size']==1 and not cfg['freeze_batchnorm']:
        raise ValueError('При batch_size=1 используйте freeze_batchnorm=true')


def capture_rng():
    """Сохраняет генераторы, используемые моделью; аугментации привязаны к индексам."""
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python':random.getstate()}


def restore_rng(state):
    """Восстанавливает случайные состояния после построения модели и оптимизатора."""
    torch.set_rng_state(state['torch']); random.setstate(state['python'])
    if state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])


def compatible_config(old,new):
    """Разрешает менять только местонахождение файлов, устройство и число рабочих процессов."""
    ignored={'dataset_root','output_dir','device','num_workers'}
    changed=[key for key in set(old)|set(new) if key not in ignored and old.get(key)!=new.get(key)]
    if changed: raise ValueError(f'Для resume изменены параметры {changed}. Для нового обучения используйте --init-from.')


def evaluate(model,records,cfg,device,output,epoch):
    """Считает метрики каждого пикселя val один раз после объединения окон."""
    from PIL import Image
    matrix=np.zeros((2,2),np.int64); groups={}; per_image=[]
    preview=output/'previews';preview.mkdir(exist_ok=True)
    for i,row in enumerate(tqdm(records,desc='val',mininterval=2)):
        rgb=read_rgb(row['image']);mask=read_mask(row['mask'])
        probability=predict_probability(model,rgb,device,cfg['tile_size'],cfg['overlap'],cfg['amp'],cfg['normalization_mean'],cfg['normalization_std'])
        cm=confusion(probability,mask,cfg['threshold']); matrix+=cm
        group=row.get('group','unknown');groups.setdefault(group,np.zeros((2,2),np.int64));groups[group]+=cm
        per_image.append({'name':row['name'],'group':group,**scores(cm)})
        if i<3:
            overlay=rgb.copy();hit=probability>=cfg['threshold']
            overlay[hit]=(.55*overlay[hit]+.45*np.array([255,40,20])).astype(np.uint8)
            gt=np.zeros_like(rgb);gt[mask==1]=[255,255,255];gt[mask==255]=[180,0,180]
            canvas=np.concatenate([rgb,gt,overlay],axis=1)
            image=Image.fromarray(canvas);image.thumbnail((1440,480))
            image.save(preview/f'val_{i:02d}.jpg')
    result=scores(matrix);result['groups']={g:scores(m) for g,m in groups.items()}
    write_json(output/'val_metrics.json',{'epoch':epoch+1,**result,'images':per_image})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--data');parser.add_argument('--output');parser.add_argument('--device')
    group=parser.add_mutually_exclusive_group();group.add_argument('--resume');group.add_argument('--init-from')
    parser.add_argument('--check-data',action='store_true')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    for name,key in [('data','dataset_root'),('output','output_dir'),('device','device')]:
        if getattr(args,name):cfg[key]=getattr(args,name)
    validate_config(cfg);output=Path(cfg['output_dir']).resolve();output.mkdir(parents=True,exist_ok=True)
    setup_logging(output);records=load_records(cfg['dataset_root'])
    report,fingerprint=inspect_dataset(records);write_json(output/'dataset_report.json',report)
    LOG.info('Датасет: %s',report)
    if args.check_data:return
    if not args.resume and any((output/p).exists() for p in ('last.pt','best.pt')):
        raise ValueError('В output уже есть чекпоинт. Укажите --resume либо новую папку.')
    checkpoint=torch.load(args.resume,map_location='cpu',weights_only=True) if args.resume else None
    if checkpoint:
        if checkpoint.get('format_version')!=1:raise ValueError('Неподдерживаемый формат resume')
        compatible_config(checkpoint['config'],cfg)
        if checkpoint['dataset_fingerprint']!=fingerprint:raise ValueError('Датасет изменился с момента чекпоинта')
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=choose_device(cfg['device']);model=create_model(cfg,pretrained=not bool(checkpoint)).to(device)
    if args.init_from:initialize_weights(model,args.init_from)
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if n.startswith('encoder.')],'lr':cfg['encoder_lr']},
        {'params':[p for n,p in model.named_parameters() if not n.startswith('encoder.')],'lr':cfg['decoder_lr']}],
        weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5,min_lr=1e-7)
    scaler=torch.amp.GradScaler('cuda',enabled=cfg['amp'] and device.type=='cuda')
    loss_fn=ElevationLoss(cfg['pos_weight'],cfg['dice_weight'])
    epoch=0;next_batch=0;global_step=0;best=-1.;loss_sum=0.;loss_batches=0
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']);scheduler.load_state_dict(checkpoint['scheduler'])
        if scaler.is_enabled() and checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])
        elif scaler.is_enabled():
            LOG.warning('В чекпоинте нет состояния AMP; GradScaler инициализирован заново')
        epoch=checkpoint['epoch'];next_batch=checkpoint['next_batch'];global_step=checkpoint['global_step']
        best=checkpoint['best_iou'];loss_sum=checkpoint['loss_sum'];loss_batches=checkpoint['loss_batches']
        restore_rng(checkpoint['rng'])
        LOG.info('Продолжение: эпоха %d, следующий батч %d, шаг %d',epoch+1,next_batch,global_step)
    write_json(output/'config.json',cfg)
    dataset=ElevationDataset(records['train'],cfg)
    writer=SummaryWriter(str(output/'tensorboard'),purge_step=epoch+1 if checkpoint else None)
    def snapshot():
        return {'format_version':1,'config':cfg,'dataset_fingerprint':fingerprint,
                'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':capture_rng(),
                'epoch':epoch,'next_batch':next_batch,'global_step':global_step,'best_iou':best,
                'loss_sum':loss_sum,'loss_batches':loss_batches}
    if not checkpoint:atomic_save(snapshot(),output/'last.pt')
    history_path=output/'history.jsonl'
    if checkpoint and history_path.exists():
        lines=[json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
        history_path.write_text(''.join(json.dumps(x)+'\n' for x in lines if x['epoch']<=epoch))
    try:
        while epoch<cfg['epochs']:
            dataset.epoch=epoch
            generator=torch.Generator().manual_seed(cfg['seed']+epoch)
            order=torch.randperm(len(dataset),generator=generator).tolist()
            batches=[order[i:i+cfg['batch_size']] for i in range(0,len(order),cfg['batch_size'])]
            if next_batch>len(batches):raise ValueError('Некорректный номер батча в чекпоинте')
            start=next_batch
            loader=DataLoader(dataset,batch_sampler=batches[start:],num_workers=cfg['num_workers'],
                              pin_memory=device.type=='cuda',generator=generator,persistent_workers=False)
            model.train()
            if cfg['freeze_batchnorm']:freeze_batchnorm(model)
            optimizer.zero_grad(set_to_none=True)
            for batch_index,(images,target) in enumerate(tqdm(loader,desc=f'train {epoch+1}/{cfg["epochs"]}',mininterval=2),start):
                images=images.to(device,non_blocking=True);target=target.to(device,non_blocking=True)
                group_start=(batch_index//cfg['accumulation_steps'])*cfg['accumulation_steps']
                group_end=min(group_start+cfg['accumulation_steps'],len(batches))
                # Последняя неполная группа нормируется по фактическому числу примеров.
                group_examples=sum(len(batches[j]) for j in range(group_start,group_end))
                with torch.autocast(device_type=device.type,enabled=cfg['amp'] and device.type=='cuda'):
                    logits=model(normalize(images,cfg['normalization_mean'],cfg['normalization_std']));loss=loss_fn(logits,target)
                if not torch.isfinite(loss):raise FloatingPointError('Нечисловая функция потерь')
                scaler.scale(loss*(len(images)/group_examples)).backward()
                loss_sum+=float(loss.detach())*len(images);loss_batches+=len(images)
                if batch_index+1==group_end:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    global_step+=1;next_batch=batch_index+1
                    if global_step%cfg['save_every_steps']==0 or next_batch==len(batches):
                        atomic_save(snapshot(),output/'last.pt')
            metrics=evaluate(model,records['val'],cfg,device,output,epoch)
            score=metrics['iou_elevation']
            if score is None:raise ValueError('В val нет пикселей для вычисления IoU возвышенностей')
            scheduler.step(score);improved=score>best
            if improved:best=score
            summary={'epoch':epoch+1,'train_loss':loss_sum/max(loss_batches,1),**metrics,
                     'encoder_lr':optimizer.param_groups[0]['lr'],'decoder_lr':optimizer.param_groups[1]['lr']}
            LOG.info('Эпоха %d: train_loss=%.4f; IoU возвышенностей=%.4f; IoU фона=%s',
                     epoch+1,summary['train_loss'],score,metrics['iou_background'])
            with history_path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(summary)+'\n')
            for key,value in summary.items():
                if isinstance(value,(float,int)):writer.add_scalar(key,value,epoch+1)
            writer.flush();epoch+=1;next_batch=0;loss_sum=0.;loss_batches=0
            state=snapshot()
            if improved:atomic_save(state,output/'best.pt')
            atomic_save(state,output/'last.pt')
    except KeyboardInterrupt:
        LOG.warning('Обучение прервано. Для продолжения используйте last.pt; незаписанные шаги будут повторены.')
    finally:writer.close()


if __name__=='__main__':main()
