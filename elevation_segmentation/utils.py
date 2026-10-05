"""Матрица ошибок, журналирование и атомарная запись чекпоинтов."""
import json
import logging
import os
from pathlib import Path
import numpy as np
import torch


def setup_logging(output=None):
    """Подключает консоль и, при необходимости, файл журнала."""
    handlers=[logging.StreamHandler()]
    if output is not None:
        handlers.append(logging.FileHandler(Path(output)/'train.log',encoding='utf-8'))
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s %(name)s — %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',handlers=handlers,force=True)


def atomic_save(state,path):
    """Заменяет чекпоинт только после успешной записи временного файла."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as stream:
        torch.save(state,stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)


def confusion(probability,mask,threshold):
    """Возвращает матрицу 2×2: строки — разметка, столбцы — предсказание."""
    valid=mask!=255
    target=mask[valid].astype(np.int64); predicted=(probability[valid]>=threshold).astype(np.int64)
    return np.bincount(2*target+predicted,minlength=4).reshape(2,2)


def scores(matrix):
    """Рассчитывает метрики по общей матрице, а не среднее метрик батчей."""
    tn,fp,fn,tp=map(int,matrix.ravel())
    def ratio(a,b): return a/b if b else None
    values={'iou_elevation':ratio(tp,tp+fp+fn),'iou_background':ratio(tn,tn+fp+fn),
            'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
            'dice':ratio(2*tp,2*tp+fp+fn),'accuracy':ratio(tp+tn,tp+tn+fp+fn),
            'tn':tn,'fp':fp,'fn':fn,'tp':tp}
    present=[values[k] for k in ('iou_elevation','iou_background') if values[k] is not None]
    values['mean_iou']=sum(present)/len(present) if present else None
    return values


def write_json(path,data):
    """Записывает JSON без нечисловых значений NaN/Infinity."""
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
