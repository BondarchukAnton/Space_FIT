"""Предсказание полноразмерной маски по RGB с перекрытием окон."""
import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import torch
if __package__ is None or __package__ == "":
    import sys
    _project_root = Path(__file__).resolve().parents[1]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from elevation_segmentation.model import create_model, normalize, MEAN, STD
    from elevation_segmentation.utils import setup_logging
else:
    from .model import create_model, normalize, MEAN, STD
    from .utils import setup_logging

LOG = logging.getLogger(__name__)


def choose_device(name='auto'):
    """Выбирает CPU или CUDA по явному параметру либо доступности CUDA."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def positions(length, tile, overlap):
    """Покрывает ось окнами без пропущенных крайних пикселей."""
    if length <= tile:
        return [0]
    values = list(range(0,length-tile+1,tile-overlap))
    if values[-1] != length-tile:
        values.append(length-tile)
    return values


@torch.inference_mode()
def predict_probability(model, rgb, device, tile_size=512, overlap=128, amp=True, mean=MEAN, std=STD):
    """Возвращает float32 H×W, сохраняя размер и масштаб исходного RGB.

    Args:
        rgb: Массив uint8 H×W×3 в порядке RGB.
        tile_size: Размер окна, кратный 32.
        overlap: Перекрытие соседних окон в пикселях.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('Ожидается непустой RGB uint8 H×W×3')
    if tile_size < 32 or tile_size % 32 or not 0 <= overlap < tile_size:
        raise ValueError('tile_size должен быть кратен 32; 0 <= overlap < tile_size')
    model.eval(); device=torch.device(device)
    h,w=rgb.shape[:2]
    padded=np.pad(rgb,((0,max(0,tile_size-h)),(0,max(0,tile_size-w)),(0,0)),mode='edge')
    ph,pw=padded.shape[:2]; total=np.zeros((ph,pw),np.float32); weights=np.zeros_like(total)
    window=np.maximum(np.outer(np.hanning(tile_size),np.hanning(tile_size)),.01).astype(np.float32)
    for y in positions(ph,tile_size,overlap):
        for x in positions(pw,tile_size,overlap):
            patch=padded[y:y+tile_size,x:x+tile_size]
            tensor=torch.from_numpy(patch.transpose(2,0,1).copy()).unsqueeze(0).to(device).float()/255
            with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
                probability=model(normalize(tensor,mean,std)).float().sigmoid()[0,0].cpu().numpy()
            total[y:y+tile_size,x:x+tile_size]+=probability*window
            weights[y:y+tile_size,x:x+tile_size]+=window
    if not (weights>0).all():
        raise RuntimeError('Обнаружены непокрытые пиксели')
    return (total/weights)[:h,:w]


class ElevationPredictor:
    """Загружает обученный модуль и предсказывает маску 0/1 по RGB."""
    def __init__(self, checkpoint, device='auto', threshold=None, tile_size=None, overlap=None):
        self.device=choose_device(device)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state.get('format_version') != 1:
            raise ValueError('Нужен чекпоинт, созданный elevation_segmentation.train')
        cfg=state['config']; self.model=create_model(cfg,pretrained=False).to(self.device)
        self.model.load_state_dict(state['model_state_dict'],strict=True); self.model.eval()
        self.threshold=cfg['threshold'] if threshold is None else threshold
        self.tile_size=cfg['tile_size'] if tile_size is None else tile_size
        self.overlap=cfg['overlap'] if overlap is None else overlap
        self.amp=cfg['amp']
        self.mean=cfg['normalization_mean']; self.std=cfg['normalization_std']
        if not 0 < self.threshold < 1:
            raise ValueError('threshold должен быть между 0 и 1')

    def predict_proba(self, rgb):
        """Возвращает вероятность класса возвышенности, массив float32 H×W."""
        return predict_probability(self.model,rgb,self.device,self.tile_size,self.overlap,self.amp,self.mean,self.std)

    def predict(self, rgb):
        """Возвращает индексную маску uint8 H×W со значениями 0 и 1."""
        return (self.predict_proba(rgb)>=self.threshold).astype(np.uint8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True); parser.add_argument('--image',required=True)
    parser.add_argument('--output',default='elevation_prediction'); parser.add_argument('--device',default='auto')
    parser.add_argument('--threshold',type=float); parser.add_argument('--tile-size',type=int); parser.add_argument('--overlap',type=int)
    args=parser.parse_args(); setup_logging()
    with Image.open(args.image) as image: rgb=np.array(image.convert('RGB'))
    predictor=ElevationPredictor(args.checkpoint,args.device,args.threshold,args.tile_size,args.overlap)
    probability=predictor.predict_proba(rgb); mask=(probability>=predictor.threshold).astype(np.uint8)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(mask).save(output/'mask.png')
    Image.fromarray(mask*255).save(output/'mask_preview.png')
    np.save(output/'probability.npy',probability)
    overlay=rgb.copy(); hit=mask==1
    overlay[hit]=(.55*overlay[hit]+.45*np.array([255,40,20])).astype(np.uint8)
    Image.fromarray(overlay).save(output/'overlay.png')
    LOG.info('Маска %s сохранена в %s',mask.shape,output)


if __name__=='__main__': main()
