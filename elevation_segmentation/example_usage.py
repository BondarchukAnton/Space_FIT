"""Пример: python -m elevation_segmentation.example_usage --checkpoint ... --image ..."""
import argparse
import logging
import numpy as np
from PIL import Image
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    _project_root = Path(__file__).resolve().parents[1]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from elevation_segmentation.inference import ElevationPredictor
    from elevation_segmentation.utils import setup_logging
else:
    from .inference import ElevationPredictor
    from .utils import setup_logging


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--image',required=True)
    parser.add_argument('--output',default='elevation_mask.png');args=parser.parse_args()
    setup_logging()
    predictor=ElevationPredictor(args.checkpoint)
    # Модель создаётся один раз; затем predict можно вызывать для каждого RGB-кадра.
    with Image.open(args.image) as source:rgb=np.array(source.convert('RGB'))
    mask=predictor.predict(rgb)
    assert mask.shape==rgb.shape[:2] and mask.dtype==np.uint8
    Image.fromarray(mask).save(args.output)
    logging.info('Сохранена маска %s; классы: %s',mask.shape,np.unique(mask).tolist())
    # Для OpenCV: rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).
    # mask == 1 — локальные возвышенности; mask == 0 — остальная местность.


if __name__=='__main__':main()
