"""Бинарная кросс-энтропия и Dice с исключением индекса 255."""
import torch
import torch.nn.functional as F
from torch import nn


class ElevationLoss(nn.Module):
    """Учитывает фон в BCE и исключает недостоверные пиксели из обеих составляющих."""
    def __init__(self, pos_weight=2., dice_weight=.5):
        super().__init__(); self.pos_weight=pos_weight; self.dice_weight=dice_weight

    def forward(self, logits, target):
        logits = logits[:,0].float(); valid = target != 255
        if not valid.any():
            return logits.sum()*0
        y = (target == 1).float(); v = valid.float()
        bce = F.binary_cross_entropy_with_logits(logits,y,reduction='none',
                                                  pos_weight=logits.new_tensor(self.pos_weight))
        bce = (bce*v).sum()/v.sum()
        p = logits.sigmoid()*v; y = y*v
        # Dice вычисляется по батчу, включая полностью фоновые вырезки.
        dice = 1-(2*(p*y).sum()+1.)/(p.sum()+y.sum()+1.)
        return (1-self.dice_weight)*bce+self.dice_weight*dice
