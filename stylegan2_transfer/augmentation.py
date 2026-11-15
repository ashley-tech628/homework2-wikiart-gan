"""A memory-conscious subset of NVIDIA ADA: horizontal flip and color."""
import torch
from runtime import VENDOR
from training.augment import AugmentPipe


class FlipColorAugment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.color = AugmentPipe(brightness=1,contrast=1,lumaflip=1,hue=1,saturation=1)

    @property
    def p(self):
        return self.color.p

    def forward(self,x):
        flip=(torch.rand(x.shape[0],1,1,1,device=x.device)<self.p*.5)
        return self.color(torch.where(flip,x.flip(3),x))
