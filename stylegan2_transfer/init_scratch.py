"""Optional random initialization. No pretrained tensors are read or copied."""
import argparse
import json
from pathlib import Path
import torch
from runtime import Generator, Discriminator, atomic_save

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.manual_seed(2026)
    cfg=json.loads((Path(__file__).parent/'architecture.json').read_text())
    g=Generator(**cfg['G']);d=Discriminator(**cfg['D'])
    atomic_save({'model_config':cfg,'G':g.state_dict(),'G_ema':g.state_dict(),'D':d.state_dict(),
                 'nimg':0,'step':0,'pretrained':{'source':None,'initialization':'random seed 2026'}},a.output)
    print('Random 1024px model initialized. From-scratch training needs substantially more compute than transfer.')
