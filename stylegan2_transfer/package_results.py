import argparse
import csv
import json
import zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    log=a.run/'losses.csv'
    if log.exists():
        with log.open() as f: rows=list(csv.DictReader(f))
        if rows:
            fig,axes=plt.subplots(1,2,figsize=(10,3.8))
            x=[float(r['nimg'])/1000 for r in rows]
            for col in ['G','D']:
                axes[0].plot(x,[float(r[col]) for r in rows],label=col,alpha=.7,linewidth=.7)
            axes[0].set(xlabel='Fine-tuning kimg',ylabel='Non-saturating logistic loss');axes[0].legend()
            for col in ['real_score','fake_score']:
                axes[1].plot(x,[float(r[col]) for r in rows],label=col,linewidth=.7)
            axes[1].set(xlabel='Fine-tuning kimg',ylabel='Discriminator score');axes[1].legend()
            fig.tight_layout();fig.savefig(a.run/'loss_curves.png',dpi=160);plt.close(fig)
    with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.run.rglob('*')):
            if f.is_file() and f.suffix.lower() in {'.json','.jsonl','.csv','.png','.md','.txt'}:
                z.write(f,f.relative_to(a.run))
    print(f'Results: {a.output} ({a.output.stat().st_size/2**20:.1f} MiB); checkpoints are saved separately.')
