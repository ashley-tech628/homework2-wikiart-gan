"""Full-resolution WikiArt transfer training with exact optimizer/RNG recovery.

Networks, loss, path regularization and adaptive augmentations are NVIDIA's.
This single-GPU loop adds a deterministic full-data pass, regularizer
subsampling, finite-gradient checks, validation selection and durable saves.
"""
import argparse
import copy
import csv
import json
import math
import signal
import time
from pathlib import Path
import numpy as np
import torch
from runtime import make_models, load_state, atomic_save, cpu_state
from training.loss import StyleGAN2Loss
from augmentation import FlipColorAugment
from torch_utils import training_stats
from data import Paintings
from sample import sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--persist', type=Path, required=True)
    p.add_argument('--kimg', type=float, default=80)
    p.add_argument('--max-hours', type=float, default=2.5)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--microbatch', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--gamma', type=float, default=10)
    p.add_argument('--pl-weight', type=float, default=2)
    p.add_argument('--reg-interval', type=int, default=16)
    p.add_argument('--save-minutes', type=float, default=10)
    p.add_argument('--eval-kimg', type=float, default=10)
    p.add_argument('--val-count', type=int, default=512)
    p.add_argument('--max-steps', type=int, default=0, help='Local diagnostic only; zero means unlimited')
    p.add_argument('--skip-validation', action='store_true', help='Local diagnostic only')
    a = p.parse_args()
    if a.batch < 1 or a.microbatch < 1 or a.microbatch > a.batch or a.reg_interval < 1:
        p.error('Invalid batch sizes / regularization interval')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(2026)
    a.output.mkdir(parents=True,exist_ok=True)
    a.persist.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(a.manifest.read_text(encoding='utf-8'))
    checkpoint = a.persist/'latest.pt'
    resume = checkpoint.exists()
    s = load_state(checkpoint if resume else a.initial)
    if resume and s['dataset_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Dataset changed since the checkpoint. Refusing to resume a different experiment.')
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(regularizer_microbatch=1,regularizer_precision='fp32',
                  augmentation='NVIDIA ADA color subset + horizontal flip',
                  ema_half_life_kimg=10,ada_target=0.6,ada_kimg=500,
                  model='NVIDIA StyleGAN2; native 1024px; unconditional WikiArt transfer')
    critical = ['batch','microbatch','lr','gamma','pl_weight','reg_interval']
    if resume:
        for k in critical:
            if config[k] != s['config'][k]:
                raise ValueError(f'Resume requires unchanged {k}={s["config"][k]}')
    (a.persist/'config.json').write_text(json.dumps(config,indent=2))
    (a.persist/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (a.persist/'pretrained_provenance.json').write_text(json.dumps(s['pretrained'],indent=2))
    G,D,ema = make_models(s)
    aug = FlipColorAugment().cuda().train().requires_grad_(False)
    aug.p.copy_(torch.as_tensor(s.get('augment_p',0.0),device='cuda'))
    loss = StyleGAN2Loss(torch.device('cuda'),G.mapping,G.synthesis,D,augment_pipe=aug,
                        style_mixing_prob=0.9,r1_gamma=a.gamma,pl_weight=a.pl_weight,pl_batch_shrink=1)
    # Lazy regularization LR/beta correction, as in the official loop.
    ratio=a.reg_interval/(a.reg_interval+1)
    opts={name:torch.optim.Adam(net.parameters(),lr=a.lr*ratio,betas=(0.0,0.99**ratio))
          for name,net in [('G',G),('D',D)]}
    nimg,step,epoch,offset = [int(s.get(k,0)) for k in ['nimg','step','epoch','offset']]
    best_fid=s.get('best_fid',float('inf'))
    best_partial_fid=s.get('best_partial_fid',float('inf'))
    baseline=s.get('baseline_validation')
    bad_evals=int(s.get('bad_evals',0))
    last_eval_nimg=int(s.get('last_eval_nimg',0))
    if resume:
        for name,opt in opts.items(): opt.load_state_dict(s['optimizers'][name])
        loss.pl_mean.copy_(s['pl_mean'].cuda())
        torch.set_rng_state(s['rng_cpu'])
        torch.cuda.set_rng_state_all(s['rng_cuda'])
    training_stats.init_multiprocessing(rank=0,sync_device=None)
    stats=training_stats.Collector(regex='.*')
    stop={'requested':False}
    def request_stop(*_):
        stop['requested']=True
        print('Stop requested: will finish this optimizer step and save.',flush=True)
    signal.signal(signal.SIGINT,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    train_ids=manifest['train']
    target=max(int(a.kimg*1000),len(train_ids)) if not a.max_steps else int(a.kimg*1000)
    start=time.monotonic();last_save=start;start_nimg=nimg;start_step=step
    original_meta={k:s[k] for k in ['model_config','pretrained']}
    del s

    def inference_state():
        return dict(original_meta,G_ema=cpu_state(ema),nimg=nimg,step=step,
                    epoch=epoch,offset=offset,all_training_images_seen=epoch>=1,
                    dataset_fingerprint=manifest['fingerprint'])

    def save(reason):
        nonlocal last_save
        state=dict(original_meta,G=cpu_state(G),D=cpu_state(D),G_ema=cpu_state(ema),
            nimg=nimg,step=step,epoch=epoch,offset=offset,config=config,
            dataset_fingerprint=manifest['fingerprint'],optimizers={k:v.state_dict() for k,v in opts.items()},
            rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
            pl_mean=loss.pl_mean.detach().cpu(),augment_p=float(aug.p),best_fid=best_fid,
            best_partial_fid=best_partial_fid,
            baseline_validation=baseline,bad_evals=bad_evals,last_eval_nimg=last_eval_nimg,
            status=reason)
        atomic_save(state,checkpoint)
        status={'status':reason,'nimg':nimg,'target_nimg':target,'epoch':epoch,'offset':offset,
                'complete_training_passes':epoch,'elapsed_session_hours':(time.monotonic()-start)/3600,
                'checkpoint':str(checkpoint),'all_training_images_seen':epoch>=1}
        (a.persist/'status.json').write_text(json.dumps(status,indent=2))
        sample(ema,a.persist/'latest_grid_psi1.png',count=16)
        last_save=time.monotonic()
        print(f'SAVED: {checkpoint} | {nimg/1000:.2f} kimg | {reason}',flush=True)

    def validate():
        nonlocal best_fid,best_partial_fid,baseline,bad_evals,last_eval_nimg
        if a.skip_validation: return
        from evaluate import evaluate
        for net in [G,D]: net.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        result=evaluate(ema,manifest,'validation',a.val_count,a.output/'metric_cache')
        result['nimg']=nimg
        result['all_training_images_seen']=epoch>=1
        with (a.persist/'validation.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
        if nimg==0:
            baseline=result
            (a.persist/'baseline_validation.json').write_text(json.dumps(result,indent=2))
        else:
            full_pass=epoch>=1
            previous=best_fid if full_pass else best_partial_fid
            if result['fid_pool3']<previous:
                if full_pass:best_fid=result['fid_pool3']
                else:best_partial_fid=result['fid_pool3']
                best=inference_state();best['selection']=result
                checkpoint_name='best_finetuned.pt' if full_pass else 'best_partial.pt'
                prefix='best' if full_pass else 'partial'
                atomic_save(best,a.persist/checkpoint_name)
                sample(ema,a.persist/f'{prefix}_grid_psi1.png',count=16,individual=True)
                sample(ema,a.persist/f'{prefix}_grid_psi07.png',count=16,psi=.7)
            bad_evals = bad_evals+1 if baseline and result['fid_pool3']>baseline['fid_pool3']*1.25 else 0
        last_eval_nimg=nimg

    if not resume:
        sample(ema,a.persist/'pretrained_before_finetuning_psi1.png',count=16,individual=True)
        sample(ema,a.persist/'pretrained_before_finetuning_psi07.png',count=16,psi=.7)
        save('initialized')  # Tests actual home quota before spending hours training.
        validate()
    elif baseline is None and not a.skip_validation:
        # The server may have stopped after the initial durable save but
        # before baseline evaluation finished. Reconstruct the true starting
        # baseline, not the already fine-tuned current EMA.
        from evaluate import evaluate
        initial_G=make_models(load_state(a.initial),generator_only=True)
        baseline=evaluate(initial_G,manifest,'validation',a.val_count,a.output/'metric_cache')
        baseline['nimg']=0
        (a.persist/'baseline_validation.json').write_text(json.dumps(baseline,indent=2))
        with (a.persist/'validation.jsonl').open('a') as f:f.write(json.dumps(baseline)+'\n')
        del initial_G
        torch.cuda.empty_cache()
    start=time.monotonic()
    training_seconds=0.0
    print(f'Training {len(train_ids)} originals at 1024px; resume={resume}; {nimg}/{target} images.',flush=True)
    print('Full-state durable checkpoint may need about 2 GiB free home quota for atomic replacement.',flush=True)
    log_path=a.persist/'losses.csv'
    fields=['step','nimg','G','D','real_score','fake_score','r1_penalty','pl_penalty','augment_p','seconds_per_image']
    reason='target_reached'
    while nimg<target:
        permutation=np.random.default_rng(manifest['seed']+epoch).permutation(len(train_ids))
        remaining=[train_ids[int(i)] for i in permutation[offset:]]
        ds=Paintings(manifest,remaining,1024)
        loader=torch.utils.data.DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=a.workers,
              pin_memory=True,generator=torch.Generator().manual_seed(2026+epoch),
              **({'prefetch_factor':1} if a.workers else {}))
        for batch in loader:
            iteration_start=time.monotonic()
            G.train();D.train()
            b=len(batch)
            phases=[('Gmain',G,opts['G']),('Dmain',D,opts['D'])]
            if step%a.reg_interval==0:
                if a.gamma: phases.append(('Dreg',D,opts['D']))
                if a.pl_weight: phases.append(('Greg',G,opts['G']))
            for phase,net,opt in phases:
                net.requires_grad_(True);opt.zero_grad(set_to_none=True)
                # One uniformly selected image estimates each expensive regularizer.
                # Its expectation is unchanged; variance is higher than a full-batch estimate.
                regularizer=phase.endswith('reg')
                chunks=[batch[torch.randint(b,(1,)).item()].unsqueeze(0)] if regularizer else batch.split(a.microbatch)
                # Native fp16 convolution double backward is exceptionally slow
                # on the tested Blackwell runtime. FP32 regularizer passes use
                # the same weights/graph and avoid this slow path.
                precision_flags=[]
                if regularizer:
                    for block in net.modules():
                        if hasattr(block,'use_fp16'):
                            precision_flags.append((block,block.use_fp16));block.use_fp16=False
                for chunk in chunks:
                    real=chunk.cuda(non_blocking=True).float()/127.5-1
                    z=torch.randn(len(real),G.z_dim,device='cuda')
                    c=torch.zeros(len(real),G.c_dim,device='cuda')
                    gain=a.reg_interval if regularizer else len(real)/b
                    loss.accumulate_gradients(phase,real,c,z,c,True,gain)
                for block,flag in precision_flags: block.use_fp16=flag
                grads=[x.grad for x in net.parameters() if x.grad is not None]
                if not grads or not bool(torch.stack([torch.isfinite(x).all() for x in grads]).all()):
                    raise FloatingPointError(f'Invalid gradients in {phase}; keeping previous valid checkpoint. Do not continue blindly.')
                opt.step();opt.zero_grad(set_to_none=True);net.requires_grad_(False)
            with torch.no_grad():
                beta=0.5**(b/10000)  # EMA 10 kimg half-life; no ramp-up that discards pretraining.
                for pe,p in zip(ema.parameters(),G.parameters()): pe.lerp_(p,1-beta)
                for be,bg in zip(ema.buffers(),G.buffers()): be.copy_(bg)
            step+=1;nimg+=b;offset+=b
            stats.update();values=stats.as_dict()
            def mean(k): return float(values[k].mean) if k in values else 0.0
            # ADA, 0.6 target and 500kimg adjustment timescale.
            aug.p.copy_((aug.p+np.sign(mean('Loss/signs/real')-.6)*b/500000).clamp(0,1))
            elapsed=time.monotonic()-start
            training_seconds+=time.monotonic()-iteration_start
            row=dict(step=step,nimg=nimg,G=mean('Loss/G/loss'),D=mean('Loss/D/loss'),
                     real_score=mean('Loss/scores/real'),fake_score=mean('Loss/scores/fake'),
                     r1_penalty=mean('Loss/r1_penalty'),pl_penalty=mean('Loss/pl_penalty'),
                     augment_p=float(aug.p),seconds_per_image=training_seconds/max(nimg-start_nimg,1))
            new=not log_path.exists()
            with log_path.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields)
                if new:w.writeheader()
                w.writerow(row)
            if step%10==0 or step==start_step+1:
                eta=(target-nimg)*row['seconds_per_image']/3600
                timing='warming up' if step<start_step+10 else f'rough compute remaining={eta:.1f} h (plus I/O/evaluation)'
                print(f"step {step} | {nimg/1000:.2f}/{target/1000:.2f} kimg | G={row['G']:.3f} D={row['D']:.3f} | elapsed={elapsed/3600:.2f} h | {timing}",flush=True)
            if offset==len(train_ids): epoch+=1;offset=0
            # The first optimizer step proves sufficient quota for optimizer-inclusive recovery.
            if step==start_step+1 or time.monotonic()-last_save>=a.save_minutes*60:
                save('training')
            if nimg-last_eval_nimg>=a.eval_kimg*1000:
                validate();save('validated')
            if bad_evals>=2:reason='quality_guard';break
            if stop['requested']:reason='user_stop';break
            if elapsed>=a.max_hours*3600:reason='session_time_limit';break
            if a.max_steps and step-start_step>=a.max_steps:reason='local_diagnostic_limit';break
            if nimg>=target:break
        if reason!='target_reached' or nimg>=target:break
    save(reason)  # Save before any optional expensive evaluation.
    if nimg>last_eval_nimg and nimg>0 and not a.skip_validation:
        validate()
        if bad_evals>=2:reason='quality_guard'
        save(reason)
    print(f'Finished this session: {reason}. Same command resumes latest.pt; optimizer and data position are preserved.',flush=True)


if __name__=='__main__':
    main()
