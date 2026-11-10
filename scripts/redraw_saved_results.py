#!/usr/bin/env python3
"""Portable CPU redraw from packaged numbers. Does not open checkpoints or train."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):
    op=gzip.open if p.suffix=='.gz' else open
    with op(p,'rt',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['seed_means','ablation'],default='seed_means');args=ap.parse_args()
    out=ROOT/'reproduced';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    blue='#176da6';red='#cc2440';ink='#173344'
    if args.kind=='seed_means':
        base=ROOT/'results/all35_seed_means';cells=read(base/'cells.csv');records=read(base/'means.csv.gz')
        wd=sorted(set(float(r['weight_decay']) for r in cells));betas=sorted(set(float(r['beta']) for r in cells))
        fig,axs=plt.subplots(7,5,figsize=(20,21),sharey=True)
        fig.subplots_adjust(left=.075,right=.985,top=.935,bottom=.055,hspace=.5,wspace=.17)
        for c in cells:
            w,b=float(c['weight_decay']),float(c['beta']);ax=axs[wd.index(w),betas.index(b)]
            aligned=int(c['seeds_reaching50'])==3
            axis='t50' if aligned else 'raw'
            rows=[r for r in records if r['axis']==axis and float(r['weight_decay'])==w and float(r['beta'])==b]
            assert rows and all(int(r['n'])==3 for r in rows)
            x=np.array([float(r['step'])/1000 for r in rows]);aligned=int(c['seeds_reaching50'])==3
            if aligned:ax.axvline(0,c='#8294a0',lw=.8);ax.set_xlim(-30,30)
            else:ax.set_facecolor('#faf3e4');ax.set_xlim(0,float(c['raw_common_last'])/1000)
            for prefix,col in [('train',blue),('test',red)]:
                y=np.array([float(r[prefix+'_mean']) for r in rows])*100;s=np.array([float(r[prefix+'_sd']) for r in rows])*100
                ax.fill_between(x,np.maximum(y-s,0),np.minimum(y+s,100),color=col,alpha=.15,linewidth=0)
                ax.plot(x,y,color=col,lw=1.5,label=prefix)
            ax.set_ylim(-3,103);ax.set_yticks([0,50,100]);ax.grid(alpha=.2)
            ax.set_title('beta=%g | wd=%.6g\n%s | %s/3 cross 50%%'%(b,w,'t50 aligned' if aligned else 'original steps',c['seeds_reaching50']),fontsize=9)
        fig.suptitle('35 configurations: three-seed mean and sample standard deviation',fontsize=19,color=ink)
        fig.supxlabel('Training steps (thousands); white: relative to t50; beige: original steps')
        fig.supylabel('Accuracy (%)');axs[0,0].legend(fontsize=8)
        dest=out/'seed_means_35.png'
    else:
        r=read(ROOT/'results/mode_order_strength_300k/mode_function_aligned_means.csv')
        assert all(int(v['n'])==58 for v in r)
        x=np.array([float(v['relative_step']) for v in r])/1000
        fig,axs=plt.subplots(1,2,figsize=(12,4.5));fig.subplots_adjust(bottom=.22,wspace=.28)
        for key,col,label in [('m1',blue,'mode 1'),('m2','#d1650c','mode 2'),('m3','#008466','mode 3'),('m4','#8752a3','mode 4'),('C_sd',ink,'all modes')]:
            axs[0].plot(x,[float(v[key]) for v in r],c=col,label=label,lw=2)
        for key,col,label in [('full',ink,'Full model'),('best_raw','#c25418','Best single mode'),('best_rms',blue,'Best single mode, RMS matched')]:
            axs[1].plot(x,[100*float(v[key]) for v in r],c=col,label=label,lw=2)
        for ax in axs:ax.axvline(0,c='#8294a0',lw=.8);ax.grid(alpha=.2);ax.set_xlabel('Steps relative to t50 (thousands)');ax.legend(fontsize=8)
        axs[0].set_ylabel('Directional correlation standard deviation');axs[1].set_ylabel('Test accuracy (%)')
        fig.suptitle('58 runs; frozen second-layer single-mode interventions')
        dest=out/'single_mode_ablation_58.png'
    fig.savefig(dest,dpi=160);plt.close(fig);print(dest)
if __name__=='__main__':main()
