"""The 35-cell figures follow the archived three-seed alignment policy."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,math,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager,pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
out=Path(__file__).resolve().parent
for v in ('Regular','Bold'):font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+v+'.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'axes.spines.right':False,
  'axes.spines.top':False,'savefig.facecolor':'white','path.simplify':False})
ink='#192e40';sub='#4b6070';blue='#006aa2';red='#cd4d18';grid='#d9e1e6'; gray='#708597'
colors=['#0072ad','#d16108','#118367','#8550a2']
outcomes=pd.read_csv(out/'outcomes.csv').set_index('run')
d=pd.read_csv(out/'trajectories.csv.gz')
by={k:r.sort_values('step').reset_index(drop=True) for k,r in d.groupby('run')}
cells=pd.read_csv(out.parent/'all35_seed_means/cells.csv').sort_values(['weight_decay','beta'])
g=pd.read_csv(out/'matched_means.csv');matches=json.loads((out/'summary.json').read_text())['matched_runids']
ev=pd.read_csv(out/'strict_events.csv').set_index('run')
files=[]
def finish(f,name):
    path=out/name;f.savefig(path,dpi=170);plt.close(f)
    with Image.open(path) as im:im.verify()
    files.append(path)
def style(ax,rel=True):
    ax.grid(axis='y',color=grid,lw=.65);ax.tick_params(labelcolor=sub,color=sub,labelsize=8.7,length=3)
    for n in ('left','bottom'):ax.spines[n].set_color('#82909e');ax.spines[n].set_linewidth(.7)
    if rel:ax.axvline(0,color='#76828d',lw=.9,zorder=1)
def source_lines(series,metric):
    z=g[(g.series==series)&(g.metric==metric)].sort_values('relative_step');assert (z['size']==6).all()
    return z.relative_step.to_numpy()/1000,z['mean'].to_numpy(),z['std'].to_numpy()
def title(fig,main,subtitle):
    fig.text(.061,.957,main,ha='left',va='top',weight='bold',color=ink,size=18.5)
    fig.text(.062,.897,subtitle,ha='left',va='top',color=sub,size=10.1)
f,axs=plt.subplots(2,2,sharex=True,figsize=(10.8,7.05))
f.subplots_adjust(left=.10,right=.97,bottom=.135,top=.775,wspace=.23,hspace=.37)
title(f,'同一配置下，哪些变化随泛化出现？','蓝线平均 6 条严格 Grokking。橙线使用同一配置下到 300k 仍未达到 90% 的运行，并对齐到蓝线相同的训练时段。')
f.legend(handles=[Line2D([],[],color=blue,lw=2.5,label='严格 Grokking'),
 Line2D([],[],color=red,lw=2.5,label='同配置未达到 90%'),Patch(facecolor='#b5c5cf',alpha=.5,label='浅色为对应均值 ±1 个标准差')],
 loc='upper left',bbox_to_anchor=(.058,.854),ncol=3,frameon=False,fontsize=9.1)
for ax,(metric,ttl,factor) in zip(axs.flat,[('test_acc','测试准确率（%）',100),('C_sd','总写回的方向关联差异',1),
        ('A','四个 mode 的总写回强度',1),('D','按正确答案分组的规则写回强度',1)]):
    style(ax)
    for group,col in [('strict',blue),('control',red)]:
        x,y,sd=source_lines(group,metric)
        lo=np.maximum(0,y-sd);hi=np.minimum(1,y+sd) if metric=='test_acc' else y+sd
        ax.fill_between(x,lo*factor,hi*factor,color=col,alpha=.14,lw=0)
        ax.plot(x,y*factor,lw=2.25,color=col)
    ax.set_title(ttl,color=ink,fontsize=11.1,weight='bold',pad=5)
    ax.set_xlim(-30,30);ax.set_xticks([-30,-15,0,15,30]);ax.set_ylim(bottom=0)
    if metric=='test_acc':ax.set_ylim(0,102)
f.text(.53,.069,'距成功运行首次穿过 50% 的训练步数（千步）',ha='center',color=ink,fontsize=10.9)
f.text(.53,.023,'共有 8 条不同的对照运行、9 个配对。一个成功运行有多个对照时，先平均这些对照，再对 6 个窗口等权平均。',ha='center',color=sub,fontsize=8.4)
finish(f,'matched_strength_and_order.png')

# Shares are measured from the actual norm, not inferred from the gate.
f,axs=plt.subplots(1,2,sharex=True,sharey=True,figsize=(10.0,4.5))
f.subplots_adjust(left=.08,right=.97,top=.70,bottom=.20,wspace=.18)
title(f,'四个 mode 分别承担了多少写回？','每个模型先计算四条写回的强度份额，再按与上一张图相同的 6 个窗口取平均。')
f.legend(handles=[Line2D([],[],c=c,lw=2.2,label=f'mode {i+1}') for i,c in enumerate(colors)],
 loc='upper left',bbox_to_anchor=(.055,.849),frameon=False,ncol=4,fontsize=9.2)
for ax,kind,name in zip(axs,['strict','control'],['严格 Grokking','同配置未达到 90%']):
    style(ax)
    for n,col in enumerate(colors,1):
        x,y,_=source_lines(kind,f'q{n}');ax.plot(x,y*100,lw=2.2,c=col)
    ax.set(xlim=(-30,30),ylim=(0,100),xticks=[-30,-15,0,15,30],yticks=[0,25,50,75,100])
    ax.set_title(name,weight='bold',fontsize=11,color=ink)
axs[0].set_ylabel('写回强度份额（%）',fontsize=10)
f.text(.52,.07,'距成功运行首次穿过 50% 的训练步数（千步）',ha='center',color=ink,fontsize=10.4)
finish(f,'matched_four_mode_shares.png')

# All 35 cells. Each cell always averages the same three seeds, including raw-time fallbacks.
weights=sorted(cells.weight_decay.unique());betas=sorted(cells.beta.unique())
def cellcurves(c,cols):
    series=[by[getattr(c,f'run_seed{s}')] for s in range(3)]
    rel=c.seeds_reaching50==3
    ts=[z.step.to_numpy(dtype=float)-(getattr(c,f't50_seed{s}') if rel else 0) for s,z in enumerate(series)]
    lo=max(a[0] for a in ts);hi=min(a[-1] for a in ts)
    t=np.unique(np.concatenate([a[(a>=lo)&(a<=hi)] for a in ts]+[[lo,hi,0.]]))
    assert min(t)>=lo and max(t)<=hi
    means={k:np.array([np.interp(t,x,z[k]) for z,x in zip(series,ts)]) for k in cols}
    return rel,t/1000,means
def make_grid(kind,cols,caption):
    f,axs=plt.subplots(7,5,sharey=(kind!='strength'),figsize=(13.6,11.65))
    f.subplots_adjust(left=.108,right=.984,top=.84,bottom=.088,hspace=.50,wspace=.17)
    f.text(.042,.964,caption,weight='bold',size=18,color=ink,va='top')
    f.text(.042,.922,'每格三个 seed 等权。26 个白底格按各自首次测试 50% 平移。9 个米色格按原始训练步数平均。',size=9.9,color=sub)
    if kind=='shares':
        f.legend(handles=[Line2D([],[],c=c,lw=2.0,label=f'mode {i+1}') for i,c in enumerate(colors)],
          loc='upper left',bbox_to_anchor=(.038,.911),frameon=False,ncol=4,fontsize=9.1)
    else:
        f.legend(handles=[Line2D([],[],c=blue,lw=2.1,label='三个 seed 均值'),Patch(facecolor=blue,alpha=.16,label='种子间 ±1 个标准差')],
          loc='upper left',bbox_to_anchor=(.038,.911),frameon=False,ncol=2,fontsize=9.2)
    f.text(.035,.855,'权重衰减',fontsize=9.4,color=ink,weight='bold')
    largest=0
    for i,w in enumerate(weights):
        bb=axs[i,0].get_position();f.text(.08,(bb.y0+bb.y1)/2,f'{w:.5g}',ha='right',va='center',size=10.7,weight='bold',color=ink)
        for j,b in enumerate(betas):
            c=next(v for v in cells.itertuples() if abs(v.weight_decay-w)<1e-9 and v.beta==b)
            rel,x,vals=cellcurves(c,cols)
            ax=axs[i,j];style(ax,rel);ax.tick_params(length=2,pad=1.5,labelsize=7.2)
            if not rel:ax.set_facecolor('#fbf4e5')
            if i==0:ax.set_title(f'β = {b:g}',color=ink,weight='bold',fontsize=11.7,pad=14)
            ax.text(.01,1.025,'50% 对齐' if rel else '原始训练步数',transform=ax.transAxes,ha='left',va='bottom',color=sub,fontsize=7.1)
            if rel:
                ax.set_xlim(-30,30);ax.set_xticks([-30,0,30]);visible=(x>=-30)&(x<=30)
            else:
                ax.set_xlim(0,300);ax.set_xticks([0,100,200,300]);visible=(x>=0)&(x<=300)
            for z,k in enumerate(cols):
                mat=vals[k];mean=mat.mean(0);sd=mat.std(0,ddof=1)
                if kind=='shares':ax.plot(x,100*mean,c=colors[z],lw=1.27)
                else:
                    ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=blue,alpha=.16,lw=0)
                    ax.plot(x,mean,color=blue,lw=1.55)
                    largest=max(largest,float((mean+sd)[visible].max()))
                    if kind=='strength':
                        ax.set_ylim(0,max(.001,float((mean+sd)[visible].max())*1.05))
            if kind=='shares':ax.set_ylim(-2,102);ax.set_yticks([0,50,100])
    if kind=='direction':
        axs[0,0].set_ylim(0,math.ceil(largest*100)/100)
    f.text(.52,.035,'横轴单位为千步。每格只画到三个 seed 的共同记录边界。'+(' 强度使用各格独立纵轴。' if kind=='strength' else ''),ha='center',color=sub,fontsize=9.2)
    finish(f,'all35_'+kind+'.png')
make_grid('direction',['C_sd'],'总写回的方向关联差异 · 全部 35 个配置')
make_grid('strength',['A'],'四个 mode 的总写回强度 · 全部 35 个配置')
make_grid('shares',['q1','q2','q3','q4'],'四个 mode 的写回强度份额（%）· 全部 35 个配置')
(out/'figures.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
print([p.name for p in files])

