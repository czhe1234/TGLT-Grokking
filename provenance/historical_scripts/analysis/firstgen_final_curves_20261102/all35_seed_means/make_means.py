"""Plot all 35 beta/weight-decay cells. Each mean always contains all three seeds."""
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

out=Path(__file__).resolve().parent
scan=out.parents[1]/'firstgen_beta_wd_gray_continuation_20261102'
prev=out.parent/'three_seed_t50_300k'
sys.path.insert(0,str(scan))
from gray_common import csv_in, csv_out, read, assert_history, check_prefix, audit, sha

roster_path=scan/'cohort_105.csv'
labels_path=scan/'results/300k_runs.csv'
previous_table=out.parent/'continuation_outcomes/run_results.csv'
benchmark=prev/'exact50_means.csv.gz'
sources={str(p):sha(p) for p in (roster_path,labels_path,previous_table,benchmark)}
meta={r['run_id']:r for r in csv_in(roster_path)}
fixed={r['run_id']:r for r in csv_in(labels_path)}
last={r['run']:int(r['last_recorded']) for r in csv_in(previous_table)}
assert len(meta)==105 and meta.keys()==fixed.keys()==last.keys()
grouped=defaultdict(list)
for key,r in meta.items():
    base=Path(r['parent_dir'])
    if r['previous_300k']:
        src=Path(r['previous_300k'])
    elif r['status'].startswith('continue_'):
        src=scan/'continuation_300k'/key
    else:
        src=base
    histfile=src/'history.json'
    sources[str(histfile)]=sha(histfile)
    h=read(histfile)
    assert_history(h,last[key])
    if src!=base:
        parentsrc=base/'history.json'
        sources[str(parentsrc)]=sha(parentsrc)
        check_prefix(read(parentsrc),h)
    assigned=audit(h)
    assert assigned['audit_fixed_grokking']==(fixed[key]['strict_by_cut']=='True')
    t=np.array([z['step'] for z in h],dtype=float)
    train=np.array([z['train_correct']/z['train_n'] for z in h])
    test=np.array([z['test_correct']/z['test_n'] for z in h])
    np.testing.assert_allclose(train,[z['train_acc'] for z in h],rtol=0,atol=1e-14)
    np.testing.assert_allclose(test,[z['test_acc'] for z in h],rtol=0,atol=1e-14)
    reg=(t%500)==0
    crossings=np.flatnonzero([2*z['test_correct']>=z['test_n'] for z in h])
    center=None
    if len(crossings):
        j=int(crossings[0])
        assert j>0 and test[j-1]<.5<=test[j] and t[j]-t[j-1]==500
        center=float(t[j-1]+(.5-test[j-1])/(test[j]-test[j-1])*(t[j]-t[j-1]))
        assert abs(np.interp(center,t,test)-.5)<1e-12
    grouped[float(r['wd']),float(r['beta'])].append(
        dict(run=key,seed=int(r['seed']),t=t,reg=reg,train=train,test=test,t50=center,
             strict=assigned['audit_fixed_grokking']))
for tracks in grouped.values():
    tracks.sort(key=lambda x:x['seed'])
    assert [z['seed'] for z in tracks]==[0,1,2]
assert len(grouped)==35

def three_seed_mean(tracks,align):
    curves=[]
    for z in tracks:
        if align:
            assert z['t50'] is not None
            m=z['reg']
            x=z['t'][m]-z['t50']
            curves.append((x,z['train'][m],z['test'][m]))
        else:
            curves.append((z['t'],z['train'],z['test']))
    lo=max(v[0][0] for v in curves)
    hi=min(v[0][-1] for v in curves)
    assert lo<hi
    if align:
        assert lo<0<hi
        knots=np.unique(np.concatenate([v[0][(v[0]>=lo)&(v[0]<=hi)] for v in curves]+[[lo,0.,hi]]))
    else:
        knots=curves[0][0][curves[0][0]<=hi]
        assert all(np.array_equal(knots,v[0][v[0]<=hi]) for v in curves)
    assert knots[0]==lo and knots[-1]==hi
    result={'step':knots}
    for name,i in [('train',1),('test',2)]:
        data=np.array([np.interp(knots,v[0],v[i]) for v in curves])
        assert data.shape==(3,len(knots))
        mean=data.mean(axis=0);sd=data.std(axis=0,ddof=1)
        result[name]=mean
        result[name+'_sd']=sd
        np.testing.assert_allclose(sd,np.sqrt(np.sum((data-mean)**2,axis=0)/2),atol=1e-14,rtol=0)
        assert np.all((mean>=0)&(mean<=1)) and np.all(sd>=0)
        if align and name=='test':
            zero=np.flatnonzero(knots==0)[0]
            assert abs(mean[zero]-.5)<1e-12 and sd[zero]<1e-12
    return result

summaries=[]
mixed,raw={},{}
reached={}
for cfg,tracks in sorted(grouped.items()):
    n=sum(z['t50'] is not None for z in tracks)
    reached[cfg]=n
    raw[cfg]=three_seed_mean(tracks,False)
    mixed[cfg]=three_seed_mean(tracks,True) if n==3 else raw[cfg]
    d=dict(weight_decay=cfg[0],beta=cfg[1],number_of_seeds_averaged=3,seeds_reaching50=n,
           strict_grokking=sum(z['strict'] for z in tracks),axis='relative_to_first_50' if n==3 else 'raw_training_step',
           common_first_step=float(mixed[cfg]['step'][0]),common_last_step=float(mixed[cfg]['step'][-1]),
           raw_common_last=int(raw[cfg]['step'][-1]))
    for z in tracks:
        s=z['seed'];d[f'run_seed{s}']=z['run'];d[f't50_seed{s}']=z['t50']
    summaries.append(d)
assert sum(reached.values())==87
assert Counter(reached.values())=={3:26,2:3,1:3,0:3}

# Independent check on the nine previously published all-strict mean test curves.
ref=defaultdict(list)
with gzip.open(benchmark,'rt',newline='') as f:
    for row in csv.DictReader(f):
        if row['wd']!='all_three_seed_configs':
            ref[(float(row['wd']),float(row['beta']))].append((float(row['relative_step']),float(row['mean_test'])))
assert len(ref)==9
max_error=0.
for cfg,data in ref.items():
    assert reached[cfg]==3 and all(z['strict'] for z in grouped[cfg])
    m=mixed[cfg];a=np.array(data)
    test=np.interp(a[:,0],m['step'],m['test'])
    max_error=max(max_error,float(np.max(abs(test-a[:,1]))))
assert max_error<1e-12

csv_out(out/'cells.csv',summaries)
with gzip.open(out/'means.csv.gz','wt',newline='',encoding='utf-8') as f:
    writer=csv.writer(f); writer.writerow(['axis','weight_decay','beta','n','step','train_mean','train_sd','test_mean','test_sd'])
    for cfg in sorted(grouped):
        views=[('raw',raw[cfg])]+([('t50',mixed[cfg])] if reached[cfg]==3 else [])
        for label,m in views:
            writer.writerows((label,*cfg,3,*z) for z in zip(*(m[k] for k in ('step','train','train_sd','test','test_sd'))))

for v in ('Regular','Bold'):
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+v+'.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
                     'axes.edgecolor':'#a3adb4','xtick.color':'#384753','ytick.color':'#384753',
                     'path.simplify':False, 'savefig.facecolor':'white'})
blue='#146cac'; red='#cf2539'; ink='#192d3e'; sub='#46596b'
raw_face='#fbf5e8'; raw_edge='#bba777'
ws=sorted(set(w for w,b in grouped));bs=sorted(set(b for w,b in grouped))
assert len(ws)==7 and len(bs)==5

def draw(view):
    event_sheet=view=='event'
    fig,axs=plt.subplots(7,5,figsize=(15.1,13.35),sharex=False,sharey=True)
    fig.subplots_adjust(left=.108,right=.982,top=.852,bottom=.067,wspace=.15,hspace=.46)
    title='35 个配置的三种子平均曲线' if event_sheet else '35 个配置的三种子平均曲线 · 原始训练步数'
    fig.text(.04,.975,title,fontsize=21,weight='bold',color=ink,va='top')
    desc=('白底 26 格按测试准确率首次穿过 50% 对齐，竖线为零点。米色 9 格按原始训练步数平均。'
          if event_sheet else '全部使用原始训练步数。相同颜色、相同坐标和相同种子权重可直接比较各配置。')
    fig.text(.041,.939,desc,fontsize=11.0,color=sub,va='top')
    fig.legend(handles=[Line2D([],[],lw=2.35,color=blue,label='训练准确率均值'),
                        Line2D([],[],lw=2.35,color=red,label='测试准确率均值'),
                        Patch(facecolor='#a9bdd3',alpha=.4,edgecolor='none',label='浅色带：种子间 ±1 个标准差')],
               loc='upper left',bbox_to_anchor=(.035,.918),frameon=False,ncol=3,
               fontsize=10.2,handlelength=2.2,columnspacing=2.4)
    fig.text(.039,.870,'权重衰减',weight='bold',color=ink,fontsize=10.8,va='center')
    for i,w in enumerate(ws):
        box=axs[i,0].get_position();fig.text(.083,(box.y0+box.y1)/2,f'{w:.5g}',ha='right',va='center',fontsize=12.3,color=ink,weight='bold')
        for j,b in enumerate(bs):
            ax=axs[i,j];cfg=(w,b);k=reached[cfg];aligned=event_sheet and k==3
            fallback=event_sheet and not aligned
            m=mixed[cfg] if event_sheet else raw[cfg]
            ax.set_facecolor(raw_face if fallback else 'white')
            ax.set_axisbelow(True)
            ax.grid(axis='both',color='#cad2d7',linewidth=.5,alpha=.64)
            for sp in ax.spines.values():
                sp.set_linewidth(.64)
                if fallback: sp.set_color(raw_edge)
            ax.tick_params(axis='both',length=2.2,pad=1.9,labelsize=8.0)
            ax.set_ylim(-3,103);ax.set_yticks([0,50,100]);ax.tick_params(labelleft=(j==0))
            if aligned:
                ax.set_xlim(-30,30);ax.set_xticks([-30,0,30]);ax.set_xticklabels(['−30','0','+30'])
                ax.axvline(0,color='#737e89',lw=.81,zorder=2)
            else:
                ax.set_xlim(0,300);ax.set_xticks([0,100,200,300])
            x=m['step']/1000
            for name,col in [('train',blue),('test',red)]:
                y,sd=m[name],m[name+'_sd']
                ax.fill_between(x,100*np.maximum(y-sd,0),100*np.minimum(y+sd,1),color=col,alpha=.16,lw=0,zorder=1)
                ax.plot(x,100*y,c=col,lw=1.65,solid_capstyle='round',zorder=4)
            if i==0:ax.set_title(f'β = {b:g}',fontsize=12.5,color=ink,weight='bold',pad=15)
            if event_sheet:
                tag=(f'3/3 到过 50%' if aligned else f'原步数 · {k}/3 到过 50%')
                ax.text(.006,1.035,tag,transform=ax.transAxes,ha='left',va='bottom',fontsize=7.1,
                        color=('#4e6474' if aligned else '#785423'),clip_on=False)
    fig.text(.012,.47,'准确率（%）',rotation=90,ha='left',va='center',fontsize=12.1,color=ink)
    footer=('横轴单位为千步。每条均线始终包含三个 seed，只画到三者都有记录的位置。'
            if event_sheet else '训练步数（千步）  ·  每条均线始终包含三个 seed，只画到最后一个共同保存点。')
    fig.text(.535,.026,footer,ha='center',fontsize=10,color=sub)
    path=out/('means_35_t50.png' if event_sheet else 'means_35_raw.png')
    fig.savefig(path,dpi=172,facecolor='white');plt.close(fig)
    with Image.open(path) as im:im.verify()
    return path

figs=[draw('event'),draw('raw')]
assert all(sha(Path(path))==v for path,v in sources.items())
summary=dict(runs=len(meta),cells=len(grouped),seeds_per_mean=3,seed_counts_reached50=dict(sorted(Counter(reached.values()).items())),
             aligned_cells=26,raw_fallback_cells=9,seeds_reached50=87,
             source_policy='All 105 existing runs, irrespective of strict label. 78 reach 300k, 27 end at 100k.',
             center='First measured crossing of 50% in test accuracy. Estimate time by linear interpolation between the bracketing saves, keep the time scale.',
             aggregation='Piecewise-linear individual means within common observed support of all three; raw uses identical original steps. No lower-n tails.',
             description_of_band='Across three seeds, sample standard deviation (ddof=1), displayed within [0,1].',
             earlier_nine_max_abs_error=max_error,raw_horizons=dict(sorted(Counter(r['raw_common_last'] for r in summaries).items())),
             terminal_steps=dict(sorted(Counter(last.values()).items())),plots_only=True)
(out/'verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
(out/'inputs_sha256.json').write_text(json.dumps(sources,indent=2))
artifacts=figs+[out/n for n in ('cells.csv','means.csv.gz','verification.json','inputs_sha256.json','make_means.py')]
(out/'sha256.txt').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in artifacts))
print(json.dumps(summary,ensure_ascii=False,indent=2))

