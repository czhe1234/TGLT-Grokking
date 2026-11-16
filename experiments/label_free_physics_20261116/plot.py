"""Plot saved label-free observables; no model weights or targets needed."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':12,'axes.titlesize':13,
    'xtick.labelsize':10,'ytick.labelsize':10,'savefig.dpi':170})
COLORS={0:'#0066AD',1:'#D64025',2:'#00856A'}
METRICS=['G1_0q','G1_0b','G2_0q','G2_0b','chi_mean','S_total','D_total','q34']


def load():
    rows=list(csv.DictReader((HERE/'curves.csv').open()))
    grouped={}
    for row in rows:
        d={k:(v if k=='run_id' else float(v) if v else np.nan) for k,v in row.items()}
        grouped.setdefault(row['run_id'],[]).append(d)
    return {k:sorted(v,key=lambda r:r['step']) for k,v in grouped.items()}


def descriptors(span,seed):
    if span==15:return 'seed 1 · 后续 Grokking' if seed==1 else 'seed 0 · 100k 未达90%'
    return 'seed 0 · 后续宽跃迁' if seed==0 else 'seed %d · 后续 Grokking'%seed


def panel(ax,data,metric,span):
    for rid,rows in data.items():
        if int(rows[0]['span'])!=span:continue
        seed=int(rows[0]['seed'])
        ax.plot([r['step']/1000 for r in rows],[r[metric] for r in rows],color=COLORS[seed],lw=2.2,label=descriptors(span,seed))
    ax.set_xlim(0,40);ax.set_xticks([0,10,20,30,40]);ax.grid(True,color='#D4DDE4',alpha=.7,lw=.75)
    ax.set_xlabel('训练步数（千步）');ax.legend(frameon=True,fontsize=9,loc='best',framealpha=.92)


def shared_limits(axes):
    lower=min(a.get_ylim()[0] for a in axes);upper=max(a.get_ylim()[1] for a in axes)
    for ax in axes:ax.set_ylim(lower,upper)


def finish(fig,name,title,subtitle):
    fig.suptitle(title,x=.06,y=.985,ha='left',fontsize=20,fontweight='bold',color='#172F40')
    fig.text(.06,.985-.46/fig.get_size_inches()[1],subtitle,ha='left',va='top',fontsize=11,color='#435D70')
    fig.text(.06,.015,'全部3,481组输入；不使用答案标签。实线连接实际保存点；原跨度每1,000步，新跨度每500步。',fontsize=10,color='#435D70')
    fig.savefig(HERE/'results'/name,facecolor='white');plt.close(fig)


def make_plots(data):
    (HERE/'results').mkdir(exist_ok=True)
    spans=[15,11]
    fig,axes=plt.subplots(4,2,figsize=(15,14));fig.subplots_adjust(top=.875,bottom=.075,hspace=.52,wspace=.2)
    specs=[('G1_0q','第一层 · 端到端'),('G1_0b','第一层 · 操作数位置之间'),('G2_0q','第二层 · 端到端'),('G2_0b','第二层 · 操作数位置之间')]
    for i,(metric,title) in enumerate(specs):
        for j,span in enumerate(spans):
            ax=axes[i,j];panel(ax,data,metric,span);ax.set_ylabel('归一化关联')
            pair='0 → %d'%(span if metric.endswith('q') else 7 if span==15 else 5)
            ax.set_title('%s  |  %s'%(title,pair),loc='left')
        shared_limits(axes[i])
    finish(fig,'01_空间两点关联.png','空间两点关联：远处的状态是否共同变化？','左列：原跨度15，含成功/未泛化对照。右列：跨度11，三个seed后来都达90%。')
    fig,axes=plt.subplots(2,2,figsize=(15,9));fig.subplots_adjust(top=.84,bottom=.1,hspace=.48,wspace=.2)
    for i,(metric,label) in enumerate([('chi_mean','逐输入响应范数的平均'),('chi_rms','逐输入响应范数的均方根')]):
        for j,span in enumerate(spans):
            panel(axes[i,j],data,metric,span);axes[i,j].set_ylabel('输入响应强度')
            axes[i,j].set_title('跨度%d · %s'%(span,label),loc='left')
        shared_limits(axes[i])
    finish(fig,'02_输入到查询的线性响应.png','线性响应：输入端的微小变化能传多强？','精确64×64响应矩阵，保留两层LayerNorm导数；第一层端到端响应因几何不可达而严格为0。')
    fig,axes=plt.subplots(2,2,figsize=(15,9));fig.subplots_adjust(top=.84,bottom=.1,hspace=.48,wspace=.2)
    for i,(metric,label) in enumerate([('S_total','写回谱熵（自然对数）'),('D_total','有效参与维数')]):
        for j,span in enumerate(spans):
            panel(axes[i,j],data,metric,span);axes[i,j].set_ylabel(label)
            axes[i,j].set_title('跨度%d · %s'%(span,label),loc='left')
        shared_limits(axes[i])
    finish(fig,'03_写回谱熵与有效参与率.png','写回协方差谱：变化集中在几个集体方向？','第二层查询位，四个mode的实际写回相加后跨输入中心化；低值表示谱更集中。')
    fig,axes=plt.subplots(1,2,figsize=(15,5.6));fig.subplots_adjust(top=.73,bottom=.17,wspace=.2)
    for j,span in enumerate(spans):
        panel(axes[j],data,'q34',span);axes[j].set_ylabel('中心化归一化重叠')
        axes[j].set_title('跨度%d · Mode 3 与 Mode 4'%span,loc='left')
        axes[j].axhline(0,color='#596B74',lw=.85,zorder=0)
    shared_limits(axes)
    finish(fig,'04_Mode3与Mode4写回重叠.png','通道重叠：Mode 3 与 Mode 4 是否共同起伏？','第二层查询位的实际写回；正值为同向共同变化，负值为反向共同变化。')


def episode(steps,values,threshold,until_end=False):
    signs=np.sign(values);good=np.abs(values)>=threshold
    for i,s in enumerate(steps):
        end=len(steps)-1 if until_end else np.searchsorted(steps,s+5000)
        if end>=len(steps) or steps[end]-s<5000:continue
        if good[i:end+1].all() and (signs[i:end+1]==signs[i]).all():return int(s)
    return None


def summarize(data):
    success=data['r2_T4_M4_seed1'];failure=data['r2_T4_M4_seed0']
    assert [r['step'] for r in success]==[r['step'] for r in failure]
    steps=np.array([r['step'] for r in success],dtype=int);out=[]
    for metric in METRICS:
        a=np.array([r[metric] for r in success]);b=np.array([r[metric] for r in failure])
        if metric in ['chi_mean','D_total']:
            gap=np.log(a/b)-np.log(a[0]/b[0]);threshold=math.log(1.5 if metric=='chi_mean' else 1.25);unit='log relative ratio'
        else:
            gap=(a-b)-(a[0]-b[0]);threshold=.2 if metric=='S_total' else .1;unit='baseline-adjusted raw difference'
        out.append(dict(metric=metric,threshold=threshold,comparison_unit=unit,
            first_5k_episode=episode(steps,gap,threshold),first_until_40k=episode(steps,gap,threshold,True),
            success_0=float(a[0]),failure_0=float(b[0]),success_40k=float(a[-1]),failure_40k=float(b[-1]),
            success_min=float(a.min()),success_max=float(a.max()),failure_min=float(b.min()),failure_max=float(b.max()),
            baseline_adjusted_gap_40k=float(gap[-1]),not_a_significance_test=True))
    with (HERE/'results/separation_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
    compact={rid:{str(int(r['step'])):{m:r[m] for m in METRICS} for r in rows if r['step'] in [0,10000,20000,30000,40000]} for rid,rows in data.items()}
    (HERE/'results/key_values.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2))
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    data=load();make_plots(data);summarize(data)
