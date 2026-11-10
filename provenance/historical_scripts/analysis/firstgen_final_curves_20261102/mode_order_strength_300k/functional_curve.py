"""Curve of the same endpoint ablation on the complete common event-centered time support."""
import os,sys
for name in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import io,json,math,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd,torch
import replay as R
O=Path(__file__).resolve().parent
ev=pd.read_csv(O/'strict_events.csv'); s0=pd.read_csv(O/'outcomes.csv').set_index('run')
cs=pd.read_csv(O.parent/'all35_seed_means/cells.csv')
anchors={}
for row in cs.itertuples():
    for s in range(3):
        k=getattr(row,'run_seed'+str(s))
        if k in set(ev.run):anchors[k]=getattr(row,'t50_seed'+str(s))
tmin=math.ceil(max(-10000,max(-a for a in anchors.values()))/500)*500
tmax=math.floor(min(10000,min(s0.at[k,'last']-a for k,a in anchors.items()))/500)*500
rel=np.arange(tmin,tmax+1,500)
reference=pd.read_csv(O/'trajectories.csv.gz',usecols=['run','step','test_acc','C_sd']).set_index(['run','step'])
ends=pd.read_csv(O/'mode_ablation_endpoints.csv').set_index(['run','step'])
def rms(u):return (u-u.mean(-1,keepdim=True)).square().mean().sqrt()
@torch.inference_mode()
def per_run(k):
    t0=anchors[k];lo=500*math.floor((t0+tmin)/500);hi=500*math.ceil((t0+tmax)/500)
    st,idx,h=R.state(k);cfg,m,tok,role,labels,te=st;source,ext=R.folders(k)
    points=[]
    for t in range(lo,hi+1,500):
        if (k,t) in ends.index:
            q=ends.loc[(k,t)]
            values=[q.native,q.best_single_raw,q.best_single_RMS]+[q['m'+str(j)+'_Csd'] for j in range(1,5)]
        else:
            path=source if t<=100000 else ext;raw=(path/f'checkpoint_{t:07d}.pt').read_bytes();receipt=R.read(path/f'receipt_{t:07d}.json')
            assert receipt['checkpoint_sha256']==hashlib.sha256(raw).hexdigest()
            cp=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);assert cp['step']==t;m.load_state_dict(cp['model_state'],strict=True)
            ff=R.eng.Forward(st);q,um=ff.end(permode=True);u=um.sum(1);target=rms(u)
            n=int(ff.score(ff.readout(q,u)).sum());solo=[];scaled=[];cov=[]
            for j in range(4):
                v=um[:,j];amp=rms(v);assert amp>0
                solo.append(int(ff.score(ff.readout(q,v)).sum()));scaled.append(int(ff.score(ff.readout(q,v*(target/amp))).sum()))
                cov.append(float(R.corr.continuous(v.numpy())[0].std()))
            values=[n,max(solo),max(scaled)]+cov
        entry=reference.loc[(k,t)]
        assert abs(values[0]-round(2611*entry.test_acc))<=3
        points.append([t,entry.test_acc,values[1]/2611,values[2]/2611,entry.C_sd]+values[3:])
    p=np.array(points);a=np.stack([np.interp(rel+t0,p[:,0],c) for c in p[:,1:].T],axis=1)
    assert abs(a[np.flatnonzero(rel==0)[0],0]-.5)<1e-12
    return k,a,len(points)
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=16) as ex:vals=list(ex.map(per_run,sorted(anchors)))
    assert len(vals)==58
    scores=np.stack([v[1] for v in vals])
    means=scores.mean(axis=0);sd=scores.std(axis=0,ddof=1)
    cols=['full','best_raw','best_rms','C_sd','m1','m2','m3','m4']
    df=pd.DataFrame(means,columns=cols);df.insert(0,'relative_step',rel)
    for j,k in enumerate(cols):df[k+'_sd']=sd[:,j]
    df['n']=len(vals);df.to_csv(O/'mode_function_aligned_means.csv',index=False)
    np.savez_compressed(O/'mode_function_aligned_individual_values.npz',runs=[r[0] for r in vals],relative_step=rel,columns=cols,values=scores)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    from PIL import Image
    for v in ('Regular','Bold'):font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+v+'.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10.6,4.7))
    fig.subplots_adjust(left=.082,right=.976,top=.715,bottom=.23,wspace=.22)
    ink='#183343';sub='#495d6c'
    fig.text(.05,.955,'58 条严格 Grokking 中，单个 mode 能否给出相同的预测？',fontsize=16.7,weight='bold',color=ink,va='top')
    fig.text(.05,.86,'按测试准确率首次穿过 50% 的时刻对齐。每条线始终平均全部 58 条运行。',fontsize=10.0,color=sub)
    x=rel/1000
    for ax in axs:
        ax.grid(axis='y',lw=.65,color='#dae2e7');ax.axvline(0,c='#8294a0',lw=.8);ax.set(xlim=(x[0],x[-1]),xlabel='相对训练步数（千步）')
    ax=axs[0]
    for k,col,l in [('m1','#0075a8','mode 1'),('m2','#d1650c','mode 2'),('m3','#008466','mode 3'),('m4','#8752a3','mode 4'),('C_sd','#182c39','四路合并')]:
        ax.plot(x,df[k],c=col,lw=2.4 if k=='C_sd' else 1.7,label=l)
    ax.set_title('实际写回的方向关联差异',weight='bold',fontsize=11)
    ax.legend(loc='upper left',frameon=True,framealpha=.86,ncol=2,fontsize=7.7)
    ax=axs[1]
    for k,col,l in [('full','#152e3d','完整模型'),('best_raw','#c25418','最佳单路'),('best_rms','#006e9c','等幅后的最佳单路')]:
        ax.plot(x,100*df[k],c=col,lw=2.25,label=l)
    ax.set_title('保留哪几路写回时能答对测试题？',weight='bold',fontsize=11);ax.set_ylabel('测试准确率（%）',fontsize=9);ax.set_ylim(0,101);ax.legend(loc='upper left',frameon=True,fontsize=8.2)
    fig.text(.52,.075,'第一层和读出保持原状。最佳单路取四路中测试准确率最高的一路。等幅按整张输入网格匹配写回的均方根。',ha='center',fontsize=8.6,color=sub)
    path=O/'58_mode_function_aligned.png';fig.savefig(path,dpi=170,facecolor='white');plt.close(fig)
    with Image.open(path) as im:im.verify()
    allp=sorted(O.glob('*.png'));(O/'figures.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in allp))
    print('n,real checkpoint evaluations inclusive of reused, real coverage',len(vals),sum(a[2] for a in vals),rel[[0,-1]],'t0',df[df.relative_step==0].to_dict('records')[0])

