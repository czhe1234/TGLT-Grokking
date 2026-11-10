from pathlib import Path
import json,hashlib,tempfile
import pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from fontTools.ttLib import TTFont
d=Path.cwd().parent
scan=d.parent/'firstgen_beta_wd_gray_continuation_20261102'
b=pd.read_csv(scan/'cohort_105.csv',keep_default_na=False)
labels=pd.read_csv(d/'continuation_outcomes/run_results.csv').set_index('run')
hs=json.loads((d/'all35_seed_means/inputs_sha256.json').read_text())
recs=[]
for z in b.itertuples():
 src=Path(z.previous_300k) if z.previous_300k else (scan/'continuation_300k'/z.run_id if z.status.startswith('continue_') else Path(z.parent_dir))
 f=src/'history.json';data=f.read_bytes();assert hashlib.sha256(data).hexdigest()==hs[str(f)];h=json.loads(data)
 end=h[-1]['step'];assert end==labels.loc[z.run_id,'last_recorded'] and set(range(0,end+1,500)).issubset(y['step'] for y in h)
 k=next((i for i,y in enumerate(h) if 10*y['test_correct']>=9*y['test_n']),None)
 t=w=np.nan
 if k is not None:
  t=h[k]['step'];j=next(i for i in range(k+1) if 10*h[i]['test_correct']>h[i]['test_n'])
  assert j>0;w=(t-h[j-1]['step'])/1000
 assert (pd.isna(t) and pd.isna(labels.loc[z.run_id,'first_test90'])) or t==labels.loc[z.run_id,'first_test90']
 assert end==300000 or (end==100000 and np.isfinite(w))
 recs.append((z.seed,z.wd,z.beta,end,t,w,h[-1]['test_acc']))
a=pd.DataFrame(recs,columns=['seed','wd','beta','end','t90','width_k','endpoint_acc'])
a.to_csv('width_300_by_seed.csv',index=False)
print('by seed: 100 reached, newly, ever, long, long endpoint90')
for seed,g in a.groupby('seed'):
 print(seed,(g.t90<=100000).sum(),(g.t90>100000).sum(),g.t90.count(),(g.end==300000).sum(),((g.end==300000)&(g.endpoint_acc>=.9)).sum())
cm=LinearSegmentedColormap.from_list('sea',['#073560','#277aa4','#7bbbd5','#e5f2f7']);cm.set_bad('#e8e9eb')
norm=Normalize(0,np.ceil(a.width_k.max()/10)*10)
with tempfile.TemporaryDirectory() as tmp:
 fonts={}
 for wt in ('Regular','Bold'):
  f=TTFont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+wt+'.ttc',fontNumber=2);path=tmp+'/'+wt+'.otf';f.save(path);fonts[wt]=FontProperties(fname=path)
 for seed,g in a.groupby('seed'):
  fig,ax=plt.subplots(figsize=(8.3,6.8));fig.subplots_adjust(left=.14,right=.98,top=.78,bottom=.20)
  fig.text(.04,.94,f'seed {seed}   |   截至 300k 步的首次跃迁宽度',fontsize=20,weight='bold')
  fig.text(.04,.87,'数字为千步。★ 表示 100k 后首次达到 90%；* 表示历史仅到 100k。',fontsize=11)
  rows=sorted(g.wd.unique());cols=sorted(g.beta.unique());z=g.pivot(index='wd',columns='beta',values='width_k').to_numpy()
  im=ax.imshow(z,cmap=cm,norm=norm,aspect='auto')
  for rr,r in enumerate(rows):
   for cc,c in enumerate(cols):
    v=g[(g.wd==r)&(g.beta==c)].iloc[0];w=v.width_k
    label='—' if pd.isna(w) else f'{w:g}'+('*' if v.end==100000 else (' ★' if v.t90>100000 else ''))
    ax.text(cc,rr,label,ha='center',va='center',fontsize=12,color='#82888a' if pd.isna(w) else ('white' if norm(w)<.4 else '#132d40'))
  ax.set_xticks(range(5),[f'{v:g}' for v in cols]);ax.set_yticks(range(7),[f'{v:.4g}' for v in rows]);ax.set_xlabel('耦合强度 β');ax.set_ylabel('权重衰减')
  ax.set_xticks(np.arange(.5,4.5),minor=True);ax.set_yticks(np.arange(.5,6.5),minor=True);ax.grid(which='minor',color='white',lw=1.3);ax.tick_params(which='both',length=0)
  cb=fig.colorbar(im,cax=fig.add_axes([.30,.066,.44,.025]),orientation='horizontal');cb.set_label('跃迁宽度（千步）')
  for o in fig.findobj(Text):
   sz=o.get_fontsize();o.set_fontproperties(fonts['Bold' if o.get_fontweight()=='bold' else 'Regular']);o.set_fontsize(sz)
  fig.savefig(f'width_300_seed{seed}.png',dpi=155);plt.close(fig)
