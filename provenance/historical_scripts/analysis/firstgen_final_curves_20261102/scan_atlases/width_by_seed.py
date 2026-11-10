import tempfile,json,itertools,hashlib
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap,Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
import pandas as pd,numpy as np
from fontTools.ttLib import TTFont
d=Path.cwd().parent
hashes=json.loads((d/'all35_seed_means/inputs_sha256.json').read_text())
both=[]
seedgrids=[]
for beta in (True,False):
 a=(pd.read_csv(d.parent/'firstgen_beta_wd_gray_continuation_20261102/cohort_105.csv') if beta else pd.read_csv(d/'original_architecture_scan_192/runs_192.csv').query('H>=8'))
 records=[]
 for z in a.itertuples():
  f=Path(z.parent_dir)/'history.json' if beta else Path(z.history_path)
  data=f.read_bytes();assert hashlib.sha256(data).hexdigest()==(hashes[str(f)] if beta else z.history_sha256)
  h=[y for y in json.loads(data) if y['step']<=100000]
  k=next((i for i,y in enumerate(h) if 10*y['test_correct']>=9*y['test_n']),None)
  w=np.nan
  if k is not None:
   j=next(i for i in range(k+1) if 10*h[i]['test_correct']>h[i]['test_n'])
   assert j>0
   w=(h[k]['step']-h[j-1]['step'])/1000
  records.append((z.seed,z.wd if beta else z.M,z.beta if beta else (z.r,z.T),w))
 df=pd.DataFrame(records,columns=['seed','row','col','w'])
 rows=sorted(set(df.row));cols=sorted(set(df.col),key=None if beta else lambda x:(x[0]*x[1],x))
 matrices=[np.empty((len(rows),len(cols))) for i in range(2)]
 for i,r in enumerate(rows):
  for j,c in enumerate(cols):
   vals=[w for seed,rr,cc,w in records if rr==r and cc==c]
   assert len(vals)==3
   vals=[v for v in vals if np.isfinite(v)];matrices[0][i,j]=np.median(vals) if vals else np.nan;matrices[1][i,j]=len(vals)
 assert matrices[1].sum()==(50 if beta else 106)
 for seed in range(3):
  g=df[df.seed==seed].pivot(index="row",columns="col",values="w").reindex(index=rows,columns=cols).to_numpy(dtype=float)
  seedgrids.append((beta,seed,rows,cols,g))
 both.append((beta,rows,cols,*matrices))
cm=LinearSegmentedColormap.from_list('sea',['#073560','#277aa4','#7bbbd5','#e5f2f7']);cm.set_bad('#e8e9eb')
norm=Normalize(0,np.ceil(max(np.nanmax(x[-1]) for x in seedgrids)/10)*10)
with tempfile.TemporaryDirectory() as tmp:
 fonts={}
 for wt in ('Regular','Bold'):
  f=TTFont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+wt+'.ttc',fontNumber=2);assert 'CJK SC' in f['name'].getDebugName(1);path=tmp+'/'+wt+'.otf';f.save(path);fonts[wt]=FontProperties(fname=path)
 for seed in range(3):
  fig,axes=plt.subplots(1,2,figsize=(16.8,6.4),gridspec_kw={'width_ratios':[5,10]});fig.subplots_adjust(left=.06,right=.987,top=.77,bottom=.28,wspace=.17)
  fig.text(.04,.925,f'seed {seed}  |  测试准确率首次从 10% 升到 90% 的宽度',fontsize=21,weight='bold')
  fig.text(.04,.85,'数字单位为千步。颜色越深，跃迁越快。灰格表示到 100k 步仍未达到 90%。',fontsize=12)
  for ax,(beta,_,rows,cols,z) in zip(axes,[x for x in seedgrids if x[1]==seed]):
   im=ax.imshow(z,cmap=cm,norm=norm,aspect='auto')
   n=int(np.isfinite(z).sum());print(seed,beta,n)
   ax.set_title(('耦合强度 × 权重衰减' if beta else '传播范围 × mode 数')+f'    达到 90%：{n}/{z.size}',fontsize=13)
   for i,j in np.ndindex(z.shape):
    k=z[i,j];ax.text(j,i,'—' if np.isnan(k) else f'{k:g}',ha='center',va='center',size=11,color='#888' if np.isnan(k) else ('white' if norm(k)<.40 else '#132d40'))
   ax.set_yticks(range(len(rows)),[f'{v:.4g}' if beta else f'M={v}' for v in rows]);ax.set_ylabel('权重衰减' if beta else 'mode 数')
   ax.set_xticks(range(len(cols)),[f'{c:g}' if beta else f'H={c[0]*c[1]}\n({c[0]}, {c[1]})' for c in cols]);ax.set_xlabel('耦合强度 β' if beta else '传播范围 H，以及 (r, T)')
   ax.set_xticks(np.arange(.5,len(cols)-.5),minor=True);ax.set_yticks(np.arange(.5,len(rows)-.5),minor=True);ax.grid(which='minor',color='white',lw=1.2);ax.tick_params(which='both',length=0)
  cb=fig.colorbar(im,cax=fig.add_axes([.32,.095,.42,.03]),orientation='horizontal');cb.set_label('跃迁宽度（千步）')
  for o in fig.findobj(Text):
   sz=o.get_fontsize();o.set_fontproperties(fonts['Bold' if o.get_fontweight()=='bold' else 'Regular']);o.set_fontsize(sz)
  fig.savefig(f'width_both_seed{seed}.png',dpi=155);plt.close(fig)
