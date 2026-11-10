from pathlib import Path
import json,hashlib,tempfile
import pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from fontTools.ttLib import TTFont
a=pd.read_csv('../original_architecture_scan_192/runs_192.csv').query('H>=8')
cols=sorted(set(zip(a.r,a['T'])),key=lambda x:(x[0]*x[1],x))
colors=['#211341','#4b338b','#1161a0','#1597a2','#95d1aa','#ffe59a']
cm=ListedColormap(colors);cm.set_bad('#d3d4d6');norm=BoundaryNorm([0,10,20,40,60,100,110],6)
fig,axes=plt.subplots(3,1,figsize=(14.5,10.7));fig.subplots_adjust(left=.078,right=.98,top=.855,bottom=.135,hspace=.62)
fig.text(.035,.945,'传播半径扫描  |  三个种子的测试准确率跃迁宽度',size=22,weight='bold')
fig.text(.035,.90,'0–100k 步  ·  每格一个 seed，数字为 10% 至 90% 的宽度（千步）。深紫最快，灰格表示未达到 90%。',size=12)
for seed,ax in enumerate(axes):
 z=np.full((4,11),np.nan)
 for v in a[a.seed==seed].itertuples():
  b=Path(v.history_path).read_bytes();assert hashlib.sha256(b).hexdigest()==v.history_sha256
  h=json.loads(b);assert h[-1]['step']==100000
  hi=next((i for i,q in enumerate(h) if 10*q['test_correct']>=9*q['test_n']),None)
  if hi is not None:
   j=next(i for i in range(hi+1) if 10*h[i]['test_correct']>h[i]['test_n'])
   assert j>0
   z[[1,2,4,8].index(v.M),cols.index((v.r,v.T))]=(h[hi]['step']-h[j-1]['step'])/1000
 n=np.isfinite(z).sum();assert n==[36,32,38][seed]
 ax.imshow(z,cmap=cm,norm=norm,aspect='auto')
 ax.set_title(f'seed {seed}     ·     {n}/44 组达到 90%',loc='left',fontsize=13,weight='bold')
 for i,j in np.ndindex(z.shape):
  x=z[i,j];ax.text(j,i,'—' if np.isnan(x) else f'{x:g}',va='center',ha='center',fontsize=11,color='#45494c' if np.isnan(x) else ('white' if norm(x)<3 else '#102128'))
 ax.set_xticks(range(11),[f'H={r*t}\n({r}, {t})' for r,t in cols]);ax.set_yticks(range(4),['M=1','M=2','M=4','M=8']);ax.set_xticks(np.arange(.5,10.5),minor=True);ax.set_yticks(np.arange(.5,3.5),minor=True);ax.grid(which='minor',color='white',lw=1.2);ax.tick_params(which='both',length=0)
axes[-1].set_xlabel('每列给出传播范围 H = r × T，以及（传播半径 r，传播轮数 T）')
fig.legend([Patch(color=x) for x in colors+['#d3d4d6']],['<10','10–<20','20–<40','40–<60','60–<100','≥100','未达到'],loc='lower center',bbox_to_anchor=(.5,.01),ncol=7,title='跃迁宽度（千步）')
with tempfile.TemporaryDirectory() as t:
 fonts={}
 for w in ('Bold','Regular'):
  f=TTFont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+w+'.ttc',fontNumber=2);path=t+'/'+w+'.otf';f.save(path);fonts[w]=FontProperties(fname=path)
 for o in fig.findobj(Text):
  sz=o.get_fontsize();o.set_fontproperties(fonts['Bold' if o.get_fontweight()=='bold' else 'Regular']);o.set_fontsize(sz)
 fig.savefig('arch_width_three_seeds.png',dpi=165)
