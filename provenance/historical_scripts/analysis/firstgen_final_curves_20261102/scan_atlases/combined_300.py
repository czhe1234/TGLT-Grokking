from pathlib import Path
import tempfile
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from fontTools.ttLib import TTFont
a=pd.read_csv('width_300_by_seed.csv')
assert len(a)==105 and not a.duplicated(['seed','wd','beta']).any()
colors=['#211341','#4b338b','#1161a0','#1597a2','#95d1aa','#ffe59a']; cm=ListedColormap(colors);cm.set_bad('#d3d4d6')
norm=BoundaryNorm([0,10,20,40,60,100,max(101,a.width_k.max()+1)],6)
fig,axes=plt.subplots(1,3,figsize=(17,7.4))
fig.subplots_adjust(left=.058,right=.988,bottom=.23,top=.77,wspace=.21)
fig.text(.036,.925,'耦合强度 × 权重衰减  |  三个种子的首次跃迁宽度',fontsize=22,weight='bold')
fig.text(.037,.84,'截至 300k 步。格内数字为千步。★ 表示 100k 后首次达到 90%；* 表示已在 100k 内达到 90%，历史止于 100k。',fontsize=12)
for seed,ax in enumerate(axes):
 g=a[a.seed==seed];rows=sorted(g.wd.unique());cols=sorted(g.beta.unique());ws=g.pivot(index='wd',columns='beta',values='width_k')
 n=g.width_k.count();assert n==[29,27,27][seed]
 ax.imshow(ws,cmap=cm,norm=norm,aspect='auto')
 ax.set_title(f'seed {seed}    ·    {n}/35 组达到 90%',fontsize=14,weight='bold',pad=10)
 for i,r in enumerate(rows):
  for j,c in enumerate(cols):
   v=g[(g.wd==r)&(g.beta==c)].iloc[0];w=v.width_k
   label='—' if np.isnan(w) else f'{w:g}'+('*' if v.end==100000 else (' ★' if v.t90>100000 else ''))
   ax.text(j,i,label,va='center',ha='center',fontsize=11,color='#484c52' if np.isnan(w) else ('white' if norm(w)<3 else '#102128'))
 ax.set_xticks(range(5),[f'{c:g}' for c in cols]);ax.set_yticks(range(7),[f'{r:.4g}' for r in rows]);ax.set_xlabel('耦合强度 β')
 if seed==0:ax.set_ylabel('权重衰减')
 ax.set_xticks(np.arange(.5,4.5),minor=True);ax.set_yticks(np.arange(.5,6.5),minor=True);ax.grid(which='minor',color='white',lw=1.3);ax.tick_params(which='both',length=0)
fig.legend([Patch(color=c) for c in colors+['#d3d4d6']],['<10','10–<20','20–<40','40–<60','60–<100','≥100','到 300k 仍未达到'],loc='lower center',bbox_to_anchor=(.5,.027),ncol=7,title='10% 至 90% 的跃迁宽度（千步）')
with tempfile.TemporaryDirectory() as tmp:
 fonts={}
 for wt in ('Regular','Bold'):
  f=TTFont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+wt+'.ttc',fontNumber=2);path=tmp+'/'+wt+'.otf';f.save(path);fonts[wt]=FontProperties(fname=path)
 for o in fig.findobj(Text):
  sz=o.get_fontsize();o.set_fontproperties(fonts['Bold' if o.get_fontweight()=='bold' else 'Regular']);o.set_fontsize(sz)
 fig.savefig('beta_wd_300_three_seeds.png',dpi=165)
