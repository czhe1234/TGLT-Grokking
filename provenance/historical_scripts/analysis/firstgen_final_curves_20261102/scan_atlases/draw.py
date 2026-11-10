from pathlib import Path
import tempfile,json,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from fontTools.ttLib import TTFont
from PIL import Image
d=Path(__file__).resolve().parent.parent
out=d/'scan_atlases'
a=pd.read_csv(d/'original_architecture_scan_192/runs_192.csv')
b=pd.read_csv(d.parent/'firstgen_beta_wd_gray_continuation_20261102/cohort_105.csv')
hashes=json.loads((d/'all35_seed_means/inputs_sha256.json').read_text())
def draw(beta,font):
 source=b if beta else a
 rows=sorted(set(b.wd)) if beta else sorted(set(zip(a.r,a['T'])))
 cols=sorted(set(b.beta)) if beta else sorted(set(a.M))
 nr=len(rows);nc=len(cols); assert (nr,nc)==((7,5) if beta else (16,4))
 fig=plt.figure(figsize=(25 if beta else 23, nr*1.16+2.0))
 left,right,bottom=.057,.99,.044 if beta else .031
 top=1-1.47/fig.get_figheight()
 sub=fig.add_gridspec(1,3,left=left,right=right,bottom=bottom,top=top,wspace=.12)
 fig.text(.025,1-.34/fig.get_figheight(),('耦合强度 × 权重衰减' if beta else '传播半径 × 传播轮数 × mode 数')+'   |   '+str(len(source))+' 条训练曲线',fontproperties=font,size=24)
 fig.text(.025,1-.69/fig.get_figheight(),'蓝线：训练准确率     橙线：测试准确率     浅橙底：测试首次完整跨过 10% 至 90%     横轴：训练步数（千步）     纵轴：准确率（%）',fontproperties=font,size=13)
 seen=set(); bands=0
 for seed in range(3):
  g=sub[seed].subgridspec(nr,nc,hspace=.20,wspace=.12)
  box=sub[seed].get_position(fig)
  fig.text((box.x0+box.x1)/2,top+.48/fig.get_figheight(),f'seed {seed}',ha='center',size=15,weight='bold',color='#123d62')
  for i,row in enumerate(rows):
   for j,col in enumerate(cols):
    ax=fig.add_subplot(g[i,j])
    if beta:
     z=b[(b.seed==seed)&(b.wd==row)&(b.beta==col)].iloc[0];f=Path(z.parent_dir)/'history.json'; sha=hashes[str(f)]
    else:
     z=a[(a.seed==seed)&(a.r==row[0])&(a['T']==row[1])&(a.M==col)].iloc[0];f=Path(z.history_path);sha=z.history_sha256
    assert z.run_id not in seen;seen.add(z.run_id)
    data=f.read_bytes();assert hashlib.sha256(data).hexdigest()==sha
    h=pd.DataFrame(json.loads(data));h=h[h.step<=100000]; assert set(range(0,100001,500)).issubset(h.step) and h.step.is_monotonic_increasing
    x=h.step.to_numpy()/1000;y=h.test_correct.to_numpy()/h.test_n.to_numpy();tr=h.train_correct.to_numpy()/h.train_n.to_numpy()
    assert np.allclose(y,h.test_acc,rtol=0,atol=1e-14) and np.allclose(tr,h.train_acc,rtol=0,atol=1e-14)
    hi=np.flatnonzero(y>=.9)
    if len(hi):
     k=hi[0]; first=np.flatnonzero(y[:k+1]>.10)[0];assert first>0 and y[first-1]<=.1
     ax.axvspan(x[first-1],x[k],color='#ffb940',alpha=.23,lw=0);bands+=1
    ax.plot(x,tr*100,color='#0069AB',lw=.8);ax.plot(x,y*100,color='#c8490e',lw=1.02)
    ax.set(xlim=(0,100),ylim=(-3,103),xticks=[0,50,100],yticks=[0,50,100]);ax.tick_params(labelsize=7,length=2,pad=1.5,labelbottom=i==nr-1,labelleft=j==0)
    ax.grid(color='#e6e8ea',lw=.45);ax.spines[['top','right']].set_visible(False)
    if i==0:ax.set_title(('β=' if beta else 'M=')+f'{col:g}',fontsize=11)
    if j==0 and seed==0:ax.set_ylabel(f'wd={row:.4g}' if beta else f'r={row[0]}\nT={row[1]}',rotation=0,ha='right',va='center',labelpad=12,size=10)
 assert len(seen)==len(source)
 name='beta_wd_105.png' if beta else 'architecture_192.png';fig.savefig(out/name,dpi=195);plt.close(fig)
 with Image.open(out/name) as img:
  print(name,img.size,bands,flush=True);img.thumbnail((1200,1300));img.save('/tmp/scan_preview_'+name)
with tempfile.TemporaryDirectory() as tmp:
 f=TTFont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2);assert 'CJK SC' in f['name'].getDebugName(1);path=tmp+'/SC.otf';f.save(path);font=FontProperties(fname=path)
 draw(True,font);draw(False,font)
