"""Render the spatial and temporal evidence directly from the verified summary."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','axes.titlelocation':'left',
    'axes.grid':False,'pdf.fonttype':42})
teal,slate,rust,ink='#166b61','#62778b','#a45238','#23343d'
fig,axs=plt.subplots(2,2,figsize=(7.3,5.4),layout='constrained')
a,b,c,z=axs.ravel()
points={(v['phase'],v['seed']):v for v in d['robust'] if v['centering']=='train_only_by_b' and v['dose']==1}
for ix,phase in enumerate(('t10','t50')):
    for seed in range(3):
        row=points[phase,seed]
        x=3.85*ix+seed
        pct=lambda k:100*row[k]/row['original_wrong']
        a.vlines(x,pct('mean_wrong'),pct('max_wrong'),color=slate,lw=1.4)
        a.scatter(x,pct('max_wrong'),s=27,facecolors='white',edgecolors=slate,marker='s',zorder=3,
                  label='Largest wrong pairing' if ix==seed==0 else None)
        a.scatter(x,pct('mean_wrong'),s=22,color=slate,marker='_',zorder=3,
                  label='Mean of 58 wrong pairings' if ix==seed==0 else None)
        a.scatter(x,pct('own'),s=27,color=teal,zorder=4,label='Correct pairing' if ix==seed==0 else None)
a.set(title='a  Rescuing errors in H12',ylabel='Fraction of original errors rescued (%)')
a.set_xticks([1,4.85],['Sustained 10%','Sustained 50%'])
a.set_ylim(0,47);a.grid(axis='y',alpha=.2)
a.legend(loc='upper left',frameon=False,fontsize=7.2)
for ix,phase in enumerate(('t10','t50')):
    for seed in range(3):
        row=points[phase,seed]; x=3.85*ix+seed
        b.plot([x,x],[row['fixed_adv'],row['margin_adv']],color='#b8c4c9',lw=1.25)
        b.scatter(x,row['margin_adv'],color=teal,s=25,label='Highest wrong score' if ix==seed==0 else None)
        b.scatter(x,row['fixed_adv'],color=rust,s=24,marker='d',label='Frozen wrong class' if ix==seed==0 else None)
b.axhline(0,color='#9ca9ad',lw=.8)
b.set(title='b  Margin under two definitions',ylabel='Correct vs mean wrong pairing\nMargin difference (logit units)')
b.set_xticks([1,4.85],['Sustained 10%','Sustained 50%'])
b.legend(loc='center right',bbox_to_anchor=(.99,.65),frameon=False,fontsize=7.2)
geomlabels=['Moved add, H6','Moved add, H8','Moved add, H10','Subtract, H8','Subtract, H12','Multiply, H8','Multiply, H12']
groups=[('moved_add',6),('moved_add',8),('moved_add',10),('sub',8),('sub',12),('mul',8),('mul',12)]
g=np.array([[np.median([100*(v['native']-v[k])/2611 for v in d['geometry'] if (v['task'],v['H'])==t]) for k in ('all_L1','query')] for t in groups])
c.imshow(g,cmap='BuGn',vmin=0,vmax=100,aspect='auto')
c.set_title('c  Removing operand interactions')
c.set_xticks([0,1],['All layer-1 joint terms','Query interaction'])
c.set_yticks(range(len(groups)),geomlabels)
c.tick_params(length=0)
for i in range(g.shape[0]):
    for j in range(2):
        c.text(j,i,f'{round(g[i,j],1)+0.:.1f}',va='center',ha='center',color='white' if g[i,j]>62 else ink)
c.set_xlabel('Median accuracy drop (percentage points), three seeds',labelpad=6)
for i,(key,label) in enumerate([('moved_add','Moved add'),('sub','Subtract'),('mul','Multiply')]):
    for k,phase in enumerate(('before_first_10','first_65')):
        rs=[r for r in d['cross'] if r['phase']==phase and r['task']==key]
        for j,r in enumerate(rs):
            dx=(j-1)*.075
            z.scatter(i+(k-.5)*.38+dx,100*(r['own']-r['mean_wrong'])/r['original_wrong'],
                      color=slate if k==0 else teal,s=25,marker='o' if k==0 else '^',
                      label=('Before first 10%' if k==0 else 'Around first 65%') if i==j==0 else None)
z.axhline(0,c='#9ca9ad',lw=.8)
z.set(title='d  Rescue advantage across tasks',ylabel='Rescue gain over mean wrong pairing\n(percentage points of original errors)')
z.set_xticks(range(3),['Moved add','Subtract','Multiply'])
z.set_ylim(-.8,36)
z.legend(loc='upper left',frameon=False,fontsize=7.2)
for fmt in ('png','pdf'):
    kw={'dpi':350} if fmt=='png' else {'metadata':{'CreationDate':None,'ModDate':None}}
    fig.savefig(HERE/f'PNAS_checkpoint_evidence.{fmt}',**kw)
