"""Chinese figures and fixed-three-seed summaries from the original histories."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,csv,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import mechanism as a
OLD=Path(json.loads((B/'config.json').read_text())['baseline_directory']);O=B/'results';O.mkdir(exist_ok=True)
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,
 'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':2.1,'savefig.facecolor':'white'})
BLUE='#0072B2';RED='#D62728';BLACK='#192F3B';C=[BLUE,'#D55E00','#00845A','#7B3294']
GROUPS=['TGLT','T0','T1','T2'];LABEL={'TGLT':'局域多 mode','T0':'带 FFN · wd=0.1','T1':'带 FFN · wd=1.0','T2':'无 FFN · wd=0.1'}
PARAMS={'TGLT':62787,'T0':109051,'T1':109051,'T2':42619}

def source_roster():
 rows=[dict(r,group='TGLT',steps=100000) for r in a.baseline_roster()]
 for w in [8,12,15]:
  for s in range(3):
   rid='transformer_W%02d_s%d'%(w,s);rows.append(dict(kind='Transformer',group='T0',window=w,seed=s,run_id=rid,path=str(OLD/'runs'/rid),steps=100000))
 return rows+a.transformer_roster()
def collect_saved():
 rows=json.loads((B/'results/run_summary.json').read_text())
 assert len(rows)==36 and all(r['complete'] for r in rows)
 hist={r['run_id']:pd.DataFrame(json.loads((B/'histories'/(r['run_id']+'.json')).read_text())) for r in rows}
 return rows,hist

def collect():
 rows=[];hist={};raw_new=[]
 for row in source_roster():
  hp=Path(row['path'])/'history.json'
  if not hp.exists():continue
  h=json.loads(hp.read_text());r=dict(row,**a.classify(h));r.update(complete=h[-1]['step']==row['steps'],parameter_count=PARAMS[row['group']],history_sha256=a.sha(hp),final_at_least90=h[-1]['test_acc']>=.9)
  at100=next((x for x in h if x['step']==100000),None);r['test_at100k']=None if at100 is None else at100['test_acc'];r['train_at100k']=None if at100 is None else at100['train_acc']
  rows.append(r);hist[row['run_id']]=pd.DataFrame(h)
  if row['group'] in ['T1','T2']:
   raw_new.extend([dict(run_id=row['run_id'],group=row['group'],window=row['window'],seed=row['seed'],**v) for v in h])
 a.write_csv(rows,O/'per_run_summary.csv');a.json_write(rows,O/'run_summary.json');a.write_csv(raw_new,O/'18条完整训练记录.csv')
 grouped=[]
 for group in GROUPS:
  for w in [8,12,15]:
   rs=[r for r in rows if r['group']==group and r['window']==w]
   if len(rs)!=3 or not all(r['complete'] for r in rs):continue
   widths=[r['width_10_90'] for r in rs if r['width_10_90'] is not None]
   grouped.append(dict(group=group,window=w,n=3,steps=rs[0]['steps'],parameter_count=PARAMS[group],
    reached90=sum(r['reached90'] for r in rs),final_at_least90=sum(r['final_at_least90'] for r in rs),strict_grokking=sum(r['audit_fixed_grokking'] for r in rs),
    width_n=len(widths),width_median=None if not widths else float(np.median(widths)),width_min=None if not widths else min(widths),width_max=None if not widths else max(widths),
    final_train_mean=np.mean([r['final_train_acc'] for r in rs]),final_test_mean=np.mean([r['final_test_acc'] for r in rs]),final_test_std=np.std([r['final_test_acc'] for r in rs]),test_at100k_mean=np.mean([r['test_at100k'] for r in rs])))
 a.write_csv(grouped,O/'grouped_summary.csv')
 a.write_csv([{'运行':r['run_id'],'组别':LABEL[r['group']],'窗口':r['window'],'种子':r['seed'],'训练预算_步':r['steps'],'实际终点_步':r['final_step'],'首次训练99%_步':r['t_mem'],'首次测试10%_步':r['t10'],'首次测试50%_步':r['t50'],'首次测试90%_步':r['t90'],'10到90宽度_步':r['width_10_90'],'按原判据严格Grokking':r['audit_fixed_grokking'],'记忆后到90%前最低训练准确率%':None if r['train_min_mem_to_t90'] is None else 100*r['train_min_mem_to_t90'],'该区间训练低于99%的保存点数':r['train_below99_points_mem_to_t90'],'100k测试准确率%':None if r['test_at100k'] is None else 100*r['test_at100k'],'终点测试准确率%':100*r['final_test_acc'],'终点训练准确率%':100*r['final_train_acc'],'终点达到90%':r['final_at_least90'],'达90后最低测试准确率%':None if r['test_min_after_t90'] is None else 100*r['test_min_after_t90'],'达90后低于90%的保存点数':r['test_below90_points_after_t90']} for r in rows if r['group'] in ['T1','T2']],O/'逐运行汇总.csv')
 cn=[]
 for g in grouped:
  cn.append({'模型':LABEL[g['group']],'H或W':g['window'],'种子数':3,'训练预算':g['steps'],'参数量':g['parameter_count'],
   '曾达到90%条数':g['reached90'],'终点达到90%条数':g['final_at_least90'],'严格Grokking条数':g['strict_grokking'],
   '100k测试均值%':100*g['test_at100k_mean'],'终点训练均值%':100*g['final_train_mean'],'终点测试均值%':100*g['final_test_mean'],'终点测试标准差_百分点':100*g['final_test_std'],
   '完整10到90跃迁条数':g['width_n'],'10到90宽度中位数_步':g['width_median'],'宽度最小_步':g['width_min'],'宽度最大_步':g['width_max']})
 a.write_csv(cn,O/'配置汇总.csv');return rows,hist

def setup(ax,limit=150,ylim=None):
 ax.grid(alpha=.18);ax.set_xlim(0,limit);ax.set_xticks([0,25,50,75,100,125,150] if limit==150 else [0,25,50,75,100])
 if ylim is not None:ax.set_ylim(*ylim)
def budget_shade(ax,budget):
 if budget==100000:
  ax.axvspan(100,150,color='#F1F3F5',zorder=-3)
  ax.text(125,8,'该组训练至100k',ha='center',va='center',fontsize=8.5,color='#555555')
def shade(ax,row):
 if row['t10'] is not None and row['t90'] is not None:ax.axvspan(row['t10']/1000,row['t90']/1000,color='#F5B943',alpha=.2,zorder=0)
def save(fig,name,top=.9,bottom=.08):
 fig.tight_layout(rect=[.015,bottom,.995,top]);fig.savefig(O/name,dpi=180,bbox_inches='tight');plt.close(fig)
def accuracy(rows,hist):
 fig,axs=plt.subplots(6,3,figsize=(17,18),sharex=True,sharey=True)
 for row in rows:
  if row['group'] not in ['T1','T2']:continue
  i=(0 if row['group']=='T1' else 3)+[8,12,15].index(row['window']);ax=axs[i,row['seed']];h=hist[row['run_id']]
  ax.plot(h.step/1000,h.train_acc*100,color=BLUE);ax.plot(h.step/1000,h.test_acc*100,color=RED);shade(ax,row);setup(ax,ylim=(-2,103));budget_shade(ax,row['steps'])
  tag='原判据严格 Grokking' if row['audit_fixed_grokking'] else ('达到 90%' if row['reached90'] else '未达 90%')
  ax.set_title('%s · W=%d · seed %d\n%s；当前 %gk'%(LABEL[row['group']],row['window'],row['seed'],tag,h.step.iloc[-1]/1000),fontsize=10.5)
 for ax in axs[:,0]:ax.set_ylabel('准确率（%）')
 for ax in axs[-1]:ax.set_xlabel('训练步数（千步）')
 fig.suptitle('Transformer 补充对照：全部 18 条训练与测试曲线',fontsize=21,weight='bold',y=.995,color=BLACK)
 fig.legend(handles=[Line2D([0],[0],color=BLUE,label='训练准确率'),Line2D([0],[0],color=RED,label='测试准确率')],loc='upper center',bbox_to_anchor=(.5,.971),ncol=2,frameon=False)
 fig.text(.5,.013,'前3排：FFN256、wd=1.0、150k；后3排：无FFN、wd=0.1、100k。浅橙色仅标首次完整10%→90%区间。',ha='center',fontsize=11)
 save(fig,'01_全部18条训练测试曲线.png',top=.944,bottom=.05)
def means(rows,hist):
 fig,axs=plt.subplots(3,4,figsize=(22,10),sharex=True,sharey=True);agg=[]
 for i,w in enumerate([8,12,15]):
  for j,g in enumerate(GROUPS):
   ax=axs[i,j];rs=sorted([r for r in rows if r['group']==g and r['window']==w],key=lambda r:r['seed'])
   if len(rs)!=3:continue
   frames=[hist[r['run_id']].set_index('step') for r in rs];steps=sorted(set.intersection(*(set(t.index) for t in frames)))
   for key,c in [('train_acc',BLUE),('test_acc',RED)]:
    vals=np.array([t.loc[steps,key].values for t in frames])*100;mu,sd=vals.mean(0),vals.std(0)
    ax.plot(np.array(steps)/1000,mu,color=c);ax.fill_between(np.array(steps)/1000,np.clip(mu-sd,0,100),np.clip(mu+sd,0,100),color=c,alpha=.18)
    for step,m,s in zip(steps,mu,sd):agg.append(dict(group=g,window=w,step=step,metric=key,mean_percent=m,std_percent=s,n=3))
   setup(ax,ylim=(-2,103));budget_shade(ax,rs[0]['steps']);ax.set_title('%s · %s=%d\n曾达90%% %d/3；终点≥90%% %d/3；原判据严格G %d/3'%(LABEL[g],'H' if g=='TGLT' else 'W',w,sum(r['reached90'] for r in rs),sum(r['final_at_least90'] for r in rs),sum(r['audit_fixed_grokking'] for r in rs)),fontsize=10.5)
 for ax in axs[:,0]:ax.set_ylabel('准确率（%）')
 for ax in axs[-1]:ax.set_xlabel('训练步数（千步）')
 fig.suptitle('局域多 mode、原 Transformer、强正则与无 FFN：三种子平均',fontsize=21,weight='bold',y=.995,color=BLACK)
 fig.legend(handles=[Line2D([0],[0],color=BLUE,label='训练均值'),Line2D([0],[0],color=RED,label='测试均值')],loc='upper center',bbox_to_anchor=(.5,.952),ncol=2,frameon=False)
 fig.text(.5,.014,'每个均值点固定包含3个seed，阴影为±1标准差。TGLT/原Transformer/无FFN训练到100k；强正则组到150k。参数量分别62,787 / 109,051 / 42,619。\n严格G沿用原判据；逐运行汇总同时列出记忆后的训练跌落和达90%后的测试回落。',ha='center',fontsize=11)
 save(fig,'02_四组模型均值对照.png',top=.89,bottom=.08);a.write_csv(agg,O/'mean_std_curves.csv')

def mechanisms(rows,hist):
 if not (B/'mechanism/completion.json').exists():return
 a.aggregate();df=pd.read_csv(B/'mechanism/curves.csv');ep=pd.read_csv(B/'mechanism/endpoint.csv');pos=pd.read_csv(B/'mechanism/positions.csv')
 completion=json.loads((B/'mechanism/completion.json').read_text());selected=completion['selected']
 a.write_csv([dict(模型=LABEL[r.get('group','TGLT')],窗口=r['window'],种子=r['seed'],终点步数=r.get('steps',100000),运行=r['run_id']) for r in selected],O/'机制分析纳入清单.csv')
 for w in [8,12,15]:
  for group in ['T1','T2']:
   rs=[next(r for r in rows if r['run_id']==v['run_id']) for v in selected if v.get('group')==group and v['window']==w]
   if not rs:continue
   fig,axs=plt.subplots(3,len(rs),figsize=(5.7*len(rs),10),squeeze=False,sharex=True,sharey='row')
   for col,r in enumerate(rs):
    z=df[df.run_id==r['run_id']].sort_values('step');h=hist[r['run_id']]
    axs[0,col].plot(h.step/1000,h.train_acc*100,color=BLUE);axs[0,col].plot(h.step/1000,h.test_acc*100,color=RED)
    axs[0,col].set_title('seed %d · %s'%(r['seed'],'原判据严格 Grokking' if r['audit_fixed_grokking'] else '终点达到90%'))
    for k in range(4):
     axs[1,col].plot(z.step/1000,z['C_fixed_final%d'%(k+1)],color=C[k])
     axs[2,col].plot(z.step/1000,z['q%d'%(k+1)]*100,color=C[k])
    for ri in range(3):setup(axs[ri,col]);shade(axs[ri,col],r)
    axs[0,col].set_ylim(-2,103);axs[1,col].axhline(0,color='#777',lw=.8);axs[2,col].set_ylim(-2,103);axs[2,col].set_xlabel('训练步数（千步）')
   for ri,label in enumerate(['训练/测试准确率（%）','条件判别贡献','写回强度份额（%）']):axs[ri,0].set_ylabel(label)
   fig.suptitle('%s · W=%d：全部终点达标运行（%d/3）'%(LABEL[group],w,len(rs)),fontsize=19,weight='bold',y=.995)
   fig.legend(handles=[Line2D([0],[0],color=C[k],label='Head %d'%(k+1)) for k in range(4)],loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,frameon=False)
   fig.text(.5,.014,'蓝色训练、红色测试；贡献固定各自终点最强错误类，并冻结原模型LayerNorm分母。带FFN时另冻结GELU乘性因子。',ha='center',fontsize=9.5)
   save(fig,'03_%s_W%02d_全部达标Head轨迹.png'%(group,w),top=.90,bottom=.08)
 # One common-scale endpoint table; no averaging successful-only into a full-group denominator.
 conds=['native']+['L2_only_%d'%k for k in range(1,5)]+['L2_none'];mats=[];labs=[]
 for g in ['TGLT','T1','T2']:
  for w in [8,12,15]:
   ids=[r['run_id'] for r in selected if r.get('group','TGLT')==g and r['window']==w];z=ep[ep.run_id.isin(ids)]
   if not ids:continue
   mats.append([z[z.condition==c].test_acc.mean()*100 for c in conds]);labs.append('%s · %s=%d · n=%d'%(LABEL[g],'H' if g=='TGLT' else 'W',w,len(ids)))
 fig,ax=plt.subplots(figsize=(15,max(4.5,len(labs)*.55+2)));mat=np.array(mats);im=ax.imshow(mat,vmin=0,vmax=100,cmap='Blues',aspect='auto')
 ax.set_xticks(range(6));ax.set_xticklabels(['完整模型','仅第1路','仅第2路','仅第3路','仅第4路','关闭第二层写回'])
 ax.set_yticks(range(len(labs)));ax.set_yticklabels(labs,fontsize=10)
 for i in range(len(labs)):
  for j in range(6):ax.text(j,i,'%.2f%%'%mat[i,j],ha='center',va='center',color='white' if mat[i,j]>55 else 'black',fontsize=11)
 fig.colorbar(im,ax=ax,label='测试准确率（%）',shrink=.8);ax.set_title('终点仅保留第二层单 Mode / Head：真实消融',pad=15,fontsize=18,weight='bold')
 fig.text(.5,.015,'TGLT包含全部3seed；Transformer按用户要求仅纳入终点≥90%的运行，逐行标出n。第一层完整，下游LayerNorm及FFN按干预后状态重算。',ha='center',fontsize=10)
 save(fig,'04_终点单路消融对照.png',top=.93,bottom=.12)
 relay=pos.groupby(['run_id','kind','window','seed','step','phase','source'],as_index=False).agg(relay_share=('relay_share','first'),whole_input_change_rms=('whole_input_change_rms','first'))
 relay.to_csv(O/'relay_summary.csv',index=False);ep.to_csv(O/'endpoint_summary.csv',index=False)
 # Show every selected run's three phases and both changed operands.
 for w in [8,12,15]:
  rs=[r for r in selected if r['window']==w];fig,axs=plt.subplots(len(rs),2,figsize=(15,2.15*len(rs)+2.2),squeeze=False,sharex=True)
  for i,r in enumerate(rs):
   for j,src in enumerate(['a','b']):
    z=pos[(pos.run_id==r['run_id'])&(pos.source==src)];ph=['before','during','after'];mat=np.stack([z[z.phase==p].sort_values('position').influence_share.values*100 for p in ph]);steps=[int(z[z.phase==p].step.iloc[0]) for p in ph]
    ax=axs[i,j];im=ax.imshow(mat,vmin=0,vmax=100,cmap='Blues',aspect='auto');ax.set_yticks(range(3));ax.set_yticklabels(['%gk'%(v/1000) for v in steps]);ax.set_xticks(range(16))
    ax.set_title('%s · seed %d · 改变%s'%(LABEL[r.get('group','TGLT')],r['seed'],src),fontsize=10)
    for k in range(3):
     for l in range(16):
      if mat[k,l]>=5:ax.text(l,k,'%.0f'%mat[k,l],ha='center',va='center',fontsize=8,color='white' if mat[k,l]>55 else 'black')
  for ax in axs[-1]:ax.set_xlabel('替换的第1层位置')
  fig.suptitle('H / W = %d：第一层位置反事实替换'%w,fontsize=20,weight='bold',y=.995)
  fig.text(.5,.018,'每次只替换一个位置，再实际计算后续预测。颜色为干预强度归一化占比，不是可加因果贡献。达标运行取自身跃迁前、中及终点；未达标TGLT取固定对照时刻。',ha='center',fontsize=10)
  fig.tight_layout(rect=[.01,.06,.92,.96]);ca=fig.add_axes([.94,.17,.013,.66]);fig.colorbar(im,cax=ca,label='单位置影响占比（%）');fig.savefig(O/('05_W%02d_第1层位置替换.png'%w),dpi=180,bbox_inches='tight');plt.close(fig)

def paired_mechanisms(rows,hist):
 if not (B/'mechanism/completion.json').exists():return
 completion=json.loads((B/'mechanism/completion.json').read_text());selected=completion['selected']
 df=pd.read_csv(B/'mechanism/curves.csv');phase=pd.read_csv(B/'mechanism/phases.csv');signs=[]
 for row in selected:
  z=phase[phase.run_id==row['run_id']];pre=z[z.phase=='before'].iloc[0];mid=z[z.phase=='during'].iloc[0];post=z[z.phase=='after'].iloc[0]
  for k in range(1,5):
   key='C_fixed_final%d'%k
   signs.append(dict(run_id=row['run_id'],group=row.get('group','TGLT'),window=row['window'],seed=row['seed'],channel=k,
    before_step=int(pre.step),during_step=int(mid.step),after_step=int(post.step),before_C=pre[key],during_C=mid[key],after_C=post[key],
    before_q=pre['q%d'%k],during_q=mid['q%d'%k],after_q=post['q%d'%k],negative_to_positive=bool(pre[key]<0<post[key])))
 a.write_csv(signs,O/'逐通道三阶段贡献及符号变化.csv')
 for w in [8,12,15]:
  for g in ['T1','T2']:
   successes=[r for r in selected if r.get('group')==g and r['window']==w]
   if not successes:continue
   # Success selections are fixed by endpoint accuracy, not by the observed head behavior.
   for tf in successes:
    tg=next(r for r in rows if r['group']=='TGLT' and r['window']==w and r['seed']==tf['seed'])
    fig,axs=plt.subplots(3,2,figsize=(13,10),sharex=True,sharey='row')
    for col,r in enumerate([tg,tf]):
     h=hist[r['run_id']];z=df[df.run_id==r['run_id']].sort_values('step');lab=next(x for x in rows if x['run_id']==r['run_id'])
     axs[0,col].plot(h.step/1000,h.train_acc*100,color=BLUE);axs[0,col].plot(h.step/1000,h.test_acc*100,color=RED)
     for k in range(4):
      axs[1,col].plot(z.step/1000,z['C_fixed_final%d'%(k+1)],color=C[k]);axs[2,col].plot(z.step/1000,z['q%d'%(k+1)]*100,color=C[k])
     axs[0,col].set_title('%s · %s=%d · seed %d'%(LABEL[r.get('group','TGLT')],'H' if col==0 else 'W',w,r['seed']))
     for ri in range(3):setup(axs[ri,col]);shade(axs[ri,col],lab);budget_shade(axs[ri,col],r.get('steps',100000))
     axs[0,col].set_ylim(-2,103);axs[1,col].axhline(0,color='#888',lw=.8);axs[2,col].set_ylim(-2,103);axs[2,col].set_xlabel('训练步数（千步）')
    for ri,label in enumerate(['准确率（%）','条件判别贡献','写回强度份额（%）']):axs[ri,0].set_ylabel(label)
    fig.suptitle('同窗口、同seed：Mode 与 Attention Head 的完整轨迹',fontsize=19,weight='bold',y=.995)
    fig.legend(handles=[Line2D([0],[0],color=C[k],label='Mode / Head %d'%(k+1)) for k in range(4)],loc='upper center',bbox_to_anchor=(.5,.953),ncol=4,frameon=False)
    fig.text(.5,.015,'上排蓝色训练、红色测试；浅橙色是首次完整10%→90%。贡献相对于各自终点的固定错误类，分解口径与前一轮一致。\n相同seed不代表相同内部通道语义；条件贡献的符号变化需与真实消融结果一起解释。',ha='center',fontsize=10)
    save(fig,'06_%s_W%02d_s%d_Mode与Head并排.png'%(g,w,tf['seed']),top=.90,bottom=.10)

def per_seed_ablation():
 if not (B/'mechanism/completion.json').exists():return
 completion=json.loads((B/'mechanism/completion.json').read_text());ep=pd.read_csv(B/'mechanism/endpoint.csv')
 rs=sorted(completion['selected'],key=lambda r:(r['window'],['TGLT','T1','T2'].index(r.get('group','TGLT')),r['seed']))
 conds=['native']+['L2_only_%d'%k for k in range(1,5)]+['L2_none'];mat=[];labs=[]
 for r in rs:
  z=ep[ep.run_id==r['run_id']];mat.append([z[z.condition==c].test_acc.iloc[0]*100 for c in conds]);labs.append('%s · %s=%d · seed %d'%(LABEL[r.get('group','TGLT')],'H' if r['kind']=='TGLT' else 'W',r['window'],r['seed']))
 mat=np.array(mat);fig,ax=plt.subplots(figsize=(15,len(rs)*.50+2.4));im=ax.imshow(mat,vmin=0,vmax=100,cmap='Blues',aspect='auto')
 ax.set_xticks(range(6));ax.set_xticklabels(['完整模型','仅第1路','仅第2路','仅第3路','仅第4路','关闭第二层写回']);ax.set_yticks(range(len(labs)));ax.set_yticklabels(labs,fontsize=10)
 for i in range(len(labs)):
  for j in range(6):ax.text(j,i,'%.2f%%'%mat[i,j],ha='center',va='center',color='white' if mat[i,j]>55 else 'black',fontsize=10.5)
 fig.colorbar(im,ax=ax,label='测试准确率（%）',shrink=.8);ax.set_title('每个seed分别看：终点单 Mode / Head 消融',fontsize=18,pad=18,weight='bold')
 fig.text(.5,.015,'第一层保持完整，只改变第二层的写回通道；下游运算全部重算。TGLT全部seed保留；Transformer仅纳入终点达到90%的全部运行。',ha='center',fontsize=10)
 save(fig,'07_逐seed单通道消融.png',top=.94,bottom=.08)

def main():
 global O
 p=argparse.ArgumentParser();p.add_argument('--training-only',action='store_true');p.add_argument('--saved',action='store_true',help='Redraw from archived histories/CSV without model checkpoints.');p.add_argument('--output',type=Path);args=p.parse_args()
 rows,hist=collect_saved() if args.saved else collect()
 if args.output is not None:O=args.output.resolve()
 elif args.saved:O=B.parents[1]/'reproduced'/B.name
 O.mkdir(parents=True,exist_ok=True)
 accuracy(rows,hist);means(rows,hist)
 if not args.training_only:
  mechanisms(rows,hist);paired_mechanisms(rows,hist);per_seed_ablation()
 print(json.dumps({'runs':len(rows),'complete':sum(r['complete'] for r in rows),'figures':[p.name for p in sorted(O.glob('*.png'))]},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
