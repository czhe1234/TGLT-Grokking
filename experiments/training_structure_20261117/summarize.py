"""Summarize saved checkpoint analyses and draw the nine paired trajectories."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont, fontManager

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE/'results/curves.csv').open()))
roster = json.loads((HERE/'roster.json').read_text())
fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
findfont(font, fallback_to_default=False)
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,
    'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})

summary = []
for run in roster:
    rr = sorted([r for r in rows if r['run_id'] == run['run_id']], key=lambda r:int(r['step']))
    base, end = rr[0], rr[-1]
    post_memory = [r for r in rr if int(r['step']) >= run['anchors']['t_mem']['logged_step']]
    increasing = [r for r in post_memory if float(r['delta_rule_fraction']) > 1e-10]
    record = dict(run_id=run['run_id'], window=run['window'], seed=run['seed'],
        strict_grokking=run['strict_grokking'], reached90=run['reached90'],
        start_rule_fraction=float(base['rule_fraction']), final_rule_fraction=float(end['rule_fraction']),
        final_test_acc=float(end['test_acc']), final_rule_energy=float(end['rule_energy']),
        final_within_energy=float(end['within_energy']), sampled_post_memory_points=len(post_memory),
        positive_rule_drift_points=len(increasing),
        positive_decay_attribution_points=sum(float(r['decay_delta_rule_fraction'])>1e-10 for r in post_memory),
        positive_gradient_attribution_points=sum(float(r['gradient_delta_rule_fraction'])>1e-10 for r in post_memory),
        positive_momentum_attribution_points=sum(float(r['momentum_delta_rule_fraction'])>1e-10 for r in post_memory))
    for milestone in [10000, 30000, 50000, 70000, 100000]:
        r = next(r for r in rr if int(r['step']) == milestone)
        record['R_at_%dk'%(milestone//1000)] = float(r['rule_fraction'])
        record['test_acc_at_%dk'%(milestone//1000)] = float(r['test_acc'])
    summary.append(record)
with (HERE/'results/run_summary.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
    writer.writeheader();writer.writerows(summary)

fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True,sharey=True)
for run in roster:
    ax=axes[[8,12,15].index(run['window']),run['seed']]
    rr=sorted([r for r in rows if r['run_id']==run['run_id']],key=lambda r:int(r['step']))
    history=json.loads((HERE.parents[1]/'experiments/transformer_window_bridge_20261115/histories'/(run['run_id']+'.json')).read_text())
    ax.plot([h['step']/1000 for h in history],[100*h['test_acc'] for h in history],color='#D55E00',lw=1.6,label='测试准确率')
    ax.plot([int(r['step'])/1000 for r in rr],[100*float(r['rule_fraction']) for r in rr],color='#0072B2',marker='o',ms=3,lw=1.6,label='规则份额 R')
    verdict='严格 Grokking' if run['strict_grokking'] else ('达到90%，非严格' if run['reached90'] else '未达90%')
    ax.set_title('H=%d，seed %d · %s'%(run['window'],run['seed'],verdict),fontsize=10)
    ax.set_xlim(0,100);ax.set_ylim(-2,102);ax.set_yticks([0,25,50,75,100]);ax.set_xticks([0,25,50,75,100])
    ax.grid(alpha=.17)
    if run['seed']==0:ax.set_ylabel('占比 / 准确率（%）')
    if run['window']==15:ax.set_xlabel('训练步数（千步）')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,frameon=False)
fig.suptitle('规则相关的内部表示怎样随训练形成',fontsize=15,y=.995)
fig.text(.5,.015,'R 使用全部输入与已知加法规则；圆点为真实分析检查点，连线不代表新增观测。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.04,1,.925))
fig.savefig(HERE/'results/representation_and_generalization.png',dpi=180)
plt.close(fig)

result=dict(runs=summary,
    interpretation='Observational trajectories plus fixed-vector one-step attribution; no long-horizon optimizer intervention.',
    curve_rows=len(rows),
    positive_decay_points=sum(r['positive_decay_attribution_points'] for r in summary),
    post_memory_points=sum(r['sampled_post_memory_points'] for r in summary))
(HERE/'results/summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
