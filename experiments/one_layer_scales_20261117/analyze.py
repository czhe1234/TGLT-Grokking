import csv
import json
import math
import argparse
import sqlite3
import subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from common import ROOT, CFG, make_model, sha, dump


def analyze():
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    data = torch.load(ROOT/'data.pt', map_location='cpu', weights_only=False)
    summaries, curves, checks, trajectories = [], [], {}, {}
    for group in CFG['groups']:
        trajectories[group] = []
        for seed in CFG['seeds']:
            rid = group+'_s'+str(seed)
            run = ROOT/'runs'/rid
            status = json.loads((run/'status.json').read_text())
            assert status['state'] == 'complete' and status['step'] == CFG['steps'], rid
            history = json.loads((run/'history.json').read_text())
            assert [r['step'] for r in history] == list(range(0,100001,500)), rid
            checkpoints = sorted(run.glob('checkpoint_*.pt'))
            assert [int(p.stem.split('_')[-1]) for p in checkpoints] == list(range(0,100001,500)), rid
            for expected_step, checkpoint_path in zip(range(0,100001,500),checkpoints):
                saved = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                assert saved['step'] == expected_step
                assert saved['data_sha256'] == sha(ROOT/'data.pt')
                assert saved['config']['group'] == group and saved['config']['seed'] == seed
                assert saved['history'] == history[:expected_step//500+1]
                assert saved['model_state'] and all(torch.isfinite(v).all() for v in saved['model_state'].values())
                assert saved['rng_state'] and saved['optimizer_state']['state']
                assert set(int(v['step']) for v in saved['optimizer_state']['state'].values()) == {expected_step}
            cp = saved
            assert cp['step'] == 100000 and cp['history'] == history
            assert cp['data_sha256'] == sha(ROOT/'data.pt')
            optsteps = [int(v['step']) for v in cp['optimizer_state']['state'].values()]
            assert optsteps and set(optsteps) == {100000}
            assert cp['rng_state']
            model = make_model(group,seed)
            model.load_state_dict(cp['model_state'])
            model.eval()
            errors = {}
            with torch.no_grad():
                for name in ['train','test_familiar','seen_new','test_new']:
                    ds = data[name]
                    logits = torch.cat([model(ds['x'][i:i+1024]) for i in range(0,len(ds['y']),1024)])
                    count = int((logits.argmax(1)==ds['y']).sum())
                    assert count == history[-1][name+'_correct'], (rid,name,count)
                    errors[name] = abs(float(torch.nn.functional.cross_entropy(logits,ds['y']))-history[-1][name+'_ce'])
                    assert math.isfinite(errors[name]) and errors[name] < 5e-4, (rid,name,errors[name])
                x = data['test_new']['x'][:64]
                local_error = float((model(x)-model.forward_local(x)).abs().max())
                assert local_error < 0.002, (rid,local_error)
            summary = json.loads((run/'summary.json').read_text())['test_acc']
            last = history[-1]
            summaries.append(dict(run=rid,group=group,seed=seed,final_train_acc=last['train_acc'],
                final_test_acc=last['test_acc'],final_seen_new_acc=last['seen_new_acc'],
                final_test_new_acc=last['test_new_acc'],max_test_acc=max(x['test_acc'] for x in history),
                reached90=summary['t90'] is not None,strict_grokking=summary['audit_fixed_grokking'],
                t10=summary['t10'],t50=summary['t50'],t90=summary['t90'],width_10_90=summary['width_10_90'],
                elapsed_sec=last['wall_sec'],checkpoint_count=len(checkpoints)))
            trajectories[group].append(history)
            curves.extend(dict(run=rid,group=group,seed=seed,**row) for row in history)
            checks[rid] = dict(passed=True,checkpoint_count=len(checkpoints),history_count=len(history),
                optimizer_step=100000,replayed_correct_counts_match=True,cpu_ce_errors=errors,
                cpu_local_forward_max_error=local_error,code_commit=cp['code_commit'],
                final_checkpoint_sha256=sha(checkpoints[-1]))
    for seed in CFG['seeds']:
        a=json.loads((ROOT/'runs'/('multiscale_s'+str(seed))/'origin.json').read_text())
        b=json.loads((ROOT/'runs'/('all_long_s'+str(seed))/'origin.json').read_text())
        assert a['initial_trainable_sha256'] == b['initial_trainable_sha256']
        assert a['data_sha256'] == b['data_sha256']
    for name, rows in [('run_summary',summaries),('complete_curves',curves)]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    dump(dict(passed=True,runs=checks,paired_initialization_and_data=True),out/'verification.json')
    plot(trajectories)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))

def plot(trajectories):
    out = ROOT/'results'
    metrics=[('train_acc','训练集'),('test_acc','新数字组合 · 熟悉布局'),
             ('seen_new_acc','已见组合 · 新布局'),('test_new_acc','新数字组合 · 新布局')]
    font_path = subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family':family,'axes.unicode_minus':False,
                         'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    colors={'multiscale':'#D55E00','all_long':'#0066BB'}
    labels={'multiscale':'四路不同尺度','all_long':'四路同尺度 32'}
    fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for ax,(key,title) in zip(axs.flat,metrics):
        for group in CFG['groups']:
            ys=np.array([[row[key]*100 for row in h] for h in trajectories[group]])
            x=np.array([row['step']/1000 for row in trajectories[group][0]])
            mean,std=ys.mean(0),ys.std(0)
            ax.plot(x,mean,color=colors[group],lw=2.2,label=labels[group])
            ax.fill_between(x,np.maximum(0,mean-std),np.minimum(100,mean+std),color=colors[group],alpha=.17)
        ax.set_title(title,loc='left');ax.set_xlim(0,100);ax.set_ylim(-1,103)
        ax.set_ylabel('准确率（%）');ax.grid(alpha=.2);ax.legend(loc='best',fontsize=10)
        ax.set_xlabel('训练步数（千步）')
    fig.suptitle('一层模59加法：不同尺度与同尺度对照',fontsize=19,y=.99)
    fig.text(.5,.015,'每组3个seed，实线为均值，阴影为种子间±1个标准差；每500步完整记录。',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(out/'mean_curves.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True)
    palette=['#1B1B1B','#D55E00','#008859','#7838AB']
    for row,group in enumerate(CFG['groups']):
        for seed in CFG['seeds']:
            ax=axs[row,seed];h=trajectories[group][seed];x=[r['step']/1000 for r in h]
            for (key,label),color in zip(metrics,palette):
                ax.plot(x,[r[key]*100 for r in h],lw=1.7,color=color,label=label)
            ax.set_title(labels[group]+' · seed '+str(seed),loc='left')
            ax.set_xlim(0,100);ax.set_ylim(-1,103);ax.grid(alpha=.2);ax.set_xlabel('训练步数（千步）')
        axs[row,0].set_ylabel('准确率（%）')
    handles,legend=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.94),fontsize=11)
    fig.suptitle('全部六条完整训练曲线',fontsize=19,y=.99)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(out/'seed_curves.png',dpi=160);plt.close(fig)


def verify_dashboard():
    log=ROOT/'swanlab/logs'
    con=sqlite3.connect(str(log/'runs.swanlab'))
    exps=con.execute('select id,run_id,name,status from experiment').fetchall()
    assert len(exps)==6
    checks,total={},0
    for eid,rid,name,status in exps:
        group='multiscale' if '多尺度' in name else 'all_long'
        run=ROOT/'runs'/(group+'_s'+name[-1])
        receipt=json.loads((run/'swanlab.json').read_text())
        assert receipt['finished'] and receipt['last_step']==100000 and status==1
        by_step={z['step']:z for z in json.loads((run/'history.json').read_text())}
        tags=con.execute('select name,folder from tag where experiment_id=?',(eid,)).fetchall()
        assert len(tags)==16
        count,max_error=0,0
        for key,folder in tags:
            stored=[json.loads(line) for p in (log/rid/'logs'/folder).glob('*.log') for line in p.read_text().splitlines()]
            assert sorted(z['index'] for z in stored)==list(range(0,100001,500))
            src,scale=receipt['metric_map'][key]
            for z in stored:
                error=abs(z['data']-by_step[z['index']][src]*scale)
                assert math.isfinite(error) and error<1e-9
                max_error=max(max_error,error)
            count+=len(stored)
        checks[run.name]=dict(metrics=len(tags),values=count,max_error=max_error,finished=True,dashboard_run_id=rid)
        total+=count
    con.close()
    assert total==19296
    result=dict(passed=True,run_count=6,value_count=total,runs=checks,url='http://127.0.0.1:16100/charts')
    dump(result,ROOT/'results/dashboard_verification.json')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--verify-dashboard',action='store_true')
    args=parser.parse_args()
    if args.verify_dashboard:
        verify_dashboard()
    elif args.plot_only:
        trajectories={g:[[] for s in CFG['seeds']] for g in CFG['groups']}
        with (ROOT/'results/complete_curves.csv').open() as f:
            for row in csv.DictReader(f):
                group,seed=row['group'],int(row['seed'])
                row={k:float(v) for k,v in row.items() if k not in ['run','group','seed']}
                trajectories[group][seed].append(row)
        assert all(len(h)==201 for v in trajectories.values() for h in v)
        plot(trajectories)
    else:
        analyze()
