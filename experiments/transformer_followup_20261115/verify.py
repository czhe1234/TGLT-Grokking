"""Validate fixed-budget histories, complete checkpoints, mechanisms and SwanLab."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,math,sqlite3,urllib.request,urllib.parse,ast,time
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));import mechanism as a
import torch
import subprocess
import numpy as np
import pandas as pd
torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
if '--wait' in sys.argv:
 while not (B/'training_complete.json').exists():time.sleep(5)
 assert json.loads((B/'training_complete.json').read_text())['training_complete']
d=a.make_data();reports=[];hashes=[];cps0={};maxdiff=0
for row in a.transformer_roster():
 dest=Path(row['path']);budget=row['steps'];st=json.loads((dest/'status.json').read_text());assert st['state']=='complete' and st['step']==budget
 h=json.loads((dest/'history.json').read_text());expected=list(range(0,budget+1,500));assert [x['step'] for x in h]==expected
 paths=sorted(dest.glob('checkpoint_*.pt'));assert [int(x.stem.split('_')[-1]) for x in paths]==expected
 cfg=json.loads((dest/'config.json').read_text());origin=json.loads((dest/'origin.json').read_text());assert origin['gpu_uuid'] in a.ALLOWED
 for name,digest in origin['code_hashes'].items():assert a.sha(B/name)==digest
 for r in h:
  assert (r['train_n'],r['test_n'])==(870,2611)
  assert all(math.isfinite(v) for v in r.values())
  for pre in ['train','test']:assert abs(r[pre+'_acc']-r[pre+'_correct']/r[pre+'_n'])<1e-15
 m,cp,_=a.load_model(row,budget);assert cp['step']==budget and cp['config']==cfg and cp['data_sha256']==a.sha(B/'data.pt')
 assert all(torch.equal(cp[k+'_idx'],d[k]) for k in ['train','test'])
 assert set(cp['rng_state'])=={'python','torch','cuda'} and len(cp['rng_state']['cuda'])==1
 opstates=cp['optimizer_state']['state'];assert opstates and all(int(x['step'])==budget for x in opstates.values())
 decay=[g['weight_decay'] for g in cp['optimizer_state']['param_groups']];assert decay==([1.,0.] if row['group']=='T1' else [.1,0.])
 with torch.no_grad():z=torch.cat([m(d['tokens'][i:i+256],d['roles'][i:i+256]) for i in range(0,3481,256)])
 correct=z.argmax(-1)==d['labels'];dif={k:int(correct[d[k]].sum())-h[-1][k+'_correct'] for k in ['train','test']}
 assert max(abs(v) for v in dif.values())<=1,(row['run_id'],dif)
 maxdiff=max(maxdiff,max(abs(v) for v in dif.values()))
 for p in paths:hashes.append(dict(run_id=row['run_id'],step=int(p.stem.split('_')[-1]),sha256=a.sha(p),bytes=p.stat().st_size,path=str(p)))
 cp0=torch.load(paths[0],map_location='cpu',weights_only=False)['model_state'];cps0[row['run_id']]=cp0
 old=Path(json.loads((B/'config.json').read_text())['baseline_directory'])/'runs'/('transformer_W%02d_s%d'%(row['window'],row['seed']))/'checkpoint_0000000.pt'
 base=torch.load(old,map_location='cpu',weights_only=False)['model_state']
 assert all(torch.equal(p,base[k]) for k,p in cp0.items()),row['run_id']
 report=dict(run_id=row['run_id'],group=row['group'],window=row['window'],seed=row['seed'],budget=budget,points=len(h),gpu_uuid=origin['gpu_uuid'],final_train=h[-1]['train_acc'],final_test=h[-1]['test_acc'],CPU_float64_replay_count_difference=dif,optimizer_step=budget,decay_groups=decay)
 reports.append(report);print(json.dumps(report),flush=True)
a.write_csv(hashes,B/'validation/checkpoint_manifest.csv')
for group in ['T1','T2']:
 for seed in range(3):
  xs=[cps0['%s_W%02d_s%d'%(group,w,seed)] for w in [8,12,15]]
  assert all(all(torch.equal(xs[0][k],z[k]) for k in xs[0]) for z in xs[1:])
# Formula parity: core functions are byte-for-byte identical ASTs to the original analysis.
old=Path(json.loads((B/'config.json').read_text())['baseline_directory']);trees=[ast.parse((p/'analyze.py' if p==old else p/'mechanism.py').read_text()) for p in [old,B]]
parity={}
for name in ['first_layer','native_parts','affine_ln','conditional_partition','endpoints']:
 nodes=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
 parity[name]=ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False);assert parity[name]
if '--wait' in sys.argv:
 while not (B/'mechanism/completion.json').exists():time.sleep(5)
 subprocess.run([sys.executable,str(B/'postprocess.py')],check=True)
mech=json.loads((B/'mechanism/completion.json').read_text());assert mech['complete']
expected_ids={r['run_id'] for r in a.transformer_roster() if next(z for z in reports if z['run_id']==r['run_id'])['final_test']>=.9}
actual_ids={r['run_id'] for r in mech['selected'] if r['kind']=='Transformer'};assert expected_ids==actual_ids
checks=[]
for row in mech['selected']:
 rec=json.loads((B/'mechanism'/row['run_id']/'complete.json').read_text());assert rec['points']==row.get('steps',100000)//(500 if row['kind']=='Transformer' else 1000)+1;assert rec['max_partition_error']<1e-8;checks.append(rec)
# Mean curves must always contain exactly three seeds and match saved source points.
means=pd.read_csv(B/'results/mean_std_curves.csv');summ=json.loads((B/'results/run_summary.json').read_text());source={r['run_id']:r for r in summ}
for (group,w),frame in means.groupby(['group','window']):
 rs=[r for r in summ if r['group']==group and r['window']==w];assert len(rs)==3
 hs=[{x['step']:x for x in json.loads((Path(r['path'])/'history.json').read_text())} for r in rs]
 for rec in frame.itertuples():
  vals=np.array([h[rec.step][rec.metric]*100 for h in hs]);assert rec.n==3
  assert np.isclose(rec.mean_percent,vals.mean(),rtol=1e-12,atol=1e-12) and np.isclose(rec.std_percent,vals.std(),rtol=1e-12,atol=1e-12)
# Compare the independent dashboard with its source histories.
db=sqlite3.connect(str(B/'swanlab/logs/runs.swanlab'));exps=db.execute('select id,name from experiment').fetchall();assert len(exps)==18
mapping={'准确率/训练集（%）':('train_acc',100),'准确率/测试集（%）':('test_acc',100),'损失/训练交叉熵':('train_ce',1),'损失/测试交叉熵':('test_ce',1),'分类间隔/测试集平均':('test_margin_mean',1),'参数/参数范数':('weight_norm',1)}
names={}
for row in a.transformer_roster():
 name=('带FFN_wd1' if row['group']=='T1' else '纯Attention_wd0.1')+'_窗口%d_seed%d'%(row['window'],row['seed']);names[name]=row
scalars=0
for eid,name in exps:
 row=names[name];dest=Path(row['path']);h=json.loads((dest/'history.json').read_text());receipt=json.loads((dest/'swanlab.json').read_text());assert receipt['finished'] and receipt['last_step']==row['steps']
 for key,(col,factor) in mapping.items():
  url='http://127.0.0.1:16098/api/v1/experiment/%d/tag/%s'%(eid,urllib.parse.quote(key,safe=''))
  data=json.load(urllib.request.urlopen(url,timeout=15))['data']['list'];got={int(r['index']):float(r['data']) for r in data};expect={int(r['step']):r[col]*factor for r in h}
  assert got.keys()==expect.keys(),(name,key,len(got),len(expect));assert all(math.isclose(got[k],v,rel_tol=1e-6,abs_tol=1e-7) for k,v in expect.items());scalars+=len(got)
result=dict(passed=True,training_runs=18,checkpoints=len(hashes),expected_checkpoints=4518,history_points=sum(r['points'] for r in reports),final_checks=reports,shared_initialization_matched=True,mechanism_formula_AST_parity=parity,selected_final90_transformer_runs=len(actual_ids),mechanism_runs=len(checks),mechanism_points=sum(r['points'] for r in checks),max_partition_error=max(r['max_partition_error'] for r in checks),max_final_CPU_replay_difference=maxdiff,mean_std_points_verified=len(means),swanlab_runs=18,swanlab_scalars_verified=scalars)
assert len(hashes)==4518;a.json_write(result,B/'validation/completion_checks.json');print(json.dumps({k:v for k,v in result.items() if k!='final_checks'}),flush=True)
