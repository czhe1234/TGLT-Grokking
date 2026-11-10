"""Only the fixed, historical two-layer model is evaluated. No training entry point."""
import os,sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import csv,gzip,hashlib,importlib.util,inspect,io,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
import torch.nn.functional as F
import torch as th
out=Path(__file__).resolve().parent;analysis=out.parents[1]
scan=analysis/'firstgen_beta_wd_gray_continuation_20261102'
phase=analysis/'firstgen_beta_wd_phase_20261027'
sys.path.insert(0,str(scan))
from gray_common import csv_in,read,assert_history,check_prefix,audit,sha
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
eng=load('actual_modes_engine',analysis/'firstgen_multiscale_function_20261102/engine.py')
corr=load('tested_direction_corr',analysis/'firstgen_beta_continuous_write_correlation_20261102/measure_beta.py')
g11=load('tested_mode_power',phase/'source/g11_science_utils.py')
native=eng.ref.native
torch.set_num_threads(1)
roster={r['run_id']:r for r in csv_in(scan/'cohort_105.csv')}
prior={r['run']:r for r in csv_in(out.parent/'continuation_outcomes/run_results.csv')}
labels={r['run_id']:r for r in csv_in(scan/'results/300k_runs.csv')}
with gzip.open(analysis/'firstgen_beta_continuous_write_correlation_20261102/trajectories.csv.gz','rt',encoding='utf-8-sig') as f:
    cache={(r['run_id'],int(r['step'])):(float(r['C_sd']),float(r['C_add'])) for r in csv.DictReader(f) if r['run_id'] in roster}
assert len(roster)==len(prior)==105 and len(cache)>20000

def folders(k):
    rr=roster[k];base=Path(rr['parent_dir'])
    ext=Path(rr['previous_300k']) if rr['previous_300k'] else scan/'continuation_300k'/k if rr['status'].startswith('continue_') else None
    return base,ext

def state(k):
    base,ext=folders(k)
    cfg=read(base/'config.json')
    assert tuple(cfg[s] for s in ('r','T','layers','M','p','d','d_a','train_fraction','query_pos','pos_a','pos_b'))==(3,4,2,4,59,64,8,.25,15,0,7)
    assert cfg['local_coupling']==float(roster[k]['beta']) and cfg['wd']==float(roster[k]['wd']) and cfg['seed']==int(roster[k]['seed'])
    kw={s:v for s,v in cfg.items() if s in inspect.signature(native.SequenceLocalModeModel).parameters}
    kw['lambdas']=cfg.get('lambda_values') or native.xi_spread_lambdas(4,cfg['xi_min'],cfg['xi_max']).tolist()
    model=native.SequenceLocalModeModel(**kw).double().eval()
    tok,roles,lab,_=native.build_sequence_data(59,'add',16,0,7,15)
    idx=read(base/'split.json')
    tst=np.zeros(3481,bool);tst[idx['test_idx']]=True
    h=read((ext or base)/'history.json');assert_history(h,int(prior[k]['last_recorded']))
    return (cfg,model,tok,roles,lab,tst),idx,{a['step']:a for a in h}

@torch.inference_mode()
def replay(k,step,st,idx,h,full=False):
    base,ext=folders(k);p=base if step<=100000 else ext
    payload=(p/f'checkpoint_{step:07d}.pt').read_bytes()
    rec=read(p/f'receipt_{step:07d}.json')
    fingerprint=hashlib.sha256(payload).hexdigest()
    assert rec['checkpoint_bytes']==len(payload) and fingerprint==rec['checkpoint_sha256'] and int(rec['step'])==step
    cp=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    cfg,model,tok,roles,lab,tst=st
    assert cp['step']==step and all(cp['config'][s]==cfg[s] for s in ('r','T','layers','M','p','seed','wd','local_coupling','split_seed','train_fraction'))
    model.load_state_dict(cp['model_state'],strict=True)
    f=eng.Forward(st)
    q,u=f.end(permode=True)
    total=u.sum(1)
    c,ok,z=corr.continuous(total.numpy())
    cs,sd,other,contrast=corr.summary(c)
    strengths=torch.linalg.vector_norm(u,dim=-1).mean(0).numpy()
    A=float(strengths.sum())
    logits=f.readout(q,total);pred=logits.argmax(1).numpy()
    true=lab.numpy()
    testdiff=int((pred[tst]==true[tst]).sum()-h[step]['test_correct'])
    traindiff=int((pred[~tst]==true[~tst]).sum()-h[step]['train_correct'])
    error=0.
    if (k,step) in cache:
        refsd,refc=cache[k,step]
        error=max(abs(sd-refsd),abs(cs-refc))
        assert error<1e-8,(k,step,sd,refsd,cs,refc)
    row=dict(run=k,step=step,A=A,s1=strengths[0],s2=strengths[1],s3=strengths[2],s4=strengths[3],
         C_sd=sd,C_add=cs,largest_direction=int(np.argmax(c)) if np.isfinite(c).all() else -1,
         invalid=int((~ok).sum()),test_count_diff=testdiff,train_count_diff=traindiff,cache_error=error,checkpoint_sha=fingerprint)
    if full:
        # Full original sweeps and auto-hook capture; includes each mode rather than only the sum.
        sm,ar,checks=g11.measure(model,cfg,tok,roles,lab,torch.tensor(idx['train_idx']),torch.tensor(idx['test_idx']),include_mixed=False)
        np.testing.assert_allclose(strengths,ar['raw_mean_norm'],rtol=2e-10,atol=2e-11)
        c_native,diff=eng.ref.write_exact(model,tok,roles)
        assert th.allclose(total,th.from_numpy(c_native),rtol=1e-10,atol=1e-10)
        assert np.max(abs(diff-logits.numpy()))<1e-9
        dc=corr.direct(z,ok)
        assert np.max(abs(c-dc))<1e-9
        row['pilot_strength_abs_error']=float(np.max(abs(strengths-ar['raw_mean_norm'])))
        row['pilot_direction_abs_error']=float(np.max(abs(c-dc)))
        row['pilot_rule_D_relative_error']=float(abs(sm['effective_D']-h[step]['D'])/max(h[step]['D'],1e-14))
    return row

def process(task):
    k,steps=task;st,idx,h=state(k)
    part=out/'parts'/f'{k}_{steps[0]:07d}_{steps[-1]:07d}.csv'
    if part.exists():
        with part.open() as f:rs=list(csv.DictReader(f))
        assert [int(q['step']) for q in rs]==steps
        return len(steps)
    rows=[replay(k,t,st,idx,h) for t in steps]
    tmp=part.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    os.replace(tmp,part)
    return len(rows)

if __name__=='__main__':
    if '--pilot' in sys.argv:
        ts=time.monotonic();pil=[]
        for j,(k,t) in enumerate([('b2_wd0.03_s0',150000),('b2_wd0.1_s1',66500),('b0.25_wd0.01_s0',300000)]):
            st,idx,h=state(k)
            for k0 in range(1 if j else 3):
                begin=time.monotonic();r=replay(k,t+k0*500,st,idx,h,full=k0==0);r['seconds']=time.monotonic()-begin;pil.append(r)
                print(k,t+k0*500,'seconds',round(r['seconds'],3),'native',k0==0,'countdiff',r['test_count_diff'],flush=True)
        (out/'pilot.json').write_text(json.dumps(pil,indent=2))
    else:
        with (out/'requested_steps.json').open() as f:plan=json.load(f)
        workers=int(sys.argv[1]);assert (out/'pilot.json').exists()
        jobs=[]
        for k,steps in plan.items():
            for s in range(0,len(steps),35):jobs.append((k,steps[s:s+35]))
        jobs.sort(key=lambda x:(x[1][0],x[0]))
        total=sum(len(x[1]) for x in jobs);done=0;start=time.time()
        with ProcessPoolExecutor(workers) as ex:
            fut={ex.submit(process,x):x for x in jobs}
            for task in as_completed(fut):
                try:done+=task.result()
                except Exception as e:print('ERROR',fut[task],repr(e),flush=True);raise
                if done%500<35 or done==total:
                    print(done,total,round(time.time()-start,1),flush=True)
                    p=out/'status.tmp';p.write_text(json.dumps(dict(done=done,total=total,seconds=time.time()-start)));os.replace(p,out/'status.json')

