"""Recompute the paired CPU diagnostic from the original TGLT checkpoints."""
import csv
import gzip
import inspect
import json
import os
import sys
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(os.environ['TGLT_SOURCE_ROOT'])
RELAY = ROOT / 'TGLT-Grokking/experiments/relay_learning_20261118'
WAIT = RELAY.parent / 'relay_waiting_20261119'
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(RELAY), str(WAIT)]
import replay as r
from controls import matched
from order import cycles
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
P = r.P
test = r.TE.numpy()
labels = r.Y.numpy()
a, b = r.A.numpy(), r.B.numpy()
total, diff = (a+b)%P, (a-b)%P
early = ROOT / 'analysis/firstgen_beta_wd_phase_20261027/runs'
late = ROOT / 'analysis/firstgen_beta_wd_gray_continuation_20261102/continuation_300k'
geo = ROOT / 'week/v1/results'
with (RELAY/'results/trajectories.csv').open() as f:
    previous = {(s['run_id'],int(s['step'])):s for s in csv.DictReader(f)}

def ckpath(run,step):
    base = geo if run.startswith('r') else (early if step<=100000 else late)
    return base/run/('checkpoint_%07d.pt'%step)

def readcp(run,step):
    return torch.load(ckpath(run,step),map_location='cpu',weights_only=False)

def history(run):
    p = (geo if run.startswith('r') else late)/run/'history.json'
    return json.loads(p.read_text())

def onset(h):
    return next((x['step'] for i,x in enumerate(h[:-2]) if all(y['test_correct']>=.10*len(test) for y in h[i:i+3])),None)

def spectrum_frequencies(run):
    step = 100000 if run.startswith('r') else 300000
    head = readcp(run,step)['model_state']['classifier.weight'].double().numpy()
    energy = (abs(np.fft.rfft(head,axis=0))**2).sum(1)
    return np.argsort(-energy[1:],kind='stable')+1

def project(x, ids, keys):
    means = np.stack([x[ids==y].mean(0) for y in range(P)])
    coef = np.fft.rfft(means,axis=0)
    kept = np.zeros_like(coef)
    kept[keys] = coef[keys]
    return np.fft.irfft(kept,n=P,axis=0)[ids]

def crossfit(x,keys):
    # An independent check: no training logits enter this projection, and each evaluated
    # row is predicted only from other held-out logits. No test answer is fitted.
    angles = 2*np.pi*total[:,None]*keys[None,:]/P
    basis = np.column_stack((np.ones(len(x)),np.cos(angles),np.sin(angles)))
    folds = np.array_split(np.random.default_rng(119).permutation(test),2)
    result = np.empty_like(x[test])
    for j in (0,1):
        src, target = folds[j], folds[1-j]
        weights=np.linalg.lstsq(basis[src],x[src],rcond=None)[0]
        result[np.searchsorted(np.sort(test),target)] = basis[target]@weights
    return result[np.searchsorted(np.sort(test),test)]

def get_model(run,step):
    cp=readcp(run,step)
    cfg=cp['config']
    assert cp['step']==step and cfg['p']==P and cfg['M']==4 and cfg['layers']==2
    assert (cfg['pos_a'],cfg['pos_b'],cfg['query_pos'])==(0,7,15)
    split=json.loads((ckpath(run,step).parent/'split.json').read_text())
    assert split['train_idx']==r.TR.tolist() and split['test_idx']==r.TE.tolist()
    kw={k:v for k,v in cfg.items() if k in inspect.signature(r.ref.SequenceLocalModeModel).parameters}
    kw['lambdas']=cfg['lambda_values'] or r.ref.xi_spread_lambdas(4,cfg['xi_min'],cfg['xi_max'])
    m=r.ref.SequenceLocalModeModel(**kw).double().eval()
    m.load_state_dict(cp['model_state'])
    native=m.sweep
    baseline=torch.cat([m(r.DATA['tokens'][ix],r.DATA['roles'][ix]) for ix in r.I.split(192)])
    sweep=r.bridge.MatrixSweep(4,cfg['r'],cfg['T'],native.lambdas.tolist(),False).double()
    sweep.load_state_dict(native.state_dict())
    sweep.rebuild()
    m.sweep=sweep
    for block in m.blocks:
        block.sweep=sweep
    return m,baseline

def common_geometry():
    for seed in range(3):
        dirs=[geo/('r%d_T4_M4_seed%d'%(rad,seed)) for rad in (2,3)]
        cfg=[json.loads((d/'config.json').read_text()) for d in dirs]
        assert {k for k in cfg[0] if cfg[0][k]!=cfg[1][k]}=={'out_dir','r'}
        states=[readcp(d.name,0)['model_state'] for d in dirs]
        assert states[0].keys()==states[1].keys() and all(torch.equal(v,states[1][k]) for k,v in states[0].items())

def embedding_paths():
    cycle,_=cycles('cpu',0)
    rows=[]
    for seed in range(3):
        run='b0.25_wd0.03_s%d'%seed
        h=history(run)
        row=dict(run=run,t10=onset(h))
        for t in (0,20000,85000):
            w=readcp(run,t)['model_state']['residue.weight'][1:].double()
            row['order_%d'%t]=float(cycle.measure(w))
            row['sha_%d'%t]=r.digest(ckpath(run,t))
        rows.append(row)
    return rows

@torch.no_grad()
def evaluate(run,step,rank,items,metrics):
    m,direct=get_model(run,step)
    s=r.capture(m)
    assert torch.allclose(s['z'],direct,atol=3e-8,rtol=1e-9)
    z=s['z'].numpy()
    pred=z[test].argmax(1)
    assert np.array_equal(pred,direct[r.TE].argmax(1).numpy())
    orig=pred==labels[test]
    native=int(orig.sum())
    hist=next(v for v in history(run) if v['step']==step)
    assert hist['test_correct']==native
    mat=r.influence_matrix(m,s,7)
    no7=r.write_output(m,s,mat,-s['j'])
    bl=m.blocks[1]
    v=torch.einsum('bd,mdo->bmo',s['h'][:P,15],bl.visible_factor)
    v=v-v.mean(0)
    incoming=s['received2']-s['received2'].reshape(P,P,4,8).mean(0)[r.B]
    x=torch.einsum('nmc,mco->nmo',incoming,bl.mode_factor[:,:8])*bl.local_coupling*bl.mode_logits.softmax(0)[None,:,None]
    q=F.linear((x*v[r.B]).sum(1),bl.out.weight)
    wrong_b=F.linear((x*v[(r.B+1)%P]).sum(1),bl.out.weight)
    pre=s['pre2']
    outputs={'native':z[test],'drop7':no7[r.TE].numpy(),'drop15':r.at_head(m,pre-q)[r.TE].numpy(),
             'double15':r.at_head(m,pre+q)[r.TE].numpy(),'add_wrong_b15':r.at_head(m,pre+matched(wrong_b,q))[r.TE].numpy(),
             'shift15':r.at_head(m,pre-q+matched(wrong_b,q))[r.TE].numpy(),
             'shift7':r.write_output(m,s,mat,-s['j']+matched(s['j'][r.DONOR],s['j']))[r.TE].numpy()}
    dc=z.mean(0)
    for k in (3,5,10):
        keys=rank[:k]
        part=project(z,total,keys)
        outputs['keep%d'%k]=(dc+part)[test]
        outputs['exclude%d'%k]=(z-part)[test]
        if k==5:
            # Four real 2-D same-frequency product terms per frequency.
            outputs['keep5_block']=(dc+part+project(z,diff,keys))[test]
            outputs['keep5_diff']=(dc+project(z,diff,keys))[test]
            outputs['keep5_held_crossfit']=crossfit(z,keys)
            local_part=project(q.numpy(),total,keys)
            outputs['add15_sum5']=r.at_head(m,pre+torch.from_numpy(local_part))[r.TE].numpy()
            q_wave=torch.from_numpy(local_part)
            j_wave=torch.from_numpy(project(s['j'].numpy(),total,keys))
            outputs['add15_rest5']=r.at_head(m,pre+q-q_wave)[r.TE].numpy()
            outputs['drop15_sum5']=r.at_head(m,pre-q_wave)[r.TE].numpy()
            outputs['drop15_rest5']=r.at_head(m,pre-q+q_wave)[r.TE].numpy()
            outputs['drop7_sum5']=r.write_output(m,s,mat,-j_wave)[r.TE].numpy()
            outputs['drop7_rest5']=r.write_output(m,s,mat,-s['j']+j_wave)[r.TE].numpy()
            # Can perfectly memorized training logits induce a misleading restricted score?
            # Permute only held-out logits, destroying their input identities.
            null_hits=[]
            for seed in range(32):
                first=z.copy()
                first[test]=z[np.random.default_rng(seed).permutation(test)]
                null_scores=(dc+project(first,total,keys))[test]
                null_hits.append(int((null_scores.argmax(1)==labels[test]).sum()))
            assert np.allclose(part.mean(0),0,atol=1e-10)
            # Independently recover the sum terms using a full 2-D FFT of the grid.
            fft=np.fft.fft2(z.reshape(P,P,-1),axes=(0,1))
            picked=np.zeros_like(fft)
            for key in keys:
                picked[key,key]=fft[key,key]
                picked[-key,-key]=fft[-key,-key]
            assert np.allclose(np.fft.ifft2(picked,axes=(0,1)).real.reshape(z.shape),part,atol=1e-9)
    ref=previous.get((run,step))
    sha=r.digest(ckpath(run,step))
    if ref:
        assert sha==ref['sha256']
        for name,field in [('drop7','joint_removed_correct'),('drop15','no_query_mod_correct')]:
            assert (outputs[name].argmax(1)==labels[test]).sum()==int(ref[field])
    detail={n:o.argmax(1) for n,o in outputs.items()}
    fourier_ok=detail['keep5']==labels[test]
    wrong=~orig
    row=dict(run=run,step=step,held=len(test),native_wrong=int(wrong.sum()),native_correct=native,
             top5=' '.join(map(str,rank[:5])),keep5_null_min=min(null_hits),keep5_null_median=float(np.median(null_hits)),keep5_null_max=max(null_hits),path=str(ckpath(run,step)),sha256=sha)
    for name,scores in outputs.items():
        ok=detail[name]==labels[test]
        correct=scores[np.arange(len(test)),labels[test]]
        wrong_scores=scores.copy()
        wrong_scores[np.arange(len(test)),labels[test]]=-np.inf
        row_metrics=dict(row,method=name,correct=int(ok.sum()),repaired=int((wrong&ok).sum()),
                         spoiled=int((orig&~ok).sum()),both_repair_with_keep5=int((wrong&ok&fourier_ok).sum()),
                         ce=float(np.mean(np.logaddexp.reduce(scores,axis=1)-correct)),
                         margin=float((correct-wrong_scores.max(1)).mean()))
        assert row_metrics['correct']==native+row_metrics['repaired']-row_metrics['spoiled']
        metrics.append(row_metrics)
    for i,ix in enumerate(test):
        items.append(dict(run=run,step=step,a=a[ix],b=b[ix],answer=labels[ix],
                          **{n:int(vals[i]) for n,vals in detail.items()}))
    print(run,step,native,flush=True)

def output(name,rows):
    p=OUT/name
    op=gzip.open if name.endswith('.gz') else open
    with op(p,'wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

def main():
    assert r.digest(Path(r.ref.__file__))==r.SRC_HASH
    common_geometry()
    output('waiting.csv',embedding_paths())
    jobs={('r%d_T4_M4_seed%d'%(rad,seed),100000) for rad in (2,3) for seed in range(3)}
    jobs.update((r.RUNS[0],t) for t in (105500,155000))
    jobs.update((r.RUNS[1],t) for t in (105500,155000,256000,300000))
    metrics,items=[],[]
    for run,step in sorted(jobs):
        evaluate(run,step,spectrum_frequencies(run),items,metrics)
    output('comparison.csv',metrics)
    output('predictions.csv.gz',items)

if __name__=='__main__':
    main()
