"""Paired donor and dose check on the archived H12 trajectories."""
import csv
import hashlib
import json
import os
import runpy
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('TGLT_SOURCE_ROOT', str(OUT.parents[1]))).resolve()
os.environ['TGLT_SOURCE_ROOT'] = str(ROOT)
legacy = ROOT / 'analysis/paired_evidence_20261121/paired.py'
paired_source = OUT / 'source' / 'paired.py'
p = runpy.run_path(str(paired_source if paired_source.is_file() else legacy))
r, torch, F = (p[k] for k in ('r', 'torch', 'F'))
OFFSETS = (1, 7, 17, 23, 31, 42, 50, 58)
DOSES = (.25, .5, 1.)
FIXED = (4000, 20000, 50000, 85000, 105500, 155000, 256000, 300000)
LEVELS = (.10, .50, .65, .90)

def phases(hist):
    rows = sorted({x['step']: x for x in hist}.values(), key=lambda x: x['step'])
    result = {f'fixed_{t}': t for t in FIXED}
    for level in LEVELS:
        result[f't{int(level*100)}'] = next(
            (a['step'] for a,b,c in zip(rows,rows[1:],rows[2:])
             if c['step']-b['step']==b['step']-a['step']==500
             and all(x['test_correct']>=level*len(r.TE) for x in (a,b,c))), None)
    if result['t10'] is not None:
        for lag in (5000,20000):
            result[f't10_minus_{lag}'] = result['t10']-lag
    return result

def effects(z,y,rival):
    true = z.gather(1,y[:,None]).flatten()
    return (z.argmax(1)==y, true-z.scatter(1,y[:,None],-torch.inf).amax(1),
            true-z.gather(1,rival[:,None]).flatten(), -true+torch.logsumexp(z,1))

@torch.no_grad()
def run():
    hist = {s:p['history'](f'b0.25_wd0.03_s{s}') for s in range(3)}
    design = {s:phases(h) for s,h in hist.items()}
    protocol = {'design':design,'donor_offsets':OFFSETS,'extra_multipliers':DOSES,
        'primary':'On native errors compare true-minus-best-wrong margin of own add with average matched donor add. Show each seed, dose and donor; also retain fixed-rival delta and correctness.',
        'comparison':'Common steps and sustained accuracy anchors. Independent positive reproduction means selective rescue becomes appreciable before sustained 90% in seed 1, across multiple donors/doses. Distinguish tests of formation from detection limits.',
        'analysis_unit':'seed; questions and times are repeated measures'}
    (OUT/'round1_protocol.json').write_text(json.dumps(protocol,indent=2))
    rows,items=[],[]
    for seed,anchors in design.items():
        run_id=f'b0.25_wd0.03_s{seed}'
        for step in sorted(set(anchors.values())-{None}):
            path=p['ckpath'](run_id,step)
            if not path.is_file():
                raise FileNotFoundError(path)
            m,direct=p['get_model'](run_id,step)
            state=r.capture(m)
            assert torch.allclose(state['z'],direct,atol=3e-8,rtol=1e-9)
            b=m.blocks[1]
            visible=torch.einsum('bd,mdo->bmo',state['h'][:r.P,15],b.visible_factor)
            visible=visible-visible.mean(0)
            x=state['received2']-state['received2'].reshape(r.P,r.P,4,8).mean(0)[r.B]
            x=torch.einsum('nmc,mco->nmo',x,b.mode_factor[:,:8])*b.local_coupling*b.mode_logits.softmax(0)[None,:,None]
            pre=state['pre2'][r.TE]
            x=x[r.TE]
            operand=r.B[r.TE]
            add=lambda offset:F.linear((x*visible[(operand+offset)%r.P]).sum(1),b.out.weight)
            q=add(0)
            wrong={o:p['matched'](add(o),q) for o in OFFSETS}
            z=r.at_head(m,pre)
            y=r.Y[r.TE]
            assert torch.allclose(z,direct[r.TE],atol=3e-8,rtol=1e-9)
            old=z.argmax(1)==y
            logged=next(v['test_correct'] for v in hist[seed] if v['step']==step)
            if abs(int(old.sum())-logged)>2:
                raise ValueError((seed,step,int(old.sum()),logged))
            err=~old
            rival=z.scatter(1,y[:,None],-torch.inf).argmax(1)
            base=effects(z,y,rival)
            modes={'native':(0,pre),'delete':(0,pre-q)}
            for d in DOSES:
                modes[f'own_{d}']=(d,pre+d*q)
                for o,w in wrong.items():
                    modes[f'donor_{o}_{d}']=(d,pre+d*w)
            per={}
            for name,(dose,value) in modes.items():
                ok,margin,fix,loss=effects(r.at_head(m,value),y,rival)
                primary=err if err.any() else torch.ones_like(err)
                per[name]=(ok,margin,fix)
                rows.append(dict(seed=seed,step=step,phase=';'.join(k for k,t in anchors.items() if t==step),
                    native=int(old.sum()),historical_correct=logged,original_wrong=int(err.sum()),mode=name,dose=dose,
                    correct=int(ok.sum()),rescued=int((ok&err).sum()),lost=int((~ok&old).sum()),
                    error_margin_gain=float((margin-base[1])[primary].mean()),
                    error_fixed_rival_gain=float((fix-base[2])[primary].mean()),
                    overall_ce=float(loss.mean()), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if name=='native' else ''))
            for d in DOSES:
                ok,u,f=per[f'own_{d}']; donors=[per[f'donor_{o}_{d}'] for o in OFFSETS]
                advantage=u-torch.stack([k[1] for k in donors]).mean(0)
                fixed_adv=f-torch.stack([k[2] for k in donors]).mean(0)
                unique=ok&err&~per['delete'][0]&~torch.stack([k[0] for k in donors]).any(0)
                for i in range(len(y)):
                    items.append((seed,step,d,int(r.TE[i]),int(old[i]),int(ok[i]),int(unique[i]),float(advantage[i]),float(fixed_adv[i])))
            print(seed,step,logged,flush=True)
    with (OUT/'round1_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import gzip
    with gzip.open(OUT/'round1_pairs.csv.gz','wt') as f:
        w=csv.writer(f);w.writerow(('seed','step','dose','input_index','native_correct','own_correct','own_only','margin_advantage','fixed_rival_advantage'));w.writerows(items)

if __name__=='__main__':
    run()
