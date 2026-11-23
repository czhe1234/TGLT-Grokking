"""Spatial pathway replay on moved layouts and arithmetic tasks."""
import csv
import importlib.util
import inspect
import json
import sys
from pathlib import Path

from round1 import ROOT, OUT, r, torch, F, effects, OFFSETS

MOVE = ROOT/'experiments/firstgen_followups_20261114'
sys.path.insert(0,str(MOVE))
import mode_mechanism as mm
sys.path.pop(0)
spec=importlib.util.spec_from_file_location('v2_arithmetic',ROOT/'week/v2/code/exp2_sequence_local_grokking.py')
ref=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=ref
spec.loader.exec_module(ref)

def load(task,H,seed,step):
    if task=='moved_add':
        home=MOVE/'runs'/f'geometry_H{H:02d}_s{seed}'
        m,_,path=mm.model_at(dict(path=home,seed=seed,H=H,span=11),step)
        data=mm.dataset(11)
        return m,data,home,path,(0,5,11)
    rad=H//4
    home=ROOT/'week/v2/results'/f'{task}_r{rad}_T4_M4_seed{seed}'
    path=home/('checkpoint_%07d.pt'%step)
    cp=torch.load(path,map_location='cpu',weights_only=False)
    cfg=cp['config']
    kwargs={k:v for k,v in cfg.items() if k in inspect.signature(ref.SequenceLocalModeModel).parameters}
    kwargs['lambdas']=cfg['lambda_values'] or ref.xi_spread_lambdas(4,.7,32.)
    m=ref.SequenceLocalModeModel(**kwargs).double().eval()
    m.load_state_dict(cp['model_state'])
    sweep=r.bridge.MatrixSweep(4,rad,4,m.sweep.lambdas.tolist(),False).double()
    sweep.load_state_dict(m.sweep.state_dict())
    sweep.rebuild()
    m.sweep=sweep
    for b in m.blocks:
        b.sweep=sweep
    tokens,roles,labels,_=ref.build_sequence_data(59,task,16,0,7,15)
    split=json.loads((home/'split.json').read_text())
    return m,dict(tokens=tokens,roles=roles,labels=labels,test=torch.tensor(split['test_idx'])),home,path,(0,7,15)

def product(b,h,received):
    v=torch.einsum('nid,mdo->nimo',h,b.visible_factor)
    x=torch.einsum('nimk,mko->nimo',received,b.mode_factor)
    return b.local_coupling*v*x*b.mode_logits.softmax(0)[None,None,:,None]

def prep1(m,h):
    b=m.blocks[0]
    received=torch.cat(tuple(b.sweep(b.drive(h).reshape(len(h),16,4,8))),dim=-1)
    return h+b.out(product(b,h,received).sum(2))

def read(m,pre):
    return m.classifier(m.final_norm(m.blocks[1].norm(pre)))

def query(m,h):
    b=m.blocks[1]
    i=m.query_pos
    src=b.drive(h).reshape(len(h),16,4,8)
    arr=torch.cat([torch.einsum('mj,njmk->nmk',getattr(m.sweep,f'K_{side}')[:,i],src) for side in ('right','left')],-1)
    v=h[:,i:i+1]
    return (v+b.out(product(b,v,arr[:,None]).sum(2)))[:,0],arr

def joint(x):
    grid=x.reshape(59,59,*x.shape[1:])
    return (grid-grid.mean(0,keepdim=True)-grid.mean(1,keepdim=True)+grid.mean((0,1),keepdim=True)).reshape_as(x)

@torch.no_grad()
def inspect_model(task,H,seed,step,what):
    m,data,home,path,(a,b,q)=load(task,H,seed,step)
    tokens,roles,te,y=(data[k] for k in ('tokens','roles','test','labels'))
    assert torch.equal(tokens[:,a]-1,r.A) and torch.equal(tokens[:,b]-1,r.B)
    direct=torch.cat([m(tokens[i],roles[i]) for i in r.I.split(192)])
    h=m.residue_features(tokens)+m.role_features(roles)+m.position[None]
    pre1=prep1(m,h)
    h1=m.blocks[0].norm(pre1)
    base,arr=query(m,h1)
    z=read(m,base)[te]; y=y[te]
    assert torch.allclose(z,direct[te],atol=3e-8,rtol=1e-9),(task,H,seed,step)
    old=z.argmax(1)==y
    hist=json.loads((home/'history.json').read_text())
    historical=next(x for x in hist if x['step']==step)
    logged=int(round(historical['test_acc']*len(te)))
    assert abs(int(old.sum())-logged)<=2,(task,H,seed,step,int(old.sum()),logged)
    rival=z.scatter(1,y[:,None],-torch.inf).argmax(1)
    reference=effects(z,y,rival)
    rows=[]
    def add(condition,pred,site=''):
        ok,mar,fix,ce=effects(pred,y,rival)
        wrong=~old
        rows.append(dict(task=task,H=H,seed=seed,step=step,query=q,condition=condition,site=site,
            candidate=int(abs(site-a)<=H and abs(site-b)<=H and abs(site-q)<=H) if isinstance(site,int) else '',
            native=int(old.sum()),logged=logged,held=len(te),correct=int(ok.sum()),
            rescued=int((ok&wrong).sum()),lost=int((~ok&old).sum()),
            margin_change=float((mar-reference[1]).mean()),
            wrong_margin_change=float((mar-reference[1])[wrong].mean()) if wrong.any() else '',
            fixed_rival_change=float((fix-reference[2]).mean()),ce=float(ce.mean()),
            checkpoint_sha256=r.digest(path) if condition=='native' else ''))
    add('native',z)
    if what=='geometry':
        j=joint(pre1-h)
        live=[i for i in range(16) if abs(i-a)<=H and abs(i-b)<=H]
        dead=[i for i in range(16) if i not in live]
        if dead:
            assert j[:,dead].abs().max()<2e-10
        for site in range(16):
            modified=h1[te].clone()
            modified[:,site]=m.blocks[0].norm(pre1[te,site]-j[te,site])
            add('delete_L1_joint',read(m,query(m,modified)[0]),site)
        alljoint=m.blocks[0].norm(pre1[te]-j[te])
        add('delete_all_L1_joint',read(m,query(m,alljoint)[0]))
    if H<q-a:
        bl=m.blocks[1]
        vis=h1[:59,q]
        assert torch.allclose(h1[:,q],vis[r.B],atol=2e-10,rtol=0)
        vis=torch.einsum('bd,mdo->bmo',vis,bl.visible_factor)
        vis=vis-vis.mean(0)
        incoming=arr-arr.reshape(59,59,4,16).mean(0)[r.B]
        x=torch.einsum('nmk,mko->nmo',incoming[te],bl.mode_factor)*bl.local_coupling*bl.mode_logits.softmax(0)[None,:,None]
        term=lambda d:F.linear((x*vis[(r.B[te]+d)%59]).sum(1),bl.out.weight)
        signal=term(0)
        before=base[te]
        add('delete_query_term', read(m,before-signal))
        if what=='temporal':
            from round1 import p
            add('double_query_term',read(m,before+signal))
            for d in OFFSETS:
                add(f'add_wrong_{d}',read(m,before+p['matched'](term(d),signal)))
    print(task,H,seed,step,int(old.sum()),flush=True)
    return rows

def main():
    what=sys.argv[1]
    jobs=[]
    anchors=[]
    if what=='geometry':
        for task,hs in [('moved_add',(6,8,10,11,13)),('sub',(8,12)),('mul',(8,12))]:
            for H in hs:
                for seed in range(3):
                    jobs.append((task,H,seed,100000))
    elif what=='temporal':
        for task,H in [('moved_add',8),('sub',12),('mul',12)]:
            for seed in range(3):
                home=(MOVE/'runs'/f'geometry_H{H:02d}_s{seed}' if task=='moved_add'
                      else ROOT/'week/v2/results'/f'{task}_r{H//4}_T4_M4_seed{seed}')
                hist=json.loads((home/'history.json').read_text())
                available=sorted(int(f.stem.split('_')[-1]) for f in home.glob('checkpoint_*.pt'))
                steps={100000}
                for level in (.10,.50,.65,.90):
                    t=next((x['step'] for x in hist if x['test_acc']>=level),None)
                    if t is not None:
                        after=min(k for k in available if k>=t)
                        steps.add(after)
                        anchors.append(dict(task=task,H=H,seed=seed,phase=f'first_{int(level*100)}',history_step=t,replayed_step=after))
                        if level==.10:
                            before=max(k for k in available if k<=max(0,t-5000))
                            steps.add(before)
                            anchors.append(dict(task=task,H=H,seed=seed,phase='before_first_10',history_step=t-5000,replayed_step=before))
                jobs.extend((task,H,seed,t) for t in sorted(steps))
    else:
        raise ValueError(what)
    (OUT/f'{what}_roster.json').write_text(json.dumps(jobs,indent=2))
    if anchors:
        (OUT/'temporal_anchors.json').write_text(json.dumps(anchors,indent=2))
    rows=[]
    for job in jobs:
        rows.extend(inspect_model(*job,what))
    with (OUT/f'{what}_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    main()
