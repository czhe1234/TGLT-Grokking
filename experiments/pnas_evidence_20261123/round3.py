"""All wrong operands, comparing full-grid and training-only centering."""
import csv
import json
from round1 import OUT, F, effects, p, r, torch

@torch.no_grad()
def run():
    plan=json.loads((OUT/'round1_protocol.json').read_text())['design']
    rows=[]
    for s in range(3):
        rid=f'b0.25_wd0.03_s{s}'
        steps=plan[str(s)]
        tags=['fixed_85000','t10_minus_20000','t10_minus_5000','t10','t50','fixed_300000']
        roster={steps[tag]:tag for tag in tags if steps[tag] is not None}
        for step,phase in sorted(roster.items()):
            model,z=p['get_model'](rid,step)
            state=r.capture(model)
            te=r.TE
            z=z[te]
            pre=state['pre2'][te]
            y=r.Y[te]
            assert torch.allclose(r.at_head(model,pre),z,atol=3e-8,rtol=1e-9)
            rival=z.scatter(1,y[:,None],-torch.inf).argmax(1)
            nr=effects(z,y,rival); bad=~nr[0]
            assert bad.any()
            bl=model.blocks[1]
            vis=torch.einsum('bd,mdo->bmo',state['h'][:59,15],bl.visible_factor)
            assert set(r.B[r.TR].tolist())==set(range(59))
            vis=vis-vis.mean(0)
            arrived=state['received2']
            by_b=torch.stack([arrived[r.TR[r.B[r.TR]==b]].mean(0) for b in range(59)])
            full=arrived.reshape(59,59,4,8).mean(0)
            for name,baseline in [('full_grid',full),('train_only_by_b',by_b)]:
                x=arrived[te]-baseline[r.B[te]]
                x=torch.einsum('nmc,mco->nmo',x,bl.mode_factor[:,:8])*bl.local_coupling*bl.mode_logits.softmax(0)[None,:,None]
                def term(offset):
                    return F.linear((x*vis[(r.B[te]+offset)%59]).sum(1),bl.out.weight)
                q=term(0)
                for d in (.5,1.):
                    for offset in range(59):
                        update=q if offset==0 else p['matched'](term(offset),q)
                        logits=r.at_head(model,pre+d*update)
                        ok,margin,fixed,loss=effects(logits,y,rival)
                        changed=logits-z
                        changed=changed-changed.mean(1,keepdim=True)
                        true=changed.gather(1,y[:,None]).flatten()
                        old_rival=changed.gather(1,rival[:,None]).flatten()
                        rows.append(dict(seed=s,step=step,phase=phase,centering=name,dose=d,offset=offset,held=len(y),
                            original_wrong=int(bad.sum()),rescued=int((ok&bad).sum()),lost=int((~ok&~bad).sum()),
                            best_rival_gain=float((margin-nr[1])[bad].mean()),fixed_rival_gain=float((fixed-nr[2])[bad].mean()),
                            centered_true_gain=float(true[bad].mean()),centered_old_rival_gain=float(old_rival[bad].mean()),
                            ce=float(loss.mean()),native_ce=float(nr[3].mean())))
            print(s,step,phase,int(nr[0].sum()),flush=True)
    with (OUT/'round3_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    run()
