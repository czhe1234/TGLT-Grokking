"""Aggregate saved evaluations using the initial seed as the repetition unit."""
import collections
import csv
import gzip
import hashlib
import json
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE

def read(name):
    with (HERE/name).open() as f:
        return list(csv.DictReader(f))

def group(rows,keys,condition):
    out=collections.defaultdict(dict)
    for r in rows:
        out[tuple(r[k] for k in keys)][r[condition]]=r
    return out

def stats(o,ds,gain,frozen):
    conv=lambda k:float(o[k])-statistics.mean(float(v[k]) for v in ds)
    return dict(own=int(o['rescued']),max_wrong=max(int(v['rescued']) for v in ds),
        mean_wrong=statistics.mean(int(v['rescued']) for v in ds),
        margin_adv=conv(gain),fixed_adv=conv(frozen))

def main():
    plan=json.loads((HERE/'round1_protocol.json').read_text())
    ri=group(read('round1_metrics.csv'),('seed','step'),'mode')
    elig={(s,str(t)) for s,a in plan['design'].items() for t in a.values() if t is not None}
    assert set(ri)==elig
    h12=[]
    for (s,step),a in ri.items():
        assert len(a)==2+3*9
        native=int(a['native']['native'])
        eps=abs(native-int(a['native']['historical_correct']))
        assert eps<=2
        for r in a.values():
            assert int(r['correct'])==native+int(r['rescued'])-int(r['lost'])
        a1=a['own_1.0']
        ds=[v for k,v in a.items() if k.startswith('donor_') and k.endswith('_1.0')]
        h12.append(dict(seed=int(s),step=int(step),native=native,historical_delta=eps,original_wrong=2611-native,
                       phase=a1['phase'],**stats(a1,ds,'error_margin_gain','error_fixed_rival_gain')))
    r3=group(read('round3_metrics.csv'),('centering','seed','phase','step','dose'),'offset')
    robust=[]
    for (c,s,ph,t,d),a in r3.items():
        assert set(map(int,a))==set(range(59))
        own=a['0'];ds=[a[str(i)] for i in range(1,59)]
        vals=stats(own,ds,'best_rival_gain','fixed_rival_gain')
        robust.append(dict(centering=c,seed=int(s),phase=ph,step=int(t),dose=float(d),
                           original_wrong=int(own['original_wrong']),**vals))
    geo=group(read('geometry_metrics.csv'),('task','H','seed','step'),'condition')
    spatial=collections.defaultdict(dict)
    for r in read('geometry_metrics.csv'):
        if r['condition']=='delete_L1_joint':spatial[r['task'],r['H'],r['seed'],r['step']][int(r['site'])]=r
    assert set(geo)==set(spatial) and len(geo)==27
    geometry=[]
    for (tab,a) in geo.items():
        task,H,s,step=tab
        n=int(a['native']['native'])
        assert len(spatial[tab])==16
        assert all(int(r['correct'])==n for r in spatial[tab].values() if r['candidate']=='0')
        min1=min(int(r['correct']) for r in spatial[tab].values())
        q=a.get('delete_query_term')
        geometry.append(dict(task=task,H=int(H),seed=int(s),native=n,
          min_L1=min1, strongest_sites=[i for i,r in spatial[tab].items() if int(r['correct'])==min1] if n!=min1 else [],
          all_L1=int(a['delete_all_L1_joint']['correct']),query=int(q['correct']) if q else None))
    tt=group(read('temporal_metrics.csv'),('task','H','seed','step'),'condition')
    assert len(tt)==len(json.loads((HERE/'temporal_roster.json').read_text()))
    cross=[]
    for q in json.loads((HERE/'temporal_anchors.json').read_text()):
        tab=tuple(str(q[k]) for k in ('task','H','seed','replayed_step'))
        a=tt[tab]
        native=int(a['native']['native'])
        controls=[v for k,v in a.items() if k.startswith('add_wrong_')]
        assert len(controls)==8
        if native==2611:continue
        vals=stats(a['double_query_term'],controls,'wrong_margin_change','fixed_rival_change')
        cross.append(dict(**q,native=native,original_wrong=2611-native,**vals))
    with gzip.open(HERE/'round1_pairs.csv.gz','rt') as f:
        pairs=collections.defaultdict(dict)
        for r in csv.DictReader(f):
            if r['dose']=='1.0':
                pairs[r['seed'],r['step']][r['input_index']]=r
    common=[]
    for seed,p0 in plan['design'].items():
        t10,t50=p0['t10'],p0['t50']
        early,late=pairs[seed,str(t10)],pairs[seed,str(t50)]
        assert early.keys()==late.keys() and len(early)==2611
        both=[k for k in early if early[k]['native_correct']==late[k]['native_correct']=='0']
        common.append(dict(seed=int(seed),wrong_both=len(both),t10_only=sum(int(early[k]['own_only']) for k in both),
                           t50_only=sum(int(late[k]['own_only']) for k in both)))
    summary=dict(h12=h12,geometry=geometry,robust=robust,cross=cross,common=common,
        case_counts=dict(h12=len(ri),geometry=len(geo),cross_temporal=len(tt),donor_slices=len(r3)))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    for k in ('case_counts','common'):print(k,summary[k])

if __name__=='__main__':
    main()
