#!/usr/bin/env python3
"""Verify portable package bytes, histories and frozen labels. CPU; no training."""
import ast,collections,csv,gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def table(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
    files=table(ROOT/'provenance/source_files.csv')
    for row in files:
        data=(ROOT/row['path']).read_bytes()
        assert len(data)==int(row['bytes']) and sha(data)==row['sha256'],row['path']
        raw=gzip.decompress(data) if row['encoding'].startswith('gzip') else data
        assert len(raw)==int(row['source_bytes']) and sha(raw)==row['source_sha256'],row['path']
    # Use the unchanged historical audit function, without running its server-bound driver.
    label=ROOT/'provenance/historical_scripts/analysis/firstgen_beta_wd_phase_20261027/source/strict_label_source.py'
    tree=ast.parse(label.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
    ns={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),str(label),'exec'),ns)
    audit=ns['audit']
    arch={r['run_id']:r for r in table(ROOT/'results/original_architecture_scan_192/runs_192.csv')}
    beta={r['run_id']:r for r in table(ROOT/'results/beta_wd_available_horizon_labels.csv')}
    index=table(ROOT/'experiments/run_index.csv');assert len(index)==297
    labels=collections.Counter();horizons=collections.Counter();rows=0;max_acc_error=0.
    for row in index:
        path=ROOT/row['history'];raw=gzip.decompress(path.read_bytes());assert sha(raw)==row['history_source_sha256']
        h=json.loads(raw);s=[a['step'] for a in h]
        assert s==sorted(set(s)) and s[0]==0 and s[-1]==int(row['last_step'])
        assert set(range(0,s[-1]+1,500)).issubset(s)
        split=json.loads((ROOT/row['split']).read_text());tr,te=map(set,(split['train_idx'],split['test_idx']))
        assert len(tr)==870 and len(te)==2611 and not tr.intersection(te) and tr.union(te)==set(range(3481))
        for a in h:
            for name in ['train','test']:
                err=abs(a[name+'_correct']/a[name+'_n']-a[name+'_acc']);max_acc_error=max(max_acc_error,err)
                assert err<1e-7,(row['run_id'],a['step'],name)
        result=audit(h);k=row['run_id']
        if row['family']=='architecture_scan':
            expected=arch[k];assert result['audit_fixed_grokking']==bool(int(expected['strict_grokking']))
            assert result['t_mem']==(int(expected['t_mem']) if expected['t_mem'] else None)
            assert result['jump_width']==(int(expected['jump_width']) if expected['jump_width'] else None)
        else:
            expected=beta[k];assert result['audit_fixed_grokking']==(expected['strict_by_cut']=='True'),k
        labels[row['family']]+=int(result['audit_fixed_grokking']);horizons[row['family']+':'+str(s[-1])]+=1;rows+=len(h)
        if row['continuation_config']:
            old=json.loads(gzip.decompress((path.parent/'history_100k_original.json.gz').read_bytes()));lookup={a['step']:a for a in h}
            for a in old:
                for key in ['train_acc','test_acc','train_ce','test_ce','train_correct','test_correct','train_n','test_n']:
                    assert a[key]==lookup[a['step']][key],(k,a['step'],key)
    assert labels=={'architecture_scan':32,'beta_wd_scan':58}
    manifest=ROOT/'SHA256SUMS';checks=0
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest,name=line.split('  ',1);assert sha((ROOT/name).read_bytes())==digest,name;checks+=1
    return dict(ok=True,source_files_checked=len(files),runs=len(index),history_rows=rows,
                frozen_strict_labels=dict(labels),terminal_steps=dict(horizons),maximum_accuracy_count_error=max_acc_error,
                complete_500_step_histories=True,continuation_prefixes_match=True,package_checksum_entries=checks,
                new_training_steps=0)
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,indent=2))
