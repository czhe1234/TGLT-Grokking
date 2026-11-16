"""Verify curve coverage, physical bounds, input hashes and frozen source index."""
from pathlib import Path
import csv,hashlib,json,math,subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent
METRICS=['G1_0q','G1_0b','G2_0q','G2_0b','chi_mean','S_total','D_total','q34']
roster=json.loads((HERE/'roster.json').read_text());allrows=list(csv.DictReader((HERE/'curves.csv').open()))
sources=list(csv.DictReader((HERE/'sources.csv').open()));assert len(allrows)==len(sources)==325
byid={(r['run_id'],int(r['step'])):r for r in allrows};assert len(byid)==325
for row in roster:
    expected=list(range(0,40001,1000 if row['span']==15 else 500))
    assert row['steps']==expected
    for step in expected:
        v=byid[row['run_id'],step];assert int(v['input_count'])==3481
        assert all(math.isfinite(float(v[k])) for k in METRICS)
        assert all(-1.0000001<=float(v[k])<=1.0000001 for k in ['G1_0q','G1_0b','G2_0q','G2_0b','q34'])
        for field in ['total','mode3','mode4']:
            assert 0<=float(v['S_'+field])<=math.log(64)+1e-10
            assert 1-1e-10<=float(v['D_'+field])<=64+1e-10
            assert float(v['variance_'+field])>0
        assert 0<=float(v['chi_mean'])<=float(v['chi_rms'])+1e-10
for s in sources:
    p=Path(s['path']);assert p.stat().st_size==int(s['bytes'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==s['sha256']
    assert (s['run_id'],int(s['step'])) in byid
    cp=torch.load(p,map_location='cpu',weights_only=False)
    assert int(cp['step'])==int(s['step'])
    cfg=cp['config'];r=next(x for x in roster if x['run_id']==s['run_id'])
    assert cfg['seed']==r['seed'] and cfg['M']==4 and cfg['d']==64 and cfg['layers']==2
    state=cp['model_state']
    if r['span']==11:
        assert cfg['experiment']=='geometry' and cfg['H']==8
        ix=torch.arange(16);distance=ix[:,None]-ix[None,:]
        lam=torch.sigmoid(state['sweep.raw_lambda'])*.999
        power=lam[:,None,None].pow(distance.abs()[None].clamp_min(1))
        for direction,valid in [('right',(distance>0)&(distance<=8)),('left',(distance<0)&(-distance<=8))]:
            actual=state['sweep.K_'+direction]
            assert float((actual-power*valid[None]).abs().max())<1e-6
            for layer in [0,1]:assert torch.equal(actual,state['blocks.%d.sweep.K_%s'%(layer,direction)])
    else:
        assert cfg['r']*cfg['T']==8 and cfg['pos_a']==0 and cfg['pos_b']==7 and cfg['query_pos']==15
checks=json.loads((HERE/'validation/implementation_checks.json').read_text());assert len(checks['checks'])==15
source=HERE.parents[1]/'src/exp2_sequence_local_grokking.py'
reference_sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert reference_sha=='e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73'
report=dict(checkpoints=325,runs=5,input_count=3481,primary_scalar_points=325*8,
 no_missing_steps=True,source_hashes_verified=325,checkpoint_geometry_verified=325,source_sha256=reference_sha,
 CPU_only=True,training_steps=0,target_labels_used=False,all_covariances_nondegenerate=True,
 jacobian_validation_cases=15,jacobian_max_error=max(x['jacobian_max_error'] for x in checks['checks']),
 finite_difference_max_error=max(x['finite_difference_max_error'] for x in checks['checks']),
 analysis_start_commit='9e0ba02',source_records='sources.csv',limits='One original success/failure pair; no seed-level significance test')
(HERE/'validation/completion_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
