#!/usr/bin/env python3
import argparse,os,sys,time,platform,traceback,fcntl
from pathlib import Path
from common import ROOT,ALLOWED,save,read,now,sha
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
uuid=os.environ.get('CUDA_VISIBLE_DEVICES');assert uuid in ALLOWED,uuid
assert os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
props=torch.cuda.get_device_properties(0);actual=str(props.uuid)
if not actual.startswith('GPU-'):actual='GPU-'+actual
assert actual==uuid,(actual,uuid)
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import exp2_sequence_local_grokking as native
# Main observables use the audited all-layer replay. Disable legacy sampled proxies/ablations.
native.observe=lambda *args,**kwargs:{}
native.plot_history=lambda *args,**kwargs:None
cfg=read(a.config);rid=Path(cfg['out_dir']).name;dest=Path(cfg['out_dir']);dest.mkdir(parents=True,exist_ok=True)
assert dest.is_relative_to(ROOT)
lock=(dest/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status_path=dest/'run_status.json';old=read(status_path,{})
if old.get('state')=='completed':raise SystemExit('Already completed; no rerun')
if old.get('gpu_uuid'):assert old['gpu_uuid']==uuid,'Resuming on a different GPU is disabled'
checkpoints=sorted(dest.glob('checkpoint_*.pt'))
if checkpoints:
    last=checkpoints[-1];last_step=int(last.stem.split('_')[-1])
    if last_step>=cfg['steps'] and (dest/'summary.json').exists():
        result=read(dest/'summary.json');save(status_path,{**old,'state':'completed','finished_utc':now(),'final_step':last_step,'recovered_completed':True});raise SystemExit(0)
    if last_step>=cfg['steps']:raise RuntimeError('Final checkpoint exists without summary; audit instead of overwrite')
    cfg['resume_checkpoint']=str(last)
else:assert not cfg.get('resume_checkpoint'),'Historical weight reuse is forbidden'
for k in ('lambda_values','lambda_override_after','mode_mask_after'):cfg[k]=tuple(cfg[k])
attempt=int(old.get('attempt',0))+1
status=dict(run_id=rid,pilot=a.pilot,state='running',started_utc=now(),pid=os.getpid(),attempt=attempt,gpu_uuid=actual,gpu_name=props.name,logical_device=0,torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,config=cfg,argv=sys.argv,source_sha256=sha(Path(native.__file__)),resume_from=cfg['resume_checkpoint'])
save(status_path,status);save(dest/f'attempt_{attempt:02d}.json',status)
print('START',__import__('json').dumps(status,ensure_ascii=False),flush=True)
tick=time.monotonic()
try:
    result=native.train_one_run(native.RunConfig(**cfg))
    expected=read(ROOT/'data/split_reference.json');actual_split=read(dest/'split.json')
    assert actual_split==expected,'Dataset split changed'
    status.update(state='completed',finished_utc=now(),elapsed_seconds=time.monotonic()-tick,final_step=result['final']['step'],final_train_acc=result['final']['train_acc'],final_test_acc=result['final']['test_acc'],best_test_acc=result['best_test_acc'],peak_allocated_bytes=torch.cuda.max_memory_allocated())
    assert status['final_step']==cfg['steps']
except BaseException as exc:
    status.update(state='failed',finished_utc=now(),elapsed_seconds=time.monotonic()-tick,error=repr(exc));traceback.print_exc();raise
finally:
    save(status_path,status);save(dest/f'attempt_{attempt:02d}.json',status)
    print('END',__import__('json').dumps(status,ensure_ascii=False),flush=True)
