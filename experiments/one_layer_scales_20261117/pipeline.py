import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'config.json').read_text())
TRAIN_PY = '/home/chenzhe/.conda/envs/risk-exp4/bin/python'
TRACK_PY = '/home/chenzhe/work/tglt_experiments/.venv-swanlab/bin/python'

def dump(data,path):
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    tmp.replace(path)

def main():
    lock=(ROOT/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ROOT/'validation/cpu_checks.json').read_text())['passed']
    for seed, group in [(0,'multiscale'),(1,'all_long'),(2,'multiscale')]:
        assert json.loads((ROOT/'validation'/('pilot_'+group+'_s'+str(seed))/'pilot.json').read_text())['passed']
    commit=subprocess.check_output(['git','-C',str(ROOT.parents[1]),'rev-parse','HEAD'],text=True).strip()
    jobs=[]
    for seed,gpu in enumerate(CFG['gpu_uuids']):
        groups=['multiscale','all_long'] if seed != 1 else ['all_long','multiscale']
        for group in groups:
            run_id=group+'_s'+str(seed)
            dest=ROOT/'runs'/run_id
            dest.mkdir(exist_ok=True)
            if (dest/'history.csv').exists():
                raise RuntimeError('Existing run history; refusing duplicate dispatch: '+run_id)
            cfg=dict(CFG,group=group,seed=seed,gpu_uuid=gpu,run_id=run_id,exp_group='one_layer_'+group,code_commit=commit)
            dump(cfg,dest/'config.json')
            dump(dict(state='queued',step=0,updated=time.time()),dest/'status.json')
            jobs.append(dict(run_id=run_id,group=group,seed=seed,gpu=gpu))
    dump(jobs,ROOT/'jobs.json')
    loggers=[]
    for job in jobs:
        dest=ROOT/'runs'/job['run_id']
        f=(dest/'swanlab.log').open('a')
        loggers.append(subprocess.Popen([TRACK_PY,str(ROOT/'monitor.py'),str(dest)],stdout=f,stderr=subprocess.STDOUT))
        f.close()
    active,done,failed={},[],[]
    waiting=jobs[:]
    while waiting or active:
        for gpu in CFG['gpu_uuids']:
            if gpu not in active:
                job=next((j for j in waiting if j['gpu']==gpu),None)
                if job:
                    waiting.remove(job)
                    dest=ROOT/'runs'/job['run_id']
                    f=(dest/'train.log').open('a')
                    p=subprocess.Popen([TRAIN_PY,str(ROOT/'train.py'),'--group',job['group'],'--seed',str(job['seed']),'--gpu',gpu],stdout=f,stderr=subprocess.STDOUT)
                    f.close()
                    active[gpu]=(job,p)
                    print('started '+job['run_id']+' pid='+str(p.pid),flush=True)
        for gpu,(job,p) in list(active.items()):
            code=p.poll()
            if code is not None:
                dest=ROOT/'runs'/job['run_id']
                status=json.loads((dest/'status.json').read_text())
                if code==0 and status.get('state')=='complete' and status.get('step')==CFG['steps']:
                    done.append(job['run_id'])
                else:
                    failed.append(job['run_id'])
                    if status.get('state')!='failed':
                        dump(dict(state='failed',step=status.get('step',0),returncode=code,updated=time.time()),dest/'status.json')
                del active[gpu]
                print('ended '+job['run_id']+' returncode='+str(code),flush=True)
        dump(dict(running=[j['run_id'] for j,p in active.values()],queued=[j['run_id'] for j in waiting],
                  complete=done,failed=failed,updated=time.time(),pid=os.getpid()),ROOT/'status.json')
        time.sleep(3)
    for p in loggers:
        p.wait(timeout=120)
    dump(dict(complete=done,failed=failed,loggers_returncodes=[p.returncode for p in loggers],updated=time.time()),ROOT/'training_complete.json')
    if failed or any(p.returncode for p in loggers):
        raise SystemExit(1)

if __name__=='__main__':
    main()
