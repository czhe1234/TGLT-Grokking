import os
os.environ.update(CUDA_VISIBLE_DEVICES='',SWANLAB_MODE='local',OMP_NUM_THREADS='1')
import fcntl
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def main(dest):
    import swanlab
    from swanlab.deprecated.local import _ColumnInfo, LocalCallbacker
    logdir = ROOT/'swanlab/logs'
    logdir.mkdir(parents=True, exist_ok=True)
    def serialized(method):
        def call(self, *a, **kw):
            with (logdir/'.callback.lock').open('a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                return method(self, *a, **kw)
        return call
    for name in ['on_run_initialized', 'on_scalar_flush', 'on_media_flush', 'on_log_flush', 'on_run_finished']:
        setattr(LocalCallbacker, name, serialized(getattr(LocalCallbacker, name)))
    original = _ColumnInfo.__init__
    def grouped(self, key, kid, chart_type):
        original(self,key,kid,chart_type)
        family = key.split('/',1)[0]
        self.section_name = family
        self.section_sort = {'准确率':0,'损失':1,'写回强度':2,'门控权重':3,'参数':4}.get(family,5)
    _ColumnInfo.__init__ = grouped
    lock = (dest/'swanlab.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt = dest/'swanlab.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get('finished'):
            return
        raise RuntimeError('Existing unfinished logger receipt; inspect before recovery')
    cfg = json.loads((dest/'config.json').read_text())
    group, seed = cfg['group'], cfg['seed']
    label = '多尺度' if group == 'multiscale' else '同尺度32'
    cfg.update(display_group=label,accuracy_unit='百分比',source_history=str(dest/'history.csv'))
    colors = {'multiscale':['#D55E00','#B2182B','#A6761D'],'all_long':['#0072B2','#164A78','#009E73']}
    with (logdir/'.init.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        swanlab.init(project='TGLT-One-Layer-Scales',experiment_name='一层_'+label+'_s'+str(seed),
            group=cfg['exp_group'],color=colors[group][seed],config=cfg,mode='local',log_dir=str(logdir),
            description='一层模59加法；b在查询位，远a与近干扰c同角色；无绝对位置；固定870训练组合；新数字组合与新布局分别评估。',
            settings=swanlab.Settings(probe=dict(hardware=False,runtime=False,requirements=False,git=False,monitor=False)))
    mapping = {
        '准确率/训练集（%）':('train_acc',100),
        '准确率/新数字组合·熟悉布局（%）':('test_acc',100),
        '准确率/已见组合·新布局（%）':('seen_new_acc',100),
        '准确率/新数字组合·新布局（%）':('test_new_acc',100),
        '损失/训练交叉熵':('train_ce',1),
        '损失/新数字组合交叉熵':('test_familiar_ce',1),
        '参数/参数范数':('param_norm',1),
        '写回强度/四路总写回':('total_write_rms',1),
    }
    for m in range(1,5):
        mapping['写回强度/mode '+str(m)] = ('mode%d_write_rms'%m,1)
        mapping['门控权重/mode '+str(m)] = ('mode%d_gate'%m,1)
    last, finished, state = -1, False, 'queued'
    def save_receipt():
        tmp = receipt.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(dict(pid=os.getpid(),last_step=last,finished=finished,state=state,
            version=swanlab.__version__,log_dir=str(logdir),metric_map=mapping),ensure_ascii=False,indent=2))
        tmp.replace(receipt)
    try:
        while True:
            history = dest/'history.json'
            if history.exists():
                for row in json.loads(history.read_text()):
                    if row['step'] > last:
                        swanlab.log({k:float(row[src])*scale for k,(src,scale) in mapping.items()},step=int(row['step']))
                        last = int(row['step'])
            status = json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else {}
            state = status.get('state','queued')
            save_receipt()
            if state == 'failed' or (state == 'complete' and last >= status['step']):
                finished = True
                break
            time.sleep(3)
    finally:
        swanlab.finish()
        save_receipt()

if __name__ == '__main__':
    main(Path(sys.argv[1]))
