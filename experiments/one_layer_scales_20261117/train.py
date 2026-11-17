import argparse
import os
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--group', choices=['multiscale', 'all_long'], required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--gpu', required=True)
parser.add_argument('--pilot', action='store_true')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT/'config.json').read_text())
if args.gpu not in config['gpu_uuids']:
    raise ValueError('GPU UUID outside permitted set')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
from common import *
from classification import audit
import copy
import ctypes
import fcntl
import platform
import subprocess
import time
import traceback
import uuid

def cuda_uuid():
    class CUuuid(ctypes.Structure):
        _fields_ = [('bytes', ctypes.c_ubyte * 16)]
    driver = ctypes.CDLL('libcuda.so.1')
    assert driver.cuInit(0) == 0
    device, value = ctypes.c_int(), CUuuid()
    assert driver.cuDeviceGet(ctypes.byref(device), 0) == 0
    assert driver.cuDeviceGetUuid_v2(ctypes.byref(value), device) == 0
    return 'GPU-' + str(uuid.UUID(bytes=bytes(value.bytes)))

def main():
    checks = json.loads((ROOT/'validation/cpu_checks.json').read_text())
    assert checks['passed'] and checks['data_sha256'] == sha(ROOT/'data.pt')
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.device_count() == 1 and cuda_uuid() == args.gpu
    run_id = args.group + '_s' + str(args.seed)
    dest = ROOT/('validation' if args.pilot else 'runs')/(('pilot_' if args.pilot else '') + run_id)
    dest.mkdir(exist_ok=True)
    lock = (dest/'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (dest/'history.csv').exists():
        raise RuntimeError('Existing history; refusing to overwrite a run')
    data = torch.load(ROOT/'data.pt', map_location='cpu', weights_only=False)
    sets = {k: {v: t.cuda() for v,t in ds.items()} for k,ds in data.items() if isinstance(ds, dict)}
    model = make_model(args.group, args.seed).cuda()
    opt = optimizer(model, True)
    commit = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip()
    cfg = dict(CFG, group=args.group, seed=args.seed, run_id=run_id, gpu_uuid=args.gpu,
               exp_group='one_layer_'+args.group, code_commit=commit)
    dump(cfg, dest/'config.json')
    dump(dict(pid=os.getpid(),gpu_uuid=args.gpu,gpu_name=torch.cuda.get_device_name(),
              torch=torch.__version__,python=platform.python_version(),parameter_count=34911,
              initial_trainable_sha256=paramhash(model),data_sha256=sha(ROOT/'data.pt'),
              code_commit=commit,code_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}), dest/'origin.json')
    history, step = [], 0
    start = time.time()
    def status(state, **extra):
        dump(dict(state=state,step=step,pid=os.getpid(),gpu_uuid=args.gpu,updated=time.time(),**extra), dest/'status.json')
    try:
        status('preparing')
        x, y = sets['train']['x'], sets['train']['y']
        def update(m, o):
            o.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(m.parameters(), CFG['grad_clip'], foreach=True)
            o.step()
            return loss, norm
        snapshot = {k:v.detach().clone() for k,v in model.state_dict().items()}
        saved_opt = copy.deepcopy(opt.state_dict())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                update(model, opt)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_loss, graph_norm = update(model, opt)
        def restore_capture():
            model.load_state_dict(snapshot)
            flat = [p for g in opt.param_groups for p in g['params']]
            ids = [i for g in saved_opt['param_groups'] for i in g['params']]
            for p, i in zip(flat, ids):
                saved = saved_opt['state'].get(i)
                for k,v in opt.state[p].items():
                    if isinstance(v, torch.Tensor):
                        v.copy_(saved[k] if saved is not None else torch.zeros_like(v))
        restore_capture()
        assert paramhash(model) == json.loads((dest/'origin.json').read_text())['initial_trainable_sha256']
        if args.pilot:
            eager = copy.deepcopy(model)
            eager_opt = optimizer(eager, True)
            update(eager, eager_opt)
            graph.replay()
            torch.cuda.synchronize()
            error = max((p-q).abs().max().item() for p,q in zip(model.parameters(), eager.parameters()))
            assert error < 2e-6, error
            with torch.no_grad():
                z, local = model(x[:24]), model.forward_local(x[:24])
                forward_error = (z-local).abs().max().item()
                assert forward_error < 2e-5, forward_error
            restore_capture()
            del eager, eager_opt
        del snapshot, saved_opt
        @torch.no_grad()
        def evaluate():
            row = dict(step=step,wall_sec=time.time()-start,lr=float(opt.param_groups[0]['lr']),
                       param_norm=sum(p.square().sum().item() for p in model.parameters() if p.requires_grad)**.5)
            for name, ds in sets.items():
                logits = torch.cat([model(ds['x'][i:i+1024]) for i in range(0,len(ds['y']),1024)])
                correct = int((logits.argmax(1)==ds['y']).sum())
                row.update({name+'_acc':correct/len(ds['y']),name+'_ce':float(F.cross_entropy(logits,ds['y'])),
                            name+'_correct':correct,name+'_n':len(ds['y'])})
                if name == 'test_familiar':
                    true = logits.gather(1, ds['y'][:,None]).squeeze(1)
                    other = logits.clone()
                    other.scatter_(1, ds['y'][:,None], float('-inf'))
                    row['test_margin_mean'] = float((true-other.max(1).values).mean())
            row['test_acc'] = row['test_familiar_acc']
            probe = sets['test_new']['x'][:1024]
            _, writes, gates = model.query_state(probe)
            rms = writes.square().mean((0,2)).sqrt()
            for m in range(4):
                row['mode%d_write_rms'%(m+1)] = float(rms[m])
                row['mode%d_gate'%(m+1)] = float(gates[m])
            row['total_write_rms'] = float(writes.sum(1).square().mean().sqrt())
            row['step_loss'] = float(graph_loss) if step else row['train_ce']
            return row
        def savepoint():
            row = evaluate()
            assert all(math.isfinite(v) for v in row.values()), 'Non-finite metrics'
            history.append(row)
            write_history(history, dest/'history.csv')
            if not args.pilot:
                ref.atomic_torch_save(dict(step=step,config=cfg,model_state=model.state_dict(),
                    optimizer_state=opt.state_dict(),rng_state=ref.rng_state(),history=history,
                    data_sha256=sha(ROOT/'data.pt'),code_commit=commit), dest/('checkpoint_%07d.pt'%step))
            status('running',latest=row)
            print(json.dumps({k:row[k] for k in ['step','wall_sec','train_acc','test_acc','seen_new_acc','test_new_acc']}),flush=True)
        savepoint()
        limit = 1000 if args.pilot else CFG['steps']
        bench, bench_step = time.time(), 0
        for step in range(1,limit+1):
            if step <= CFG['warmup_steps'] + 1:
                for group in opt.param_groups:
                    group['lr'].fill_(CFG['lr']*min(1.,step/CFG['warmup_steps']))
            graph.replay()
            if step % CFG['eval_every'] == 0 or step == limit:
                torch.cuda.synchronize()
                assert bool(torch.isfinite(graph_loss)) and bool(torch.isfinite(graph_norm))
                savepoint()
            if args.pilot and step == 100:
                torch.cuda.synchronize()
                bench, bench_step = time.time(), step
        torch.cuda.synchronize()
        if args.pilot:
            sec = (time.time()-bench)/(step-bench_step)
            assert history[-1]['train_ce'] < history[0]['train_ce']
            dump(dict(passed=True,seconds_per_step=sec,estimated_100k_sec=sec*100000,
                      graph_vs_eager_parameter_error=error,gpu_local_forward_error=forward_error,
                      peak_memory=torch.cuda.max_memory_allocated(),initial=history[0],final=history[-1]),dest/'pilot.json')
        else:
            summary = {}
            for key in ['test_acc','seen_new_acc','test_new_acc']:
                h = [dict(row,test_acc=row[key]) for row in history]
                result = audit(h)
                result['t10'] = next((r['step'] for r in h if r['test_acc']>=.1),None)
                result['width_10_90'] = result['t90']-result['t10'] if result['t90'] is not None else None
                summary[key] = result
            dump(summary,dest/'summary.json')
        status('complete',latest=history[-1])
    except BaseException:
        status('failed',error=traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
