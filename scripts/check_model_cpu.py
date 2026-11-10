#!/usr/bin/env python3
"""Import the byte-identical native model and check finite-range sweep/write identities."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import csv,hashlib,importlib.util,inspect,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;torch.set_num_threads(1)
p=ROOT/'src/exp2_sequence_local_grokking.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='e419f0d77f35fecd70d1b1d19a5338a6408ed4209aff0d70c892a355ec600f73'
spec=importlib.util.spec_from_file_location('native_firstgen',p);native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
@torch.inference_mode()
def main():
    torch.manual_seed(0)
    cfg=json.loads((ROOT/'experiments/architecture_scan/r3_T4_M4_seed1/config.json').read_text())
    kw={k:v for k,v in cfg.items() if k in inspect.signature(native.SequenceLocalModeModel).parameters}
    kw['lambdas']=native.xi_spread_lambdas(cfg['M'],cfg['xi_min'],cfg['xi_max']).tolist()
    model=native.SequenceLocalModeModel(**kw).double().eval()
    native.assert_no_forbidden_modules(model)
    source=torch.randn(2,16,4,8,dtype=torch.float64);right,left=model.sweep(source)
    r=torch.zeros_like(source);l=torch.zeros_like(source)
    for q in range(1,min(model.sweep.H,15)+1):
        power=model.sweep.lambdas.pow(q)[None,None,:,None]
        r[:,q:]+=power*source[:,:-q];l[:,:-q]+=power*source[:,q:]
    sweep_error=max((right-r).abs().max().item(),(left-l).abs().max().item());assert sweep_error<1e-12
    tokens,roles,labels,_=native.build_sequence_data(59,'add',16,0,7,15)
    tokens=tokens[:8];roles=roles[:8];h=model.residue_features(tokens)+model.role_features(roles)+model.position[None]
    write_error=0.
    for block in model.blocks:
        src=block.drive(h).view(8,16,4,8);right,left=model.sweep(src)
        v=torch.einsum('bpd,mdo->bpmo',h,block.visible_factor)
        a=torch.einsum('bpma,mao->bpmo',torch.cat((right,left),-1),block.mode_factor)
        c=block.local_coupling*v*a*block.mode_logits.softmax(0)[None,None,:,None]
        u=torch.nn.functional.linear(c,block.out.weight,bias=None)
        manual=block.norm(h+u.sum(2)+block.out.bias);actual=block(h)
        write_error=max(write_error,(manual-actual).abs().max().item());h=manual
    logits=model.classifier(model.final_norm(h[:,15]));native_logits=model(tokens,roles)
    error=(logits-native_logits).abs().max().item();assert error<1e-12 and write_error<1e-12
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad);total=sum(p.numel() for p in model.parameters())
    assert trainable==62787 and total==62791
    return dict(ok=True,device='cpu',torch=torch.__version__,trainable_parameters=trainable,total_parameters=total,
                finite_exponential_sweep_max_error=sweep_error,mode_write_identity_max_error=write_error,
                full_forward_max_error=error,logit_shape=list(logits.shape),optimizer_steps=0,
                scope='Random-state mathematical identity checks; no historical checkpoint replay')
if __name__=='__main__':print(json.dumps(main(),indent=2))
