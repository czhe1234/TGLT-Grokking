"""CPU checks for fixed data, changed decay and removal of FFN only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,copy,importlib.util
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source'))
from exp2_window_transformer import *
torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
old=Path(json.loads((B/'config.json').read_text())['baseline_directory'])
sp=importlib.util.spec_from_file_location('old_model',old/'source/exp2_window_transformer.py');orig=importlib.util.module_from_spec(sp);sp.loader.exec_module(orig)
d=make_data();saved=torch.load(B/'data.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(d[k],saved[k]) for k in d)
assert [len(d['train']),len(d['test'])]==[870,2611]
t=d['tokens'][::179];r=d['roles'][::179];y=d['labels'][::179]
errors=[];gradient_errors=[];counts={};initial={};decays={}
for seed in range(3):
 for ffn in [256,0]:
  hashes=[]
  for w in [8,12,15]:
   m=WindowTransformer(seed,w,ffn).double().eval();original=orig.WindowTransformer(seed,w).double().eval()
   for name,p in m.named_parameters():assert torch.equal(p,dict(original.named_parameters())[name]),name
   if ffn==256:assert param_hash(m)==param_hash(original)
   hashes.append(param_hash(m));counts[str(ffn)]=ref.count_params(m)
   assert (m(t,r)-m(t,r,head_masks=[torch.ones(4),torch.ones(4)])).abs().max()<2e-12
   h=m.embed(t,r)
   for block in m.blocks:
    actual=block(h)
    if ffn:
     native=nn.TransformerEncoderLayer(64,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=False).double()
     native.self_attn.load_state_dict(block.attn.state_dict());native.norm1.load_state_dict(block.norm1.state_dict());native.norm2.load_state_dict(block.norm2.state_dict())
     native.linear1.load_state_dict(block.ffn_in.state_dict());native.linear2.load_state_dict(block.ffn_out.state_dict())
     expected=native(h,src_mask=block.window_mask)
    else:
     assert block.ffn_in is None and block.ffn_out is None and block.norm2 is None
     attn,_=block.attn(h,h,h,attn_mask=block.window_mask,need_weights=True,average_attn_weights=False)
     expected=block.norm1(h+attn)
    error=float((actual-expected).abs().max());assert error<2e-12;errors.append(error);h=actual
   op=optimizer(m,weight_decay=1. if ffn else .1)
   norm_ids={id(p) for module in m.modules() if isinstance(module,nn.LayerNorm) for p in module.parameters(recurse=False)}
   for group in op.param_groups:
    for p in group['params']:
     name=next(n for n,q in m.named_parameters() if q is p)
     expected=0. if id(p) in norm_ids or name.endswith('bias') else (1. if ffn else .1)
     assert group['weight_decay']==expected
   decays[str(ffn)]=[g['weight_decay'] for g in op.param_groups]
  assert len(set(hashes))==1;initial['seed%d_ffn%d'%(seed,ffn)]=hashes[0]
  assert torch.equal(WindowTransformer(seed,15,ffn).double()(t,r),WindowTransformer(seed,None,ffn).double()(t,r))
for ffn in [256,0]:
 m=WindowTransformer(0,8,ffn).double();n=copy.deepcopy(m)
 F.cross_entropy(m(t,r),y).backward();F.cross_entropy(n(t,r,head_masks=[torch.ones(4),torch.ones(4)]),y).backward()
 error=max((p.grad-q.grad).abs().max().item() for p,q in zip(m.parameters(),n.parameters()));assert error<1e-10;gradient_errors.append(error)
result=dict(passed=True,data_sha256=sha(B/'data.pt'),data_sizes=[870,2611],parameters=counts,
 shared_parameters_match_baseline=True,window_initialization_identical=True,W15_equals_global=True,
 native_max_error=max(errors),head_route_gradient_max_error=max(gradient_errors),optimizer_decay_groups=decays,
 initial_hashes=initial)
json_write(result,B/'validation/cpu_checks.json');print(json.dumps(result),flush=True)
