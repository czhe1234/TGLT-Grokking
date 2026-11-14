import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:os.environ[key]="1"
from pathlib import Path
import sys,json,copy
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/"source"))
from exp2_window_transformer import *
torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
d=make_data()
assert len(d["train"])==870 and len(d["test"])==2611
assert len(set(d["train"].tolist()) & set(d["test"].tolist()))==0
if (B/"data.pt").exists():
    saved=torch.load(B/"data.pt",map_location="cpu",weights_only=False)
    assert all(torch.equal(saved[k],v) for k,v in d.items())
else:
    ref.atomic_torch_save(d,B/"data.pt")
checks=dict(data_sha256=sha(B/"data.pt"),data_sizes=[870,2611])
errors=[];counts={};initial=[]
for seed in range(3):
    models=[WindowTransformer(seed,w).double().eval() for w in [7,8,12,15,None]]
    hs=[param_hash(m) for m in models];assert len(set(hs))==1
    initial.append(hs[0])
    ref.set_seed(seed)
    base=ref.SequenceLocalModeModel(p=59,seq_len=16,query_pos=15,d=64,d_a=8,M=4,r=3,T=4,
        layers=2,lambdas=ref.xi_spread_lambdas(4,.7,32.),learnable_lambda=False,local_coupling=1.).double()
    for name in ["residue.weight","role.weight","position","final_norm.weight","final_norm.bias","classifier.weight","classifier.bias"]:
        assert torch.equal(base.state_dict()[name],models[0].state_dict()[name])
    t=d["tokens"][::179];r=d["roles"][::179]
    for m in models:
        count=ref.count_params(m);counts[str(m.window)]=count
        z=m(t,r)
        h=m.embed(t,r)
        for b in m.blocks:
            builtin=torch.nn.TransformerEncoderLayer(64,4,256,dropout=0.,activation="gelu",batch_first=True,norm_first=False).double()
            builtin.self_attn.load_state_dict(b.attn.state_dict())
            builtin.norm1.load_state_dict(b.norm1.state_dict());builtin.norm2.load_state_dict(b.norm2.state_dict())
            builtin.linear1.load_state_dict(b.ffn_in.state_dict());builtin.linear2.load_state_dict(b.ffn_out.state_dict())
            h2,parts=b(h,return_parts=True)
            e=max((h2-b(h)).abs().max().item(),(h2-builtin(h,src_mask=b.window_mask)).abs().max().item())
            assert e<2e-12,e;errors.append(e);h=h2
        assert (m.classifier(m.final_norm(h[:,15]))-z).abs().max()<2e-12
        # Geometric dependency test: W=7 cannot carry a from0 to15 through two blocks.
        if m.window==7:
            t0=d["tokens"][:1].expand(59,-1).clone();t0[:,0]=torch.arange(1,60)
            rz=d["roles"][:1].expand(59,-1)
            assert (m(t0,rz)-m(t0,rz)[:1]).abs().max()<1e-12
        op=optimizer(m)
        decay_ids={id(x) for g in op.param_groups if g["weight_decay"] for x in g["params"]}
        assert all(id(p) not in decay_ids for n,p in m.named_parameters() if n.endswith("bias"))
    assert torch.equal(models[-1](t,r),models[-2](t,r))
# Check the manual head route has the same parameter gradients as native MHA.
m=WindowTransformer(0,8).double();n=copy.deepcopy(m)
t=d["tokens"][::137];r=d["roles"][::137];y=d["labels"][::137]
F.cross_entropy(m(t,r),y).backward()
F.cross_entropy(n(t,r,head_masks=[torch.ones(4),torch.ones(4)]),y).backward()
ge=max((p.grad-q.grad).abs().max().item() for p,q in zip(m.parameters(),n.parameters()))
assert ge<1e-10,ge
checks.update(passed=True,native_transformer_max_error=max(errors),native_head_gradient_max_error=ge,
    parameters=counts,window15_matches_global=True,W7_input_a_independent=True,
    shared_interfaces_match_tglt=True,initial_hash_per_seed=initial,
    no_bias_decay=True,source_hashes={s:sha(B/s) for s in ["source/exp2_window_transformer.py","source/exp2_sequence_local_grokking.py","source/strict_grokking.py"]})
json_write(checks,B/"validation/cpu_checks.json")
print(json.dumps(checks),flush=True)
