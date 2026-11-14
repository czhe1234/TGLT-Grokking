"""Standard windowed Transformer with the first-generation TGLT input/readout interface.

W=None is unmasked global attention; W=15 admits every pair in length 16.
All window conditions use identical trainable initialization for a given seed.
"""
from pathlib import Path
import sys, json, hashlib, copy
import torch
from torch import nn
import torch.nn.functional as F
import exp2_sequence_local_grokking as ref
from strict_grokking import audit

ALLOWED = [
    "GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009",
    "GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0",
    "GPU-70b6fceb-c7c4-9467-3c81-097c9284c160",
]

def json_write(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def param_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def make_data():
    t,r,y,_=ref.build_sequence_data(59,"add",16,0,7,15)
    tr,te=ref.make_split(3481,.25,0)
    return dict(tokens=t,roles=r,labels=y,train=tr,test=te)

class WindowTransformerBlock(nn.Module):
    def __init__(self,window,ffn_hidden=256):
        super().__init__()
        self.attn=nn.MultiheadAttention(64,4,dropout=0.,bias=True,batch_first=True)
        self.norm1=nn.LayerNorm(64)
        self.ffn_in=nn.Linear(64,ffn_hidden) if ffn_hidden else None
        self.ffn_out=nn.Linear(ffn_hidden,64) if ffn_hidden else None
        self.norm2=nn.LayerNorm(64) if ffn_hidden else None
        self.window=window
        ix=torch.arange(16)
        mask=None if window is None else (ix[:,None]-ix[None,:]).abs()>window
        self.register_buffer("window_mask",mask,persistent=False)

    def head_writebacks(self,h):
        B,L,D=h.shape
        q,k,v=F.linear(h,self.attn.in_proj_weight,self.attn.in_proj_bias).reshape(B,L,3,4,16).permute(2,0,3,1,4).unbind(0)
        scores=(q@k.transpose(-2,-1))/4.
        if self.window_mask is not None:
            scores=scores.masked_fill(self.window_mask,-torch.inf)
        probs=scores.softmax(-1)
        head_states=(probs@v).transpose(1,2)  # B,L,4,16
        weight=self.attn.out_proj.weight.reshape(64,4,16)
        writes=torch.einsum("blhc,dhc->blhd",head_states,weight)
        return writes,probs

    def finish(self,h,writes):
        pre1=h+writes.sum(-2)+self.attn.out_proj.bias
        z1=self.norm1(pre1)
        if self.ffn_in is None:
            return z1,dict(pre1=pre1,z1=z1)
        fpre=self.ffn_in(z1)
        pre2=z1+self.ffn_out(F.gelu(fpre))
        z2=self.norm2(pre2)
        return z2,dict(pre1=pre1,z1=z1,fpre=fpre,pre2=pre2,z2=z2)

    def forward(self,h,head_mask=None,return_parts=False):
        if head_mask is not None or return_parts:
            writes,attention=self.head_writebacks(h)
            if head_mask is not None:
                writes=writes*head_mask[None,None,:,None]
            out,parts=self.finish(h,writes)
            if return_parts:
                parts.update(writes=writes,attention=attention)
                return out,parts
            return out
        # need_weights=True uses the ordinary explicit scaled dot-product implementation.
        # Disable evaluation fast paths externally so train/eval share this computation.
        update,_=self.attn(h,h,h,attn_mask=self.window_mask,need_weights=True,
                           average_attn_weights=False)
        z=self.norm1(h+update)
        if self.ffn_in is None:return z
        return self.norm2(z+self.ffn_out(F.gelu(self.ffn_in(z))))

class WindowTransformer(nn.Module):
    def __init__(self,seed,window,ffn_hidden=256):
        super().__init__()
        ref.set_seed(seed)
        base=ref.SequenceLocalModeModel(p=59,seq_len=16,query_pos=15,d=64,d_a=8,M=4,
            r=3,T=4,layers=2,lambdas=ref.xi_spread_lambdas(4,.7,32.),
            learnable_lambda=False,local_coupling=1.)
        self.residue=base.residue;self.role=base.role;self.position=base.position
        self.final_norm=base.final_norm;self.classifier=base.classifier
        self.query_pos=15;self.window=window
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10000+seed)
            self.blocks=nn.ModuleList([WindowTransformerBlock(window,ffn_hidden) for _ in range(2)])

    def embed(self,tokens,roles):
        return self.residue(tokens)+self.role(roles)+self.position[None]

    def first_layer(self,tokens,roles):
        return self.blocks[0](self.embed(tokens,roles))

    def from_first(self,h,head_mask=None):
        h=self.blocks[1](h,head_mask=head_mask)
        return self.classifier(self.final_norm(h[:,15]))

    def forward(self,tokens,roles,head_masks=None):
        h=self.embed(tokens,roles)
        for j,block in enumerate(self.blocks):
            h=block(h,head_mask=None if head_masks is None else head_masks[j])
        return self.classifier(self.final_norm(h[:,15]))

def optimizer(model,capturable=False):
    no=set()
    for m in model.modules():
        if isinstance(m,nn.LayerNorm):no.update(id(p) for p in m.parameters(recurse=False))
    decay=[];nd=[]
    for name,p in model.named_parameters():
        # MultiheadAttention's packed in_proj_bias is also a bias.
        (nd if id(p) in no or name.endswith("bias") else decay).append(p)
    lr=torch.tensor(.001,device="cuda") if capturable else .001
    return torch.optim.AdamW([dict(params=decay,weight_decay=.1),dict(params=nd,weight_decay=0.)],
        lr=lr,betas=(.9,.98),eps=1e-8,fused=capturable,capturable=capturable)

def classify(history):
    result=audit(history)
    t10=next((int(x["step"]) for x in history if x["test_acc"]>=.1),None)
    result.update(t10=t10,reached90=result["t90"] is not None,
        width_10_90=None if t10 is None or result["t90"] is None else result["t90"]-t10)
    return result
