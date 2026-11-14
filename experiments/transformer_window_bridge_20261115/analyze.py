"""Common checkpoint-level mechanism analysis for TGLT modes and attention heads.

Everything runs on CPU from saved states. Counterfactuals replace one first-layer
position at a time; all downstream operations are recomputed. Conditional margin
partitions freeze native LayerNorm denominators and, for Transformer, GELU gains.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:os.environ[k]="1"
from pathlib import Path
import sys,json,csv,math,copy,time,argparse,concurrent.futures
import numpy as np
B=Path(__file__).resolve().parent;PROJECT=Path(os.environ.get("TGLT_SOURCE_ROOT",str(B.parents[1])))
sys.path.insert(0,str(B/"source"))
from exp2_window_transformer import *
torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
OUT=B/"mechanism"

def write_csv(rows,path):
    if rows:ref.write_csv(rows,Path(path))

class MatrixSweep(ref.OpenLocalExponentialSweep):
    def rebuild(self):
        lam=self.lambdas.detach()
        ix=torch.arange(16,device=lam.device);d=ix[:,None]-ix[None,:]
        for name,valid in [("K_right",(d>0)&(d<=self.H)),("K_left",(d<0)&(-d<=self.H))]:
            mat=lam[:,None,None].pow(d.abs()[None].clamp_min(1))*valid[None]
            if name in self._buffers:self._buffers[name]=mat
            else:self.register_buffer(name,mat,persistent=False)
    def forward(self,source):
        return tuple(torch.einsum("mij,bjmc->bimc",getattr(self,k),source) for k in ["K_right","K_left"])

def baseline_roster():
    out=[]
    for H,r,T in [(7,1,7),(8,2,4),(12,3,4),(15,3,5)]:
        for seed in range(3):
            rid="r%d_T%d_M4_seed%d"%(r,T,seed);path=PROJECT/"week/v1/results"/rid
            assert path.exists()
            out.append(dict(kind="TGLT",window=H,seed=seed,run_id=rid,path=str(path)))
    return out

def transformer_roster():
    return [dict(kind="Transformer",window=w,seed=s,
        run_id=("transformer_global" if w is None else "transformer_W%02d"%w)+"_s%d"%s,
        path=str(B/"runs"/(("transformer_global" if w is None else "transformer_W%02d"%w)+"_s%d"%s)))
        for w in [7,8,12,15,None] for s in range(3)]

def load_model(row,step):
    path=Path(row["path"])/("checkpoint_%07d.pt"%step)
    cp=torch.load(path,map_location="cpu",weights_only=False)
    if row["kind"]=="Transformer":
        m=WindowTransformer(row["seed"],row["window"]);m.load_state_dict(cp["model_state"]);m=m.double().eval()
    else:
        cfg=json.loads((Path(row["path"])/"config.json").read_text())
        for k,v in dict(p=59,pos_a=0,pos_b=7,query_pos=15,split_seed=0,d=64,layers=2,local_coupling=1.,wd=.1,lr=.001).items():
            assert cfg[k]==v,(row["run_id"],k,cfg[k])
        assert cfg["r"]*cfg["T"]==row["window"] and cfg["M"]==4
        m=ref.SequenceLocalModeModel(p=59,seq_len=16,query_pos=15,d=64,d_a=8,M=4,
             r=cfg["r"],T=cfg["T"],layers=2,lambdas=ref.xi_spread_lambdas(4,.7,32.),
             learnable_lambda=False,local_coupling=1.)
        m.load_state_dict(cp["model_state"])
        m=m.double().eval()
        sweep=MatrixSweep(4,cfg["r"],cfg["T"],m.sweep.lambdas.tolist(),False).double()
        # Preserve the original float32-rounded lambda values loaded from the checkpoint.
        sweep.load_state_dict(m.sweep.state_dict());sweep.rebuild()
        m.sweep=sweep
        for block in m.blocks:block.sweep=sweep
    return m,cp,path

def first_layer(m,t,r,kind):
    if kind=="Transformer":return m.first_layer(t,r)
    return m.blocks[0](m.residue_features(t)+m.role_features(r)+m.position[None])

def native_parts(m,h,kind,mask=None):
    b=m.blocks[1]
    if kind=="Transformer":
        w,attention=b.head_writebacks(h);w=w[:,15]
        if mask is not None:w=w*mask[None,:,None]
        h2,parts=b.finish(h[:,15],w)
        parts.update(w=w,base=h[:,15]+b.attn.out_proj.bias,final_input=h2)
    else:
        source=b.drive(h).reshape(len(h),16,4,8)
        state=torch.cat([torch.einsum("mj,bjmc->bmc",getattr(b.sweep,k)[:,15],source)
                          for k in ["K_right","K_left"]],-1)
        v=torch.einsum("bd,mdo->bmo",h[:,15],b.visible_factor)
        a=torch.einsum("bma,mao->bmo",state,b.mode_factor)
        u=b.local_coupling*v*a*b.mode_logits.softmax(0)[None,:,None]
        w=F.linear(u,b.out.weight,None)
        if mask is not None:w=w*mask[None,:,None]
        pre=h[:,15]+b.out.bias+w.sum(1);h2=b.norm(pre)
        parts=dict(w=w,base=h[:,15]+b.out.bias,pre1=pre,final_input=h2)
    z=m.classifier(m.final_norm(h2))
    return z,parts

def affine_ln(partition,reference,ln):
    std=(reference.var(-1,unbiased=False,keepdim=True)+ln.eps).sqrt()
    out=(partition-partition.mean(-1,keepdim=True))*ln.weight/std[:,None,:]
    out[:,-1]+=ln.bias
    return out

def conditional_partition(m,parts,kind,y,wrong):
    b=m.blocks[1]
    u=torch.cat([parts["w"],parts["base"][:,None]],1)
    u=affine_ln(u,parts["pre1"],b.norm1 if kind=="Transformer" else b.norm)
    if kind=="Transformer" and b.ffn_in is not None:
        # GELU(x)=x*Phi(x); freezing Phi at native activations gives an exact
        # conditional affine partition, not an intervention or local derivative.
        gain=.5*(1+torch.erf(parts["fpre"]/math.sqrt(2.)))
        f=F.linear(u,b.ffn_in.weight,None)
        f[:,-1]+=b.ffn_in.bias
        f=f*gain[:,None,:]
        f=F.linear(f,b.ffn_out.weight,None);f[:,-1]+=b.ffn_out.bias
        u=affine_ln(u+f,parts["pre2"],b.norm2)
    u=affine_ln(u,parts["final_input"],m.final_norm)
    delta=m.classifier.weight[y]-m.classifier.weight[wrong]
    vals=(u*delta[:,None,:]).sum(-1)
    vals[:,-1]+=m.classifier.bias[y]-m.classifier.bias[wrong]
    return vals

def endpoints(m,t,r,y,kind,fullparts,h):
    scores={}
    for j in range(4):
        mask=torch.eye(4,dtype=torch.float64)[j]
        z,_=native_parts(m,h,kind,mask);scores["L2_only_%d"%(j+1)]=int((z.argmax(-1)==y).sum())
        if kind=="Transformer":
            z=m(t,r,head_masks=[mask,mask])
        else:
            z=m(t,r,mode_mask=mask)
        scores["both_only_%d"%(j+1)]=int((z.argmax(-1)==y).sum())
    z,_=native_parts(m,h,kind,torch.zeros(4,dtype=torch.float64))
    scores["L2_none"]=int((z.argmax(-1)==y).sum())
    return scores

def phase_steps(row,baseline):
    hist=json.loads((Path(row["path"])/"history.json").read_text());lab=classify(hist)
    steps=[int(p.stem.split("_")[-1]) for p in sorted(Path(row["path"]).glob("checkpoint_*.pt"))]
    if lab["t90"] is not None and lab["t50"] is not None:
        anchors={"before":max((s for s in steps if s<lab["t10"]),default=0),
                 "during":min(steps,key=lambda s:(abs(s-lab["t50"]),s)),"after":100000}
        rule="own_first_complete_10_90"
    else:
        # No invented transition: use the same seed TGLT's event, else fixed thirds.
        bh=json.loads((Path(baseline["path"])/"history.json").read_text());bl=classify(bh)
        if bl["t90"] is not None:
            anchors={"before":max((s for s in steps if s<bl["t10"]),default=0),
                     "during":min(steps,key=lambda s:(abs(s-bl["t50"]),s)),"after":100000}
            rule="matched_TGLT_times_no_own_complete_transition"
        else:
            anchors={"before":25000,"during":50000,"after":100000}
            rule="fixed_25k_50k_100k_no_complete_transition"
    return anchors,rule,lab

@torch.no_grad()
def position_replacements(m,row,step,phase,native_y,d):
    te=d["test"];y=d["labels"][te];t=d["tokens"][te];r=d["roles"][te]
    sums={source:np.zeros((16,3)) for source in ["a","b"]}
    whole={source:0. for source in ["a","b"]};counts=0
    for lo in range(0,len(te),192):
        tt=t[lo:lo+192];rr=r[lo:lo+192];yy=y[lo:lo+192];n=len(tt);counts+=n
        h=first_layer(m,tt,rr,row["kind"]);z,_=native_parts(m,h,row["kind"])
        wrong=z.scatter(1,yy[:,None],-torch.inf).argmax(-1)
        cleanmargin=z.gather(1,yy[:,None])[:,0]-z.gather(1,wrong[:,None])[:,0]
        for source,pos in [("a",0),("b",7)]:
            donor=tt.clone();donor[:,pos]=(donor[:,pos]-1+1)%59+1
            hd=first_layer(m,donor,rr,row["kind"]);zd,_=native_parts(m,hd,row["kind"])
            dz=zd-z;dz-=dz.mean(-1,keepdim=True);whole[source]+=float(dz.square().sum())
            for j in range(16):
                hybrid=h.clone();hybrid[:,j]=hd[:,j]
                pz,_=native_parts(m,hybrid,row["kind"])
                delta=pz-z;delta-=delta.mean(-1,keepdim=True)
                margin=pz.gather(1,yy[:,None])[:,0]-pz.gather(1,wrong[:,None])[:,0]
                sums[source][j]+=[float(delta.square().sum()),float((cleanmargin-margin).sum()),int((pz.argmax(-1)==yy).sum())]
    rows=[]
    W=row["window"]
    for source,src in [("a",0),("b",7)]:
        rms=np.sqrt(sums[source][:,0]/(counts*59))
        shares=rms/rms.sum() if rms.sum()>1e-14 else np.zeros(16)
        relay=[j for j in range(16) if j not in [src,15] and abs(j-src)<=W and abs(15-j)<=W]
        for j in range(16):
            eligible=abs(j-src)<=W and abs(15-j)<=W
            if not eligible:assert rms[j]<1e-10,(row["run_id"],source,j,rms[j])
            rows.append(dict(run_id=row["run_id"],kind=row["kind"],window=W,seed=row["seed"],step=step,
                phase=phase,source=source,source_position=src,position=j,structurally_reachable=eligible,
                relay_position=j in relay,centered_logit_rms=rms[j],influence_share=shares[j],
                margin_drop=sums[source][j,1]/counts,patched_test_acc=sums[source][j,2]/counts,
                relay_share=shares[relay].sum(),whole_input_change_rms=math.sqrt(whole[source]/(counts*59))))
    return rows

def run_one(row,quick=False):
    t0=time.monotonic();dest=OUT/row["run_id"];dest.mkdir(parents=True,exist_ok=True)
    if (dest/"complete.json").exists() and not quick:return json.loads((dest/"complete.json").read_text())
    d=make_data();te=d["test"];t=d["tokens"][te];r=d["roles"][te];y=d["labels"][te]
    base=next(z for z in baseline_roster() if z["window"]==row["window"] and z["seed"]==row["seed"])
    phases,phase_rule,lab=phase_steps(row,base)
    final,cp,_=load_model(row,100000)
    fixed=[]
    with torch.no_grad():
        for lo in range(0,len(y),256):
            h=first_layer(final,t[lo:lo+256],r[lo:lo+256],row["kind"])
            z,_=native_parts(final,h,row["kind"])
            fixed.append(z.scatter(1,y[lo:lo+256,None],-torch.inf).argmax(-1))
    fixed=torch.cat(fixed);del final,cp
    paths=sorted(Path(row["path"]).glob("checkpoint_*.pt"))
    if quick:paths=[paths[-1]]
    hist={int(x["step"]):x for x in json.loads((Path(row["path"])/"history.json").read_text())}
    rows=[];endrows=[];posrows=[];source=[];maxerr=0;max_replay_difference=0
    for path in paths:
        step=int(path.stem.split("_")[-1]);digest=sha(path)
        m,cp,_=load_model(row,step);native_count=0;strength=torch.zeros(4,dtype=torch.float64)
        C=torch.zeros(5,dtype=torch.float64);Cf=C.clone();margin_sum=0.;noff=C.clone()
        solo={k:0 for k in ["L2_only_%d"%j for j in range(1,5)]+["both_only_%d"%j for j in range(1,5)]+["L2_none"]}
        with torch.no_grad():
            for lo in range(0,len(y),256):
                tt=t[lo:lo+256];rr=r[lo:lo+256];yy=y[lo:lo+256]
                h=first_layer(m,tt,rr,row["kind"]);z,parts=native_parts(m,h,row["kind"])
                wrong=z.scatter(1,yy[:,None],-torch.inf).argmax(-1)
                cv=conditional_partition(m,parts,row["kind"],yy,wrong)
                cf=conditional_partition(m,parts,row["kind"],yy,fixed[lo:lo+len(yy)])
                native_margin=z.gather(1,yy[:,None])[:,0]-z.gather(1,wrong[:,None])[:,0]
                fixed_margin=z.gather(1,yy[:,None])[:,0]-z.gather(1,fixed[lo:lo+len(yy),None])[:,0]
                err=max((cv.sum(-1)-native_margin).abs().max().item(),(cf.sum(-1)-fixed_margin).abs().max().item())
                maxerr=max(maxerr,err);assert err<1e-8,err
                if step==100000:
                    native=m(tt,rr)
                    assert (native-z).abs().max()<1e-8
                native_count+=int((z.argmax(-1)==yy).sum());C+=cv.sum(0);Cf+=cf.sum(0)
                strength+=parts["w"].square().sum((0,2));margin_sum+=float(native_margin.sum())
                if step==100000:
                    ep=endpoints(m,tt,rr,yy,row["kind"],parts,h)
                    for k,v in ep.items():solo[k]+=v
            rms=(strength/len(y)).sqrt();q=rms/rms.sum()
            record=dict(run_id=row["run_id"],kind=row["kind"],window=row["window"],seed=row["seed"],step=step,
                train_acc=hist[step]["train_acc"],test_acc=native_count/len(y),
                logged_test_acc=hist[step]["test_acc"],margin=margin_sum/len(y),remainder=float(C[4]/len(y)),
                strict=lab["audit_fixed_grokking"],phase_rule=phase_rule,decomposition_error=maxerr)
            for j in range(4):
                record.update({"q%d"%(j+1):float(q[j]),"strength%d"%(j+1):float(rms[j]),
                    "C%d"%(j+1):float(C[j]/len(y)),"C_fixed100k%d"%(j+1):float(Cf[j]/len(y))})
                if row["kind"]=="TGLT":record["gate%d"%(j+1)]=float(m.blocks[1].mode_logits.softmax(0)[j])
            diff=abs(native_count-round(hist[step]["test_acc"]*len(y)));max_replay_difference=max(max_replay_difference,diff)
            record["replay_correct_difference"]=diff;rows.append(record)
            if step==100000:
                for key,val in dict(native=native_count,**solo).items():
                    endrows.append(dict(run_id=row["run_id"],kind=row["kind"],window=row["window"],seed=row["seed"],
                        condition=key,step=step,test_acc=val/len(y),test_correct=val,test_n=len(y)))
            if step in phases.values() and not quick:
                for phase,s in phases.items():
                    if s==step:posrows.extend(position_replacements(m,row,step,phase,y,d))
        assert sha(path)==digest
        source.append(dict(run_id=row["run_id"],step=step,path=str(path),sha256=digest))
        if len(rows)%10==0:
            json_write(dict(state="running",step=step,points=len(rows),elapsed=time.monotonic()-t0),dest/"status.json")
        del m,cp
    write_csv(rows,dest/("quick_curves.csv" if quick else "curves.csv"))
    write_csv(endrows,dest/("quick_endpoint.csv" if quick else "endpoint.csv"))
    if not quick:
        write_csv(posrows,dest/"positions.csv");write_csv(source,dest/"sources.csv")
        phase_rows=[]
        for phase,step in phases.items():
            rr=next(dict(x) for x in rows if x["step"]==step);rr["phase"]=phase;phase_rows.append(rr)
        write_csv(phase_rows,dest/"phases.csv")
    result=dict(run_id=row["run_id"],points=len(rows),max_partition_error=maxerr,
        max_replay_correct_difference=max_replay_difference,phases=phases,phase_rule=phase_rule,
        strict=lab["audit_fixed_grokking"],elapsed=time.monotonic()-t0)
    json_write(result,dest/("quick_check.json" if quick else "complete.json"))
    return result

def aggregate():
    for name in ["curves","endpoint","positions","phases","sources"]:
        rows=[]
        for d in sorted(OUT.iterdir()):
            if d.is_dir() and (d/"complete.json").exists():
                rows.extend(csv.DictReader((d/(name+".csv")).open()))
        if rows:write_csv(rows,OUT/(name+".csv"))

def main():
    p=argparse.ArgumentParser();p.add_argument("--kind",choices=["TGLT","Transformer","both"],default="both")
    p.add_argument("--quick",action="store_true");p.add_argument("--workers",type=int,default=4);p.add_argument("--wait",action="store_true");a=p.parse_args()
    rows=[r for r in baseline_roster()+transformer_roster() if r["window"] in [8,12] and (a.kind=="both" or r["kind"]==a.kind)]
    if a.quick:rows=rows[:1]
    if a.wait:
        pending=list(rows);active={}
        json_write(baseline_roster()+transformer_roster(),OUT/"roster.json")
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            while pending or active:
                for row in list(pending):
                    sp=Path(row["path"])/"status.json"
                    if not sp.exists():continue
                    st=json.loads(sp.read_text())
                    if st.get("state")=="failed":raise RuntimeError("Training failed: "+row["run_id"])
                    if st.get("state")=="complete" and st.get("step")==100000 and len(active)<a.workers:
                        active[pool.submit(run_one,row)]=row;pending.remove(row)
                for fut in list(active):
                    if fut.done():
                        result=fut.result();print(json.dumps(result),flush=True);del active[fut]
                json_write(dict(waiting=[r["run_id"] for r in pending],active=[r["run_id"] for r in active.values()],updated=time.time()),OUT/"waiting.json")
                time.sleep(5)
        aggregate();json_write(dict(complete=True,runs=len(rows),updated=time.time()),OUT/("status_"+a.kind+".json"))
        return
    for row in rows:assert (Path(row["path"])/"checkpoint_0100000.pt").exists(),row["run_id"]
    if a.quick:
        print(json.dumps(run_one(rows[0],True)));return
    json_write(rows,OUT/"roster.json")
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run_one,rows):print(json.dumps(result),flush=True)
    aggregate()
    json_write(dict(complete=True,runs=len(rows),updated=time.time()),OUT/("status_"+a.kind+".json"))
if __name__=="__main__":main()
