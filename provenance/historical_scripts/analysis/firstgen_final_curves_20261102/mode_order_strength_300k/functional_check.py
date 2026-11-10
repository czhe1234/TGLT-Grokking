"""Post-primary check at fixed observed endpoints. All ablations are forward evaluations."""
import os,sys
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[v]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import io,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,pandas as pd,torch
import replay as R
O=Path(__file__).resolve().parent
event=pd.read_csv(O/'strict_events.csv')
extra=pd.read_csv(O/'all_first_10_90_events.csv').query('group=="other90"')
pairs=pd.read_csv(O/'matched_pairs_and_full_control.csv').query('metric=="C_sd"')
points={(r.run,t) for r in pd.concat([event,extra]).itertuples() for t in (r.start,r.end)}
for r in pairs.itertuples():
 a=event[event.run==r.run].iloc[0];points.update([(r.control,int(a.start)),(r.control,int(a.end))])
def rms(u):return (u-u.mean(-1,keepdim=True)).square().mean().sqrt()
@torch.inference_mode()
def do(z):
 k,t=z;st,ix,h=R.state(k);cfg,m,x,roles,y,test=st;src,ex=R.folders(k)
 p=src if t<=100000 else ex
 raw=(p/f'checkpoint_{t:07d}.pt').read_bytes();assert R.hashlib.sha256(raw).hexdigest()==R.read(p/f'receipt_{t:07d}.json')['checkpoint_sha256']
 cp=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);assert cp['step']==t
 m.load_state_dict(cp['model_state'],strict=True)
 f=R.eng.Forward(st);q,u=f.end(permode=True);total=u.sum(1)
 result=f.score(f.readout(q,total));nt=len(result);assert nt==2611 and abs(result.sum()-h[t]['test_correct'])<=3
 cs,valid,_=R.corr.continuous(total.numpy());v=cs-cs.mean();assert (v@v)>0
 row={'run':k,'step':t,'native':int(result.sum()),'n_test':nt,'C_sd':cs.std(),'top_s':int(cs.argmax()),
      'top_gap':float(np.sort(cs)[-1]-np.sort(cs)[-2]),'add_variance_share':float(v[1]**2/(v@v))}
 solos=[];match=[];target=rms(total)
 for j in range(4):
    w=u[:,j];a=rms(w);assert a>0
    raw_ok=f.score(f.readout(q,w));matched=f.score(f.readout(q,w*(target/a)))
    solos.append(raw_ok);match.append(matched)
    c,good,_=R.corr.continuous(w.numpy())
    row[f'm{j+1}_Csd']=float(c.std());row[f'm{j+1}_Cadd']=float(c[1])
    row[f'm{j+1}_valid']=int(good.sum());row[f'm{j+1}_solo']=int(raw_ok.sum())
    row[f'm{j+1}_solo_RMS']=int(matched.sum());row[f'm{j+1}_scale']=float(target/a)
 row['best_single_raw']=int(max(z.sum() for z in solos));row['best_single_RMS']=int(max(z.sum() for z in match))
 row['best_of_raw_or_RMS']=max(row['best_single_raw'],row['best_single_RMS'])
 row['needs_combination_raw']=int((result&~np.any(solos,axis=0)).sum())
 row['needs_combination_even_RMS_options']=int((result&~np.any(solos+match,axis=0)).sum())
 return row
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=8) as ex:
    data=pd.DataFrame(ex.map(do,sorted(points)))
 data.to_csv(O/'mode_ablation_endpoints.csv',index=False)
 ds=data.set_index(['run','step']);ts=pd.read_csv(O/'trajectories.csv.gz',usecols=['run','step','C_sd']).set_index(['run','step'])
 assert max(abs(ds.C_sd-ts.loc[ds.index].C_sd))<1e-12
 res={}
 for typ,ev in [('strict',event),('other90',extra)]:
    before=ds.loc[[(r.run,int(r.start)) for r in ev.itertuples()]].reset_index(drop=True)
    after=ds.loc[[(r.run,int(r.end)) for r in ev.itertuples()]].reset_index(drop=True)
    names=[f'm{i}_Csd' for i in range(1,5)]
    diff=(after[names]-before[names]).to_numpy()
    assert np.isfinite(diff).all()
    res[typ]=dict(n=len(ev),primary_start_top_gap_min=float(before.top_gap.min()),primary_end_top_gap_min=float(after.top_gap.min()),
       answer_dir_variance_share_start_median=float(before.add_variance_share.median()),answer_dir_variance_share_end_median=float(after.add_variance_share.median()),
       n_modes_up=[int(v) for v in (diff>0).sum(0)],counts_how_many_modes_up={str(n):int(((diff>0).sum(1)==n).sum()) for n in range(5)},
       hit90_native=int((after.native>=.9*2611).sum()),hit90_any_single_raw=int((after.best_single_raw>=.9*2611).sum()),
       hit90_any_single_RMS=int((after.best_single_RMS>=.9*2611).sum()),
       hit90_any_single_raw_or_RMS=int((after.best_of_raw_or_RMS>=.9*2611).sum()),
       median_native=float(after.native.median()/2611),median_best_single_raw=float(after.best_single_raw.median()/2611),
       median_best_single_RMS=float(after.best_single_RMS.median()/2611),median_best_of_either=float(after.best_of_raw_or_RMS.median()/2611),
       no_single_even_RMS_correct_fraction_at_start=float((before.needs_combination_even_RMS_options/before.native).median()),
       no_single_even_RMS_correct_fraction_at_end=float((after.needs_combination_even_RMS_options/after.native).median()),
       needs_combination_raw_before=int(before.needs_combination_raw.median()),needs_combination_raw_after=int(after.needs_combination_raw.median()),
       native_better_than_best_of_either=int((after.native>after.best_of_raw_or_RMS).sum()))
 (O/'ablation_summary.json').write_text(json.dumps(res,indent=2,allow_nan=False));print(json.dumps(res))

