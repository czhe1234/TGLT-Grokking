"""Attribute the archived strict events to direction organization, mode magnitude, and composition."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[v]='1'
import json,csv,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
out=Path(__file__).resolve().parent; root=out.parents[1]; scan=root/'firstgen_beta_wd_gray_continuation_20261102'
sys.path.insert(0,str(scan))
from gray_common import csv_in,read,audit,assert_history,check_prefix,sha
def write(df,name):df.to_csv(out/name,index=False,encoding='utf-8-sig' if str(name).endswith('.csv') else 'utf-8')
def dump(name,x): (out/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def groups():
    return {r['run_id']:r for r in csv_in(scan/'cohort_105.csv')}
roster=groups();frozen={r['run_id']:r for r in csv_in(scan/'results/300k_runs.csv')}
summaryfile=out.parent/'continuation_outcomes/run_results.csv';polarfile=out.parent/'continuation_outcomes/parameter_curves.csv.gz'
pvals=pd.read_csv(polarfile);o35=out.parent/'all35_seed_means/cells.csv'
settings=pd.read_csv(o35)
fixed90=pd.read_csv(root/'firstgen_beta_continuous_write_correlation_20261102/original_G_windows.csv').set_index('run_id')
trackcols=['C_sd','C_add','P','A','D']
analysed=['C_sd','C_add','P','log_A','D','share_displacement']
S=['q1','q2','q3','q4']
def stats(x):
    z=np.asarray(x,float); z=z[np.isfinite(z)]
    if not len(z):return {}
    return dict(min=float(z.min()),median=float(np.median(z)),max=float(z.max()))
def midpoint_kept(series,i,j):
    v=np.asarray(series,float)
    if (len(v)-1-j)*500<10000:return None
    if not np.isfinite(v[[i,j]]).all() or not np.isfinite(v[j:]).all():return None
    if v[j]<=v[i]:return False
    return bool(np.min(v[j:])>=(v[i]+v[j])/2)
def tv(v,w,axis=-1):return abs(v-w).sum(axis=axis)/2
def change(d,i,j,k):
    if k=='share_displacement':return float(tv(d[S].iloc[i].to_numpy(),d[S].iloc[j].to_numpy()))
    return float(d[k].iloc[j]-d[k].iloc[i])
def wdiff(d,n,k):
    if k=='share_displacement':
        z=d[S].to_numpy();return tv(z[n:],z[:-n])
    z=d[k].to_numpy();return z[n:]-z[:-n]
def window_record(d,k,wtype,i,j,group):
    qc=d[S].to_numpy();ret={}
    row={'run':k,'wd':float(roster[k]['wd']),'beta':float(roster[k]['beta']),'seed':int(roster[k]['seed']),
         'group':group,'window':wtype,'start':500*i,'end':500*j,'later_steps':(len(d)-1-j)*500}
    for col in trackcols+['log_A','test_acc']+S:
        row[col+'_start']=float(d[col].iloc[i]);row[col+'_end']=float(d[col].iloc[j])
        row[col+'_change']=row[col+'_end']-row[col+'_start']
    for col in ['C_sd','C_add','P','log_A','D']:row[col+'_kept']=midpoint_kept(d[col],i,j)
    row['share_displacement_change']=float(tv(qc[i],qc[j]));dist=tv(qc[i],qc[j])
    row['share_displacement_kept']=None if row['later_steps']<10000 or dist==0 else bool(np.all(tv(qc[j:],qc[j])<=tv(qc[j:],qc[i])))
    row['largest_gain_mode']=int(1+np.argmax(qc[j]-qc[i]))
    for label,m in [('start',i),('end',j)]:row['top_dir_'+label]=int(d['largest_direction'].iloc[m])
    return row

def main():
    inputs={str(p):sha(p) for p in [scan/'cohort_105.csv',scan/'results/300k_runs.csv',summaryfile,polarfile,o35]}
    plan=read(out/'requested_steps.json');parts=sorted((out/'parts').glob('*.csv'))
    replay=pd.concat([pd.read_csv(p) for p in parts],ignore_index=True)
    assert len(replay)==52305 and not replay.duplicated(['run','step']).any()
    replay=replay.merge(pvals.rename(columns={'order_parameter':'P'}),on=['run','step'],validate='one_to_one',how='left')
    assert replay.P.notna().all()
    curves={k:d.sort_values('step').reset_index(drop=True).copy() for k,d in replay.groupby('run')}
    assert set(curves)==set(roster)
    records={};events=[];vis=[];windows={};outcomes=[]
    for k,r in roster.items():
        d=curves[k];assert list(d.step)==plan[k]
        f=frozen[k];base=Path(r['parent_dir'])
        dest=Path(r['previous_300k']) if r['previous_300k'] else scan/'continuation_300k'/k if r['status'].startswith('continue_') else base
        hp=dest/'history.json';inputs[str(hp)]=sha(hp);h=read(hp);assert_history(h,int(d.step.max()))
        if dest!=base:check_prefix(read(base/'history.json'),h)
        lab=audit(h);assert bool(lab['audit_fixed_grokking'])==(f['strict_by_cut']=='True')
        h=[z for z in h if z['step']%500==0]
        assert list(d.step)==[z['step'] for z in h]
        for c in ['test_acc','train_acc','D','raw_D','I','rho']:d[c]=[z[c] for z in h]
        assert ((d.test_acc>=.9).any())==bool(f['first_test90']) and (d.A>0).all()
        vals=d[['s1','s2','s3','s4']].to_numpy();np.testing.assert_allclose(vals.sum(axis=1),d.A,rtol=1e-12)
        shares=vals/d.A.to_numpy()[:,None];d[S]=shares;d['log_A']=np.log(d.A)
        assert np.allclose(shares.sum(1),1,rtol=0,atol=1e-12)
        d['beta']=float(r['beta']);d['wd']=float(r['wd']);d['seed']=int(r['seed'])
        group='strict' if lab['audit_fixed_grokking'] else 'other90' if f['first_test90'] else 'never90'
        max_test=max(v['test_acc'] for v in h)
        outcomes.append(dict(run=k,wd=float(r['wd']),beta=float(r['beta']),seed=int(r['seed']),group=group,last=int(d.step.max()),
                             max_test=max_test,train99=lab['t_mem'] is not None))
        if group!='never90':
            j=int(f['first_test90'])//500
            plotlo=max(i for i,v in enumerate(h[:j]) if v['test_acc']<=.1)
            vis.append(window_record(d,k,'last_low_to_first90',plotlo,j,group))
            if group=='strict':
                i=lab['t_low_end']//500;assert j>i
                if k in fixed90.index:assert i*500==fixed90.at[k,'start'] and j*500==fixed90.at[k,'end']
                events.append(window_record(d,k,'original_strict',i,j,group))
                windows[k]=(i,j)
        curves[k]=d
    o=pd.DataFrame(outcomes).set_index('run');ev=pd.DataFrame(events);wv=pd.DataFrame(vis)
    assert len(ev)==58 and len(wv)==83 and (o.group=='never90').sum()==22
    write(o.reset_index(),'outcomes.csv');write(ev,'strict_events.csv');write(wv,'all_first_10_90_events.csv')
    current=ev.set_index('run');pairs=[];percase=[];controls_graph={}
    for k,(i,j) in windows.items():
        row=o.loc[k]
        ctrl=o.index[(o.group=='never90')&(o.wd==row.wd)&(o.beta==row.beta)].tolist()
        if not ctrl:continue
        controls_graph[k]=ctrl
        for metric in analysed:
            target=float(current.at[k,metric+'_change'])
            p=[]
            for cc in ctrl:
                c=curves[cc]; assert len(c)==601
                same=change(c,i,j,metric);diff=wdiff(c,j-i,metric)
                assert np.isclose(diff[i],same)
                exceed=diff>=target
                line=dict(run=k,control=cc,metric=metric,strict_change=target,control_same_clock=same,
                          control_max_change=float(np.max(diff)),windows=len(diff),windows_ge=int(exceed.sum()),
                          fraction_ge=float(np.mean(exceed)),control_below10=bool(o.at[cc,'max_test']<=.1),control_train99=bool(o.at[cc,'train99']))
                pairs.append(line);p.append(line)
            percase.append(dict(run=k,metric=metric,strict_change=target,controls=len(p),
                              control_same_clock_median=float(np.median([a['control_same_clock'] for a in p])),
                              strongest_any_control=max(a['control_max_change'] for a in p),
                              control_fraction_ge=sum(a['windows_ge'] for a in p)/sum(a['windows'] for a in p),
                              all_same_clock_smaller=all(target>a['control_same_clock'] for a in p)))
    pp=pd.DataFrame(pairs);case=pd.DataFrame(percase);write(pp,'matched_pairs_and_full_control.csv');write(case,'matched_case_summary.csv')
    exact_ref=read(out.parent/'three_seed_t50_300k/../all35_seed_means/verification.json');assert exact_ref['cells']==35
    cells={(float(r.weight_decay),float(r.beta)):r for r in settings.itertuples()}
    def origin(k):
        meta=o.loc[k];return float(getattr(cells[float(meta.wd),float(meta.beta)],'t50_seed'+str(int(meta.seed))))
    # Matched events carry equal weights. Each event's same-cell controls collectively carry its weight.
    t=np.arange(-30000,30001,500);cmpcols=['test_acc','C_sd','C_add','A','D','P']+S
    matched_curves=[]
    for k,ctrl in controls_graph.items():
        anchor=origin(k);grid=t+anchor
        for label,ids in [('strict',[k]),('control',ctrl)]:
            for metric in cmpcols:
                mats=[]
                for cid in ids:
                    d=curves[cid];assert grid.min()>=0 and grid.max()<=d.step.max()
                    mats.append(np.interp(grid,d.step,d[metric]))
                y=np.mean(mats,axis=0);matched_curves.extend((k,label,metric,int(tm),val) for tm,val in zip(t,y))
    mm=pd.DataFrame(matched_curves,columns=['event_run','series','metric','relative_step','value'])
    gm=mm.groupby(['series','metric','relative_step']).value.agg(['mean','std','size']).reset_index()
    assert (gm['size']==6).all()
    write(gm,'matched_means.csv');write(mm,'matched_component_curves.csv.gz')
    # Control-axis audit matches the archived all-three-seeds, maximum-absolute-adjacent-slope definition.
    control_axis=[]
    for time in (100000,200000,300000):
        for wd in sorted(o.wd.unique()):
            ids=o.loc[(o.wd==wd)&(o['last']>=time)]
            rr={'step':time,'wd':wd,'n_real':len(ids),'complete_slice':len(ids)==15}
            if len(ids)==15:
                bsorted=sorted(ids.beta.unique());vals=[]
                for seed in range(3):
                    yy=[curves[ids.index[(ids.seed==seed)&(ids.beta==b)][0]]['C_sd'].iloc[time//500] for b in bsorted]
                    vals.append(yy)
                diff=np.diff(vals,axis=1);sign=np.sign(diff);idx=np.argmax(np.abs(diff),axis=1)
                unique=all(np.sum(abs(abs(d)-max(abs(d)))<1e-12)==1 for d in diff)
                shared=bool(unique and len(set(idx))==1 and len(set(sign[np.arange(3),idx]))==1)
                rr.update(shared=shared)
                if shared:
                    ix=idx[0];aa=np.array(vals);rr.update(lower=bsorted[ix],upper=bsorted[ix+1],direction=int(sign[0,ix]),
                        sides_disjoint=bool(aa[:,ix].max()<aa[:,ix+1].min() or aa[:,ix+1].max()<aa[:,ix].min()),
                        deltas=list(diff[:,ix]))
            control_axis.append(rr)
    write(pd.DataFrame(control_axis),'beta_slices.csv')
    data=pd.concat(curves.values(),ignore_index=True);write(data.drop(columns=['checkpoint_sha']),'trajectories.csv.gz')
    cases_sr={}
    for metric in analysed:
        e=ev[metric+'_change'];p=case[case.metric==metric]
        sr={'change':stats(e),'larger_same_clock':int((p.strict_change>p.control_same_clock_median).sum()),
            'above_every_control_window':int((p.strict_change>p.strongest_any_control).sum()),
            'full_control_fraction_ge':stats(p.control_fraction_ge)}
        if metric!='share_displacement':
            sr.update(up=int((e>0).sum()),down=int((e<0).sum()),flat=int((e==0).sum()),
                      med_start=float(ev[metric+'_start'].median()),med_end=float(ev[metric+'_end'].median()))
        follow=ev[metric+'_kept'].dropna()
        sr.update(enough_followup=int(len(follow)),retained=int(follow.sum()))
        cases_sr[metric]=sr
    finite={}
    for g,ww in wv.groupby('group'):
        finite[g]={'n':len(ww),'C_sd_up':int((ww.C_sd_change>0).sum()),'C_add_up':int((ww.C_add_change>0).sum()),
                   'D_up':int((ww.D_change>0).sum()),'A_up':int((ww.log_A_change>0).sum()),
                   'P_up':int((ww.P_change>0).sum()),'C_sd_delta':stats(ww.C_sd_change),
                   'top_add_before':int((ww.top_dir_start==1).sum()),'top_add_after':int((ww.top_dir_end==1).sum())}
    countdiff=max(data.test_count_diff.abs().max(),data.train_count_diff.abs().max())
    result=dict(points=len(data),runs=len(o),strict=len(ev),secondary90=len(wv)-len(ev),never90=int((o.group=='never90').sum()),
        near_low=int((o.max_test<=.1).sum()),near_low_trained=int(((o.max_test<=.1)&o.train99).sum()),
        original_vs_visual_start_diff=int((ev.set_index('run').start!=wv[wv.group=='strict'].set_index('run').start.reindex(ev.run)).sum()),
        matched=len(controls_graph),unique_controls=len(set(y for x in controls_graph.values() for y in x)),
        pairs=sum(map(len,controls_graph.values())),matched_configs=len({(roster[k]['beta'],roster[k]['wd']) for k in controls_graph}),
        matched_metrics=cases_sr,first10_to90=finite,largest_share_gain_counts={str(k):int(n) for k,n in ev.largest_gain_mode.value_counts().sort_index().items()},
        winners_add_pre=int((ev.top_dir_start==1).sum()),winners_add_post=int((ev.top_dir_end==1).sum()),
        actual_beta_slices=control_axis,matched_runids=controls_graph,
        data_validation=dict(max_prior_C_error=float(data.cache_error.max()),n_invalid_input_rows=int((data.invalid>0).sum()),
                             max_absolute_accuracy_count_discrepancy=int(countdiff),nonzero_test_count_discrepancies=int((data.test_count_diff!=0).sum())))
    dump('summary.json',result);dump('input_hashes.json',inputs)
    print('core',json.dumps({k:v for k,v in result.items() if k not in ('actual_beta_slices','matched_runids')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

