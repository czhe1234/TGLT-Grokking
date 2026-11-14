"""Original strict Grokking audit, extracted unchanged."""
import math
def audit(h):
    s=[int(v['step']) for v in h]
    tr=[float(v['train_acc']) for v in h];te=[float(v['test_acc']) for v in h]
    assert s==sorted(set(s)) and all(0<=x<=1 for x in tr+te)
    mem_indices=[i for i,v in enumerate(tr) if v>=.99]
    high_indices=[i for i,v in enumerate(te) if v>=.9]
    mem=mem_indices[0] if mem_indices else None
    high=high_indices[0] if high_indices else None
    low=None;loidx=[];length=0
    if mem is not None and te[mem]<=.1:
        end=next((i for i in range(mem,len(h)) if te[i]>.1),len(h))
        loidx=list(range(mem,end));low=end-1;length=s[low]-s[mem]
    width=s[high]-s[low] if high is not None and low is not None else None
    c1=mem is not None;c2=length>=10000
    c3=width is not None and 0<width<length/2
    qualified=bool(c1 and c2 and c3)
    if qualified:reason='grokking'
    elif mem is None:reason='never_reached_train99'
    elif te[mem]>.1:reason='test_above10_at_memorization'
    elif length<10000:reason='plateau_shorter_than10k'
    elif high is None:reason='test_never_reached90'
    else:reason='transition_not_sharp_enough'
    # Independent sequential implementation of the original three thresholds.
    mt=None;le=None;f90=None;closed=False
    for z in h:
        if f90 is None and z['test_acc']>=.9:f90=int(z['step'])
        if mt is None and z['train_acc']>=.99:mt=int(z['step'])
        if mt is not None and not closed:
            if z['test_acc']<=.1:le=int(z['step'])
            else:closed=True
    leng=0 if le is None or mt is None else le-mt
    wid=None if f90 is None or le is None else f90-le
    q2=bool(mt is not None and leng>=10000 and wid is not None and 0<wid<leng/2)
    assert q2==qualified and leng==length and wid==width
    mrange=list(range(mem,(high+1 if high is not None and high>=mem else len(h)))) if mem is not None else []
    jmin=min(mrange,key=lambda i:tr[i]) if mrange else None
    dips=[i for i in mrange if tr[i]<.99]
    after=list(range(high,len(h))) if high is not None else []
    jtest=min(after,key=lambda i:te[i]) if after else None
    return dict(t_mem=None if mem is None else s[mem],test_at_mem=None if mem is None else te[mem],
       plateau_length=length,t_low_end=None if low is None else s[low],
       first_test_above10_after_mem=None if low is None or low==len(h)-1 else s[low+1],
       plateau_test_max=max(te[i] for i in loidx) if loidx else None,
       plateau_test_mean=math.fsum(te[i] for i in loidx)/len(loidx) if loidx else None,
       plateau_saved_points=len(loidx),
       t50=next((s[i] for i,v in enumerate(te) if v>=.5),None),
       t90=None if high is None else s[high],jump_width=width,
       width_over_plateau=None if width is None or length==0 else width/length,
       criterion1=c1,criterion2=c2,criterion3=c3,audit_fixed_grokking=qualified,reason=reason,
       final_step=s[-1],final_train_acc=tr[-1],final_test_acc=te[-1],
       train_min_mem_to_t90=None if jmin is None else tr[jmin],
       train_min_step=None if jmin is None else s[jmin],
       train_below99_points_mem_to_t90=len(dips),
       train_first_below99_after_mem=None if not dips else s[dips[0]],
       train_samples_mem_to_t90=len(mrange),
       test_min_after_t90=None if jtest is None else te[jtest],
       test_min_after_t90_step=None if jtest is None else s[jtest],
       test_below90_points_after_t90=sum(te[i]<.9 for i in after),
       sampling_gaps=sorted(set(s[i]-s[i-1] for i in range(1,len(s)))))
