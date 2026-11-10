#!/usr/bin/env python3
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,csv,gzip,hashlib,collections,math,html
B=Path(__file__).resolve().parent
ROOT=Path('/home/chenzhe/work/tglt_experiments')
OLD=ROOT/'analysis/all_addition_grokking_event_audit_20261026/all_addition_labels.csv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvwrite(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (B/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def esc(x):return html.escape(str(x))
def step(v):return '—' if v is None or v=='' else format(int(v),',')
def pct(v):return '—' if v is None or v=='' else '%.2f%%'%(100*float(v))
REASONS={'grokking':'通过原三条数值条件','never_reached_train99':'训练未达到99%',
 'test_above10_at_memorization':'首次训练99%时测试已超过10%',
 'plateau_shorter_than10k':'连续低平台不足10k',
 'test_never_reached90':'测试未达到90%',
 'transition_not_sharp_enough':'跃迁宽度不小于低平台的一半'}
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
def graph(row,h):
    width,height=950,255;left,right,top,bottom=55,18,20,43
    W=width-left-right;H=height-top-bottom
    x=lambda s:left+W*float(s)/100000
    y=lambda a:top+H*(1-float(a))
    color='#D55E00' if row['old_fixed_grokking'] else '#0072B2'
    out=['<svg viewBox="0 0 %d %d" role="img" aria-label="%s原始训练和测试准确率">'%(width,height,esc(row['number']))]
    out.append('<rect width="100%" height="100%" fill="white"/>')
    lo,hi=row['t_low_end'],row['t90']
    if lo is not None and hi is not None and hi>lo:
        out.append('<rect x="%.3f" y="%d" width="%.3f" height="%d" fill="#eeeeee"/>'%(x(lo),top,x(hi)-x(lo),H))
    for val in [0,.1,.5,.9,1]:
        if val in [.1,.9]:
            out.append('<line x1="%d" x2="%d" y1="%.2f" y2="%.2f" stroke="#bbbbbb" stroke-width=".6" stroke-dasharray="3 3"/>'%(left,width-right,y(val),y(val)))
        out.append('<text x="%d" y="%.2f" text-anchor="end" font-family="Arial,sans-serif" font-size="11">%d%%</text>'%(left-7,y(val)+4,val*100))
    for t in range(0,100001,20000):
        out.append('<text x="%.2f" y="%d" text-anchor="middle" font-family="Arial,sans-serif" font-size="11">%dk</text>'%(x(t),height-bottom+20,t//1000))
    out.append('<path d="M%d %dV%dH%d" fill="none" stroke="#333" stroke-width=".8"/>'%(left,top,height-bottom,width-right))
    if row['t_mem'] is not None:
        out.append('<line x1="%.2f" x2="%.2f" y1="%d" y2="%d" stroke="#777" stroke-width=".6" stroke-dasharray="2 3"/>'%(x(row['t_mem']),x(row['t_mem']),top,height-bottom))
    for key,c,dash in [('train_acc','#555555',' stroke-dasharray="5 3"'),('test_acc',color,'')]:
        pts=' '.join('%.3f,%.3f'%(x(z['step']),y(z[key])) for z in h)
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1"%s/>'%(pts,c,dash))
    out.append('<text x="%d" y="%d" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">Original training steps</text>'%(width//2,height-5))
    out.append('</svg>')
    return ''.join(out)
def main():
    protocol={'scope':'All192 first-generation modular-addition runs behind previous32G averages. Preserve all other tasks outside scope.',
      'numbering':'G01-G32: previously labeled Grokking. N001-N160: previously labeled nonG. Numeric sort r,T,M,seed within each old group. Numbers encode previous labels, not a new morphology verdict.',
      'fixed_criteria':['first saved train_acc>=0.99','test_acc remains<=0.10 from that point for>=10000 saved-step span',
                       'first test_acc>=0.90; width from final contiguous low-platform saved point must be positive and<half plateau span'],
      'no_extra_thresholds':'No new plateau, slope or sustained-accuracy requirement introduced. Training dips and later test relapse reported separately, not used to relabel.',
      'sampling':'Original histories, normally500-step records. No interpolation, smoothing, t50 alignment or averaging.',
      'old_labels_windows':'read-only; audit results separate; all192 rows retained',
      'CPU_only':True,'training':False,'model_replay':False}
    (B/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    old=list(csv.DictReader(OLD.open()));assert len(old)==192
    old.sort(key=lambda z:tuple(int(z[k]) for k in ['r','T','M','seed']))
    hashes={str(OLD):sha(OLD)}
    original=list(csv.DictReader((ROOT/'analysis/seed_consistency_20261026/runs.csv').open()))
    origadd={r['run_id'] for r in original if r['task']=='add'}
    assert origadd=={r['run_id'] for r in old}
    mp=ROOT/'analysis/grokking_specificity_comparison_20261026/pairs.csv'
    matched={r['nonG_run'] for r in csv.DictReader(mp.open())};assert len(matched)==19
    hashes[str(mp)]=sha(mp)
    rows=[];hist={};diff=[];counts=collections.Counter();maxfloat=0.;max_count_acc_error=0.
    for oldrow in old:
        hp=Path(oldrow['history_path']);hashes[str(hp)]=sha(hp);assert hashes[str(hp)]==oldrow['history_sha256']
        cp=Path(oldrow['config_path']);hashes[str(cp)]=sha(cp)
        cfg=json.loads(cp.read_text());h=json.loads(hp.read_text());r=audit(h)
        assert cfg['task']=='add' and int(cfg['p'])==59 and int(cfg['layers'])==2
        assert all(int(cfg[k])==int(oldrow[k]) for k in ['r','T','M','seed'])
        oldG=oldrow['fixed_grokking']=='True';prefix='G' if oldG else 'N';counts[prefix]+=1
        number=prefix+('%02d'%counts[prefix] if oldG else '%03d'%counts[prefix])
        mism=[]
        for k in ['t_mem','t_low_end','t90','plateau_length','jump_width','reason','final_step']:
            a=oldrow[k];b=r[k]
            if a!=('' if b is None else str(b)):mism.append(k)
        for k in ['test_at_mem','plateau_test_max','plateau_test_mean','final_train_acc','final_test_acc']:
            a=oldrow[k];b=r[k]
            if b is None:
                if a!='':mism.append(k)
            else:
                e=abs(float(a)-b);maxfloat=max(maxfloat,e)
                if e>1e-12:mism.append(k)
        if oldG!=r['audit_fixed_grokking']:mism.append('fixed_grokking')
        if mism:diff.append({'run_id':oldrow['run_id'],'mismatch_fields':mism})
        for z in h:
            for kind in ['train','test']:
                if kind+'_correct' in z and kind+'_n' in z:
                    max_count_acc_error=max(max_count_acc_error,abs(z[kind+'_acc']-z[kind+'_correct']/z[kind+'_n']))
        row={'number':number,'run_id':oldrow['run_id'],'config_id':oldrow['config_id'],'r':int(oldrow['r']),'T':int(oldrow['T']),
            'M':int(oldrow['M']),'seed':int(oldrow['seed']),'old_fixed_grokking':oldG,
            'old_reason':oldrow['reason'],'input_reachable':oldrow['input_reachable']=='True',
            'previous_analysis_included':oldrow['input_reachable']=='True',
            'previous_same_config_control':oldrow['run_id'] in matched,**r,
            'audit_matches_old':not mism,'history_path':str(hp),'history_sha256':hashes[str(hp)],
            'config_path':str(cp),'config_sha256':hashes[str(cp)],'old_source_sha256':oldrow['source_sha256']}
        rows.append(row);hist[number]=h
    assert counts=={'G':32,'N':160}
    rows.sort(key=lambda z:(not z['old_fixed_grokking'],z['number']))
    csvwrite('all_192_numbered.csv',rows);csvwrite('old_G_32_numbered.csv',[r for r in rows if r['old_fixed_grokking']])
    csvwrite('old_nonG_160_numbered.csv',[r for r in rows if not r['old_fixed_grokking']])
    csvwrite('matched_19_controls_numbered.csv',[r for r in rows if r['previous_same_config_control']])
    csvwrite('number_map.csv',[{k:r[k] for k in ['number','run_id','old_fixed_grokking','audit_fixed_grokking','input_reachable','previous_analysis_included','previous_same_config_control']} for r in rows])
    with gzip.open(B/'original_accuracy_curves.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['number','run_id','step','train_acc','test_acc','train_ce','test_ce']);w.writeheader()
        for r in rows:
            for z in hist[r['number']]:
                w.writerow({'number':r['number'],'run_id':r['run_id'],**{k:z.get(k) for k in ['step','train_acc','test_acc','train_ce','test_ce']}})
    G=[r for r in rows if r['old_fixed_grokking']]
    summary={'total':len(rows),'old_G':len(G),'old_nonG':160,'audit_G':sum(r['audit_fixed_grokking'] for r in rows),
       'label_or_field_mismatches':diff,'old_G_width_range':[min(r['jump_width'] for r in G),max(r['jump_width'] for r in G)],
       'old_G_train_dip_below99_before90':sum(r['train_min_mem_to_t90']<.99 for r in G),
       'old_G_test_relapse_below90_after90':sum(r['test_min_after_t90']<.9 for r in G),
       'reason_counts':dict(collections.Counter(r['reason'] for r in rows)),
       'previous_analysis_population':{'G':sum(r['old_fixed_grokking'] and r['previous_analysis_included'] for r in rows),
                                       'nonG':sum(not r['old_fixed_grokking'] and r['previous_analysis_included'] for r in rows),
                                       'excluded_not_reachable':sum(not r['previous_analysis_included'] for r in rows)},
       'matched_controls':[r['number'] for r in rows if r['previous_same_config_control']]}
    (B/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    text=['第一代模加全部192条运行：编号与原始历史核查',
          'G01-G32保留原32条标签；N001-N160保留原160条非G标签。编号不是新的形态裁决。',
          '固定数值规则：训练首次>=99%；从此开始测试<=10%连续>=10000步；10%到90%宽度<平台长度/2。',
          '旧规则没有要求训练持续>=99%，没有跃迁绝对时长上限，也没有要求到90%后不回落。',
          '本次逐条重读原始历史，原标签和时间字段差异数：'+str(len(diff)),
          '全部原始曲线见report.html；不平滑，不对齐，不平均。',
          '所有时间单位为训练步；原始记录间隔通常500步。原32条的跃迁宽度范围6500—28500步。',
          '训练最低列取首次训练99%至首次测试90%；没有达到测试90%者取到训练结束。',
          '']
    headers=['编号','完整运行名','复算原数值规则','t_mem','平台测试最大','平台测试平均','平台长度','跃迁起点','t90','跃迁宽度','最终测试','训练最低','训练最低时刻','不满足原因']
    text.append('\t'.join(headers))
    for r in rows:
        text.append('\t'.join([r['number'],r['run_id'],'通过' if r['audit_fixed_grokking'] else '未通过',
            step(r['t_mem']),pct(r['plateau_test_max']),pct(r['plateau_test_mean']),step(r['plateau_length']),
            step(r['t_low_end']),step(r['t90']),step(r['jump_width']),pct(r['final_test_acc']),
            pct(r['train_min_mem_to_t90']),step(r['train_min_step']),REASONS[r['reason']]]))
    (B/'完整编号清单.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    md=['|编号|运行|t_mem|平台最大/平均|平台长度|10%→90%宽度|最终测试|','|---|---|---:|---:|---:|---:|---:|']
    for r in G:
        md.append('|%s|%s|%s|%s / %s|%s|%s|%s|'%(r['number'],r['run_id'],step(r['t_mem']),pct(r['plateau_test_max']),pct(r['plateau_test_mean']),step(r['plateau_length']),step(r['jump_width']),pct(r['final_test_acc'])))
    (B/'G32_table.txt').write_text('\n'.join(md))
    config_groups={}
    for r in rows:
        config_groups.setdefault(r['config_id'],{})[int(r['seed'])]=r['number']
    grid_lines=['|配置|seed0|seed1|seed2|','|---|---|---|---|']
    for config,dd in sorted(config_groups.items(),key=lambda a:tuple(int(x[1:]) for x in a[0].split('_'))):
        assert set(dd)=={0,1,2}
        grid_lines.append('|'+config+'|'+'|'.join(dd[z] for z in range(3))+'|')
    (B/'config_number_grid.txt').write_text('\n'.join(grid_lines))
    style='''body{max-width:1500px;margin:25px auto;padding:0 20px;font:16px/1.6 Arial,sans-serif;color:#222}h1{font-size:25px}h2{font-size:21px}a{color:#006a8f}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{border:1px solid #ddd;padding:5px 9px}th{background:#eee;position:sticky;top:0}.scroll{overflow:auto;max-height:70vh}.note{background:#fff3df;padding:15px;border-left:4px solid #c96b00}details{margin:16px 0;border-top:1px solid #bbb;padding-top:10px}summary{cursor:pointer;font-weight:bold}svg{width:100%;height:auto}input{padding:9px;width:330px}button{padding:8px 12px;margin:5px}.small{font-size:13px;color:#555}.none{color:#777}'''
    out=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>192条原始曲线编号核查</title><style>'+style+'</style>',
         '<h1>原32条Grokking与全部非Grokking：逐条编号核查</h1>',
         '<div class="note"><b>按此前三条数值条件复算，32条原G标签没有计算差异。但不能据此把32条都称为形态典型的“训练稳定到顶后，测试突然跃升”。</b><br>它们的10%→90%宽度为6,500—28,500步；24条首次达到训练99%后、首次测试90%前出现过训练准确率回落。原规则没有要求训练持续保持99%，也没有规定跃迁的绝对时长上限。本轮不增加新规则，不修改旧标签，只把事实完整列出。</div>',
         '<p>本清单是第一代模加的全部192条，不是全部852条多任务实验。G01—G32保留原G标签；N001—N160保留原非G标签。此前内部指标分析使用输入可达的132条，即32条G和100条非G；另60条输入不可达、训练未达到99%，现在也完整列出。此前蓝色同配置对照是其中19条，单独标注。</p>',
         '<p><b>原数值规则：</b>①首次训练准确率≥99%；②从该点开始，测试准确率连续≤10%至少10,000步；③测试之后首次达到90%，从低平台最后保存点到该点的宽度必须大于0且小于平台长度的一半。按原保存点判断，通常500步一测。没有用末期准确率或t90延迟代替这三项。</p>',
         '<p><a href="完整编号清单.txt">全部192条纯文本清单</a> · <a href="all_192_numbered.csv">全部数值CSV</a> · <a href="old_G_32_numbered.csv">原G32条</a> · <a href="old_nonG_160_numbered.csv">原非G160条</a> · <a href="matched_19_controls_numbered.csv">此前同配置对照19条</a> · <a href="checks.json">核验</a></p>',
         '<p><input id="query" placeholder="搜索编号、运行名、原因" oninput="filterRows()"><button onclick="setGroup(\'all\')">全部192条</button><button onclick="setGroup(\'G\')">原G32条</button><button onclick="setGroup(\'N\')">原非G160条</button><button onclick="setGroup(\'matched\')">此前19条蓝色对照</button><span id="nshown"></span></p>',
         '<div class="scroll"><table id="index"><thead><tr>'+''.join('<th>'+x+'</th>' for x in ['编号/曲线','完整运行名','原标签','复算','t_mem','平台最大/平均','平台长度','10%起点','首次90%','跃迁宽度','宽度/平台','最终测试','训练最低','原判据原因','此前分析'])+'</tr></thead><tbody>']
    for r in rows:
        group='G' if r['old_fixed_grokking'] else 'N'
        vals=['<a href="#%s" onclick="openRun(\'%s\')">%s</a>'%(r['number'],r['number'],r['number']),esc(r['run_id']),
          'G' if r['old_fixed_grokking'] else '非G','通过' if r['audit_fixed_grokking'] else '未通过',
          step(r['t_mem']),pct(r['plateau_test_max'])+' / '+pct(r['plateau_test_mean']),step(r['plateau_length']),
          step(r['t_low_end']),step(r['t90']),step(r['jump_width']),
          '—' if r['width_over_plateau'] is None else '%.3f'%r['width_over_plateau'],
          pct(r['final_test_acc']),pct(r['train_min_mem_to_t90']),REASONS[r['reason']],
          '同配置蓝色对照' if r['previous_same_config_control'] else ('纳入132条' if r['previous_analysis_included'] else '未纳入：输入不可达')]
        out.append('<tr data-group="%s" data-matched="%s">'%(group,str(r['previous_same_config_control']).lower())+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
    out.append('</tbody></table></div><h2>逐条原始训练/测试曲线</h2><p>点击编号或下方标题展开。全部使用原始0—100k步横轴、0—100%纵轴，不平均、不对齐、不平滑。灰色虚线是训练准确率；橙色实线是原G测试准确率；蓝色实线是原非G测试准确率。浅灰区为低平台末点至首次90%，只标时间区间，不代表已经证实“陡跃”。</p>')
    for r in rows:
        num=r['number']
        out.append('<details class="run" id="%s"><summary>%s · %s · %s</summary>'%(num,num,esc(r['run_id']),REASONS[r['reason']]))
        out.append('<p>训练首次99%%：%s；当时测试：%s。低平台最大/平均：%s / %s；平台长度：%s步。10%%→90%%区间：%s—%s；宽度：%s步。最终训练/测试：%s / %s。</p>'%(step(r['t_mem']),pct(r['test_at_mem']),pct(r['plateau_test_max']),pct(r['plateau_test_mean']),step(r['plateau_length']),step(r['t_low_end']),step(r['t90']),step(r['jump_width']),pct(r['final_train_acc']),pct(r['final_test_acc'])))
        out.append(graph(r,hist[num]))
        out.append('<p>训练首次99%%至测试首次90%%期间，训练最低为%s（%s步），低于99%%的保存点为%d/%d。首次测试90%%之后，测试最低为%s（%s步）。这些是形态核查信息，没有追加为判定规则。</p>'%(pct(r['train_min_mem_to_t90']),step(r['train_min_step']),r['train_below99_points_mem_to_t90'],r['train_samples_mem_to_t90'],pct(r['test_min_after_t90']),step(r['test_min_after_t90_step'])))
        out.append('<p class="small">原历史：'+esc(r['history_path'])+'<br>SHA256：'+r['history_sha256']+'</p></details>')
    out.append('''<script>let group='all';function filterRows(){let q=document.getElementById('query').value.toLowerCase(),n=0;document.querySelectorAll('#index tbody tr').forEach(r=>{let ok=(group==='all'||r.dataset.group===group||(group==='matched'&&r.dataset.matched==='true'))&&r.textContent.toLowerCase().includes(q);r.hidden=!ok;if(ok)n++});document.getElementById('nshown').textContent='显示 '+n+' 条'}function setGroup(g){group=g;filterRows()}function openRun(id){document.getElementById(id).open=true}window.addEventListener('hashchange',()=>{let e=document.getElementById(location.hash.slice(1));if(e&&e.tagName==='DETAILS')e.open=true});if(location.hash)openRun(location.hash.slice(1));filterRows();</script></html>''')
    (B/'report.html').write_text(''.join(out),encoding='utf-8')
    # Plain SVG previews use the same numerical coordinates as the embedded raw plots.
    (B/'preview_G01.svg').write_text(graph(G[0],hist[G[0]['number']]).replace('<svg viewBox=', '<svg xmlns="http://www.w3.org/2000/svg" viewBox=',1))
    checks={'histories_recomputed':192,'unique_numbers':len({r['number'] for r in rows}),
      'unique_run_ids':len({r['run_id'] for r in rows}),'matches_original_addition_inventory':True,
      'two_independent_classifiers_agree':True,'label_or_time_field_mismatches':diff,
      'old_float_field_max_error':maxfloat,'accuracy_vs_integer_count_max_error':max_count_acc_error,
      'unchanged_input_hashes':all(sha(p)==v for p,v in hashes.items()),
      'native_history_points':sum(len(h) for h in hist.values()),'all192_raw_curves_embedded':len([q for q in out if '<svg viewBox=' in q])==192,
      'CPU_only':True,'model_replay':False,'training':False,'old_labels_modified':False}
    assert checks['unique_numbers']==192 and checks['unchanged_input_hashes']
    (B/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    (B/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
    files=[p for p in B.iterdir() if p.is_file() and p.name not in ['manifest.json','delivery_files.txt']]
    mf=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
    (B/'manifest.json').write_text(json.dumps(mf,ensure_ascii=False,indent=2))
    (B/'delivery_files.txt').write_text('\n'.join(r['path'] for r in mf)+'\nmanifest.json\n')
    print(json.dumps({'summary':summary,'checks':checks},ensure_ascii=False))
if __name__=='__main__':main()
