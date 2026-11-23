"""Mirror analysis summaries in local SwanLab and verify its API against the CSV-derived values."""
import collections
import json
import math
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES='',SWANLAB_MODE='local')
HERE=Path(__file__).resolve().parent
LOG=HERE/'swanlab'/'logs'
PORT=16225

def curves():
    s=json.loads((HERE/'summary.json').read_text())
    out=collections.defaultdict(dict)
    for r in s['h12']:
        out[f'H12_s{r["seed"]}'][r['step']]={'native_correct':r['native'],'own_rescued':r['own'],
            'mean_wrong_rescued':r['mean_wrong'],'best_rival_advantage':r['margin_adv'],
            'fixed_rival_advantage':r['fixed_adv']}
    for r in s['cross']:
        name=f'{r["task"]}_H{r["H"]}_s{r["seed"]}'
        out[name][r['replayed_step']]={'native_correct':r['native'],'own_rescued':r['own'],
            'mean_wrong_rescued':r['mean_wrong'],'best_rival_advantage':r['margin_adv']}
    for r in s['geometry']:
        name=f'loc_{r["task"]}_s{r["seed"]}'
        metrics={'native_correct':r['native'],'all_L1_joint_off_correct':r['all_L1']}
        if r['query'] is not None:metrics['query_term_off_correct']=r['query']
        out[name][r['H']]=metrics
    return out

def verify():
    expected=curves()
    with sqlite3.connect(LOG/'runs.swanlab') as db:
        rows=db.execute('select id,name from experiment').fetchall()
    assert len(rows)==len(expected) and {n for _,n in rows}==set(expected)
    n=0
    for i,name in rows:
        info=expected[name]
        tags={k for r in info.values() for k in r}
        for tag in tags:
            url=f'http://127.0.0.1:{PORT}/api/v1/experiment/{i}/tag/{urllib.parse.quote(tag,safe="")}'
            got={int(r['index']):r['data'] for r in json.load(urllib.request.urlopen(url,timeout=10))['data']['list']}
            wanted={t:r[tag] for t,r in info.items() if tag in r}
            assert got.keys()==wanted.keys() and all(math.isclose(got[t],v,rel_tol=1e-6,abs_tol=1e-8) for t,v in wanted.items()),(name,tag,got,wanted)
            n+=len(got)
    (HERE/'swanlab'/'validation.json').write_text(json.dumps(dict(curves=len(rows),verified_scalars=n,port=PORT,cloud=False),indent=2))
    print(len(rows),n)

def log():
    import swanlab
    LOG.mkdir(parents=True,exist_ok=True)
    for name,values in curves().items():
        swanlab.init(project='TGLT-PNAS-Evidence',experiment_name=name,mode='local',log_dir=str(LOG),
            config={'axis':'H' if name.startswith('loc_') else 'training_step','run':name},
            settings=swanlab.Settings(probe=dict(hardware=False,runtime=False,requirements=False,git=False,monitor=False)))
        for step,data in sorted(values.items()):
            swanlab.log(data,step=step)
        swanlab.finish()

if __name__=='__main__':
    verify() if '--verify' in sys.argv else log()
