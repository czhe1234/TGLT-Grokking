"""An isolated local board for the eight label-free observables."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',SWANLAB_MODE='local',OMP_NUM_THREADS='1')
import argparse,csv,json,math,sqlite3,urllib.request,urllib.parse
from datetime import datetime,timezone
from pathlib import Path
import swanlab
HERE=Path(__file__).resolve().parent
LOG=HERE/'swanlab/logs'
SECTIONS={'空间关联':('1 空间两点关联',0),'线性响应':('2 输入到查询的响应',1),'写回谱':('3 写回协方差谱',2),'通道重叠':('4 Mode间的重叠',3)}
METRICS={'空间关联/第一层_端到端':'G1_0q','空间关联/第一层_两操作数位置':'G1_0b',
 '空间关联/第二层_端到端':'G2_0q','空间关联/第二层_两操作数位置':'G2_0b',
 '线性响应/平均Jacobian范数':'chi_mean','写回谱/谱熵':'S_total','写回谱/有效参与维数':'D_total','通道重叠/Mode3与Mode4':'q34'}

def organize():
    db=sqlite3.connect(str(LOG/'runs.swanlab'));now=datetime.now(timezone.utc).isoformat()
    with db:
        for cid,key,eid,pid in db.execute('SELECT id,name,experiment_id,project_id FROM chart').fetchall():
            family=key.split('/')[0]
            if family not in SECTIONS:continue
            name,sort=SECTIONS[family]
            found=db.execute('SELECT id FROM namespace WHERE name=? AND experiment_id IS ? AND project_id IS ?',(name,eid,pid)).fetchone()
            if found:nid=found[0]
            else:nid=db.execute('INSERT INTO namespace(experiment_id,project_id,name,sort,opened,create_time,update_time) VALUES(?,?,?,?,?,?,?)',(eid,pid,name,sort,1,now,now)).lastrowid
            db.execute('UPDATE display SET namespace_id=? WHERE chart_id=?',(nid,cid))
    db.close()

def sync():
    from swanlab.deprecated.local import _ColumnInfo
    original=_ColumnInfo.__init__
    def init(self,key,kid,chart_type):
        original(self,key,kid,chart_type);self.section_name,self.section_sort=SECTIONS[key.split('/')[0]]
    _ColumnInfo.__init__=init
    LOG.mkdir(parents=True,exist_ok=True);receipts=HERE/'swanlab/receipts';receipts.mkdir(exist_ok=True)
    for row in json.loads((HERE/'roster.json').read_text()):
        receipt=receipts/(row['run_id']+'.json')
        if receipt.exists():
            assert json.loads(receipt.read_text()).get('finished');continue
        name='physics_span%d_s%d'%(row['span'],row['seed']);group='span%d_H08'%row['span']
        config=dict(row,exp_group=group,metrics=METRICS,CPU_only=True,new_training_steps=0,input_count=3481,
            input_labels_used=False,source_curves='records/'+row['run_id']+'/curves.csv')
        swanlab.init(project='TGLT-label-free-physics',experiment_name=name,group=group,config=config,
            color={0:'#0066AD',1:'#D64025',2:'#00856A'}[row['seed']],mode='local',log_dir=str(LOG),
            description='已有检查点，零标签内部量；跨度与seed分开；不含训练或其他实验。',
            settings=swanlab.Settings(probe=dict(hardware=False,runtime=False,requirements=False,git=False,monitor=False)))
        rows=list(csv.DictReader((HERE/config['source_curves']).open()))
        for x in rows:swanlab.log({key:float(x[col]) for key,col in METRICS.items() if x[col]},step=int(x['step']))
        swanlab.finish()
        receipt.write_text(json.dumps(dict(name=name,finished=True,points=len(rows),metrics=METRICS),ensure_ascii=False,indent=2))
    organize();print('SYNC_COMPLETE',flush=True)

def verify(port):
    db=sqlite3.connect(str(LOG/'runs.swanlab'));experiments=db.execute('SELECT id,name FROM experiment').fetchall();db.close()
    roster=json.loads((HERE/'roster.json').read_text());names={'physics_span%d_s%d'%(r['span'],r['seed']):r for r in roster}
    assert len(experiments)==len(names)==5
    checked=0
    for eid,name in experiments:
        if name not in names:name=name.rsplit('-',1)[0]
        r=names[name];rows=list(csv.DictReader((HERE/'records'/r['run_id']/'curves.csv').open()))
        for key,col in METRICS.items():
            url='http://127.0.0.1:%d/api/v1/experiment/%d/tag/%s'%(port,eid,urllib.parse.quote(key,safe=''))
            actual={int(x['index']):float(x['data']) for x in json.load(urllib.request.urlopen(url))['data']['list']}
            expected={int(x['step']):float(x[col]) for x in rows if x[col]}
            assert actual.keys()==expected.keys()
            assert all(math.isclose(actual[k],v,rel_tol=1e-6,abs_tol=1e-7) for k,v in expected.items()),(name,key)
            checked+=len(expected)
    report=dict(experiments=5,scalars_verified=checked,port=port,status='verified',cloud_upload=False)
    (HERE/'validation/swanlab_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify',type=int);args=ap.parse_args()
    if args.verify:verify(args.verify)
    else:sync()
