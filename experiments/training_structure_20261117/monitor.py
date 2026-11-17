"""Record completed checkpoint analysis in an isolated local SwanLab board."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', SWANLAB_MODE='local', OMP_NUM_THREADS='1')
import argparse
import csv
import json
import math
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path
import swanlab

HERE = Path(__file__).resolve().parent
LOG = HERE/'swanlab/logs'
METRICS = {
    '表现/测试准确率（%）': ('test_acc', 100),
    '表现/训练准确率（%）': ('train_acc', 100),
    '表现/测试Margin': ('test_margin', 1),
    '表征结构/规则份额（%）': ('rule_fraction', 100),
    '表征结构/共享分量平方强度B': ('rule_energy', 1),
    '表征结构/类内差异平方强度W': ('within_energy', 1),
    '表征结构/总写回规则份额（%）': ('write_rule_fraction', 100),
    '单步归因/实际规则份额变化': ('delta_rule_fraction', 1),
    '单步归因/当前梯度': ('gradient_delta_rule_fraction', 1),
    '单步归因/历史动量': ('momentum_delta_rule_fraction', 1),
    '单步归因/权重衰减': ('decay_delta_rule_fraction', 1),
    '变化来源/共享分量变化': ('fraction_from_rule_change', 1),
    '变化来源/类内差异变化': ('fraction_from_within_change', 1),
}


def sync():
    log = HERE/'swanlab'
    log.mkdir(exist_ok=True)
    receipt_path = log/'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['state'] == 'complete':
            print('Already recorded; verify existing board.')
            return
        raise RuntimeError('Existing incomplete receipt requires inspection before retry')
    config = json.loads((HERE/'config.json').read_text())
    rows = list(csv.DictReader((HERE/'results/curves.csv').open()))
    roster = json.loads((HERE/'roster.json').read_text())
    receipt = dict(state='running', completed=[], metrics=METRICS, source='results/curves.csv', cloud_upload=False, version=swanlab.__version__)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    for run in roster:
        selected = [row for row in rows if row['run_id'] == run['run_id']]
        swanlab.init(project='TGLT-Training-Structure', experiment_name='H%02d_s%d'%(run['window'],run['seed']),
            group='H%02d'%run['window'], color=['#0072B2','#D55E00','#009E73'][run['seed']],
            config=dict(config, run=run, metrics=METRICS), mode='local', log_dir=str(LOG),
            description='已有检查点的有限单步归因；规则份额使用任务标签；不是新训练或连续续训。',
            settings=swanlab.Settings(probe=dict(hardware=False,runtime=False,requirements=False,git=False,monitor=False)))
        for row in selected:
            swanlab.log({key: float(row[column])*scale for key,(column,scale) in METRICS.items()}, step=int(row['step']))
        swanlab.finish()
        receipt['completed'].append(run['run_id'])
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    receipt.update(state='complete', runs=len(roster), points=len(rows), scalars=len(rows)*len(METRICS))
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def verify(port):
    rows = list(csv.DictReader((HERE/'results/curves.csv').open()))
    roster = json.loads((HERE/'roster.json').read_text())
    by_name = {'H%02d_s%d'%(r['window'],r['seed']): r for r in roster}
    with sqlite3.connect(str(LOG/'runs.swanlab')) as database:
        experiments = database.execute('SELECT id,name FROM experiment').fetchall()
    assert len(experiments) == len(roster) == 9
    count = 0
    seen = set()
    for experiment_id, name in experiments:
        if name not in by_name:
            name = name.rsplit('-',1)[0]
        run = by_name[name]
        assert name not in seen
        seen.add(name)
        selected = [row for row in rows if row['run_id'] == run['run_id']]
        for key,(column,scale) in METRICS.items():
            url='http://127.0.0.1:%d/api/v1/experiment/%d/tag/%s'%(port,experiment_id,urllib.parse.quote(key,safe=''))
            response=json.load(urllib.request.urlopen(url))
            actual={int(row['index']):float(row['data']) for row in response['data']['list']}
            expected={int(row['step']):float(row[column])*scale for row in selected}
            assert actual.keys() == expected.keys(), (name,key,'steps')
            assert all(math.isclose(actual[step],value,rel_tol=1e-6,abs_tol=1e-9) for step,value in expected.items()), (name,key,'values')
            count += len(expected)
    report=dict(status='verified',runs=9,points=len(rows),scalars=count,port=port,cloud_upload=False)
    (HERE/'validation/swanlab_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',type=int)
    args=parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        sync()
