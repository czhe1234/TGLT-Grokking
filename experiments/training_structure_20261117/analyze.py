"""Exact finite-step attribution of rule-aligned query representations."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TGLT_SOURCE_ROOT', '/home/chenzhe/work/tglt_experiments')
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BRIDGE_PATH = REPO/'experiments/transformer_window_bridge_20261115/analyze.py'
spec = importlib.util.spec_from_file_location('structure_bridge', BRIDGE_PATH)
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)
ref = bridge.ref
torch.set_num_threads(1)
FACTORS = ['gradient', 'momentum', 'decay']
METRICS = ['rule_energy', 'within_energy', 'rule_fraction', 'test_margin', 'test_ce', 'test_acc']


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def prepare():
    table = REPO/'experiments/transformer_window_bridge_20261115/results/per_run_summary.csv'
    labels = {r['run_id']: r for r in csv.DictReader(table.open()) if r['kind'] == 'TGLT'}
    roster = []
    for row in bridge.baseline_roster():
        if row['window'] == 7:
            continue
        label = labels[row['run_id']]
        available = sorted(int(p.stem.split('_')[-1]) for p in Path(row['path']).glob('checkpoint_*.pt'))
        requested = set(range(0, 100001, 10000))
        anchors = {}
        for key in ['t_mem', 't10', 't50', 't90']:
            if label[key]:
                target = int(label[key])
                step = min(available, key=lambda s: (abs(s-target), s))
                requested.add(step)
                anchors[key] = {'logged_step': target, 'checkpoint_step': step}
        assert requested.issubset(available)
        roster.append(dict(row, steps=sorted(requested), anchors=anchors,
            strict_grokking=label['audit_fixed_grokking'] == 'True',
            reached90=label['reached90'] == 'True', final_test_acc=float(label['final_test_acc'])))
    assert len(roster) == 9
    write_json(roster, HERE/'roster.json')
    write_json(dict(analysis='training_structure_20261117', source_root=str(bridge.PROJECT),
        source_sha256=bridge.sha(REPO/'src/exp2_sequence_local_grokking.py'),
        bridge_sha256=bridge.sha(BRIDGE_PATH), classification_sha256=bridge.sha(table),
        runs=9, points=sum(len(r['steps']) for r in roster), input_count=3481,
        train_count=870, test_count=2611, precision='CPU float64', new_training_runs=0,
        sampling='0..100k every10k plus nearest saved t_mem/t10/t50/t90 from frozen classification',
        ensemble='uniform full input grid; includes held-out inputs and known task labels',
        representation='second-block query state after its LayerNorm, before final_norm/classifier',
        projection='class mean under y=(a+b)%59, after global feature centering',
        rule_energy='mean squared norm of projected feature',
        within_energy='mean squared norm of feature minus its class mean',
        rule_fraction='rule_energy/(rule_energy+within_energy)',
        finite_step='original next AdamW update reconstructed from saved optimizer state and clipped full-training gradient',
        attribution='three additive parameter displacements with shared actual next second-moment denominator; all8 subsets and Shapley average',
        caveat='This freezes update vectors for attribution, not a rerun of a modified optimizer. Saved-point drifts are not cumulative changes between checkpoints.'), HERE/'config.json')
    print(json.dumps({'runs': 9, 'points': sum(len(r['steps']) for r in roster)}, ensure_ascii=False))
    return roster


def centered_projection(features, labels):
    centered = features-features.mean(0)
    means = torch.zeros(59, features.shape[1], dtype=features.dtype)
    means.index_add_(0, labels, centered)
    means /= 59
    projected = means[labels]
    return centered, projected, centered-projected


@torch.no_grad()
def measure(model, data):
    features, logits, writes = [], [], []
    for start in range(0, 3481, 256):
        tokens = data['tokens'][start:start+256]
        roles = data['roles'][start:start+256]
        hidden = bridge.first_layer(model, tokens, roles, 'TGLT')
        z, parts = bridge.native_parts(model, hidden, 'TGLT')
        features.append(parts['final_input'])
        writes.append(parts['w'].sum(1))
        logits.append(z)
    features, logits, writes = map(torch.cat, [features, logits, writes])
    y, train, test = data['labels'], data['train'], data['test']
    centered, projected, residual = centered_projection(features, y)
    between = float(projected.square().sum()/3481)
    within = float(residual.square().sum()/3481)
    total = float(centered.square().sum()/3481)
    assert abs(between+within-total) < 1e-10*(1+total)
    assert total > 1e-20
    correct = logits.argmax(-1) == y
    true_score = logits[test].gather(1, y[test,None]).squeeze(1)
    wrong = logits[test].masked_fill(F.one_hot(y[test], 59).bool(), -torch.inf).logsumexp(1)
    wc, wp, _ = centered_projection(writes, y)
    write_total = float(wc.square().sum()/3481)
    result = dict(rule_energy=between, within_energy=within, total_variance=total,
        rule_fraction=between/total, test_margin=float((true_score-wrong).mean()),
        test_ce=float(F.cross_entropy(logits[test], y[test])),
        train_ce=float(F.cross_entropy(logits[train], y[train])),
        train_correct=int(correct[train].sum()), test_correct=int(correct[test].sum()),
        train_acc=float(correct[train].double().mean()), test_acc=float(correct[test].double().mean()),
        write_rule_fraction=float(wp.square().sum()/3481)/write_total,
        write_total_variance=write_total)
    return result, features


def update_vectors(model, checkpoint, data):
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients = [torch.zeros_like(p) for p in parameters]
    train = data['train']
    for start in range(0, len(train), 256):
        ix = train[start:start+256]
        hidden = bridge.first_layer(model, data['tokens'][ix], data['roles'][ix], 'TGLT')
        logits, _ = bridge.native_parts(model, hidden, 'TGLT')
        loss = F.cross_entropy(logits, data['labels'][ix], reduction='sum')/len(train)
        partial = torch.autograd.grad(loss, parameters)
        for target, value in zip(gradients, partial):
            target.add_(value)
    cfg = ref.RunConfig(**checkpoint['config'])
    norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
    clip = min(1., cfg.grad_clip/(float(norm)+1e-6))
    gradients = [g*clip for g in gradients]
    optimizer = ref.make_optimizer(model, cfg)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    learning_rate = ref.learning_rate(int(checkpoint['step'])+1, cfg)
    by_id = {id(p): g for p,g in zip(parameters, gradients)}
    vectors = {factor: {} for factor in FACTORS}
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
        b1, b2 = group['betas']
        for parameter in group['params']:
            state = optimizer.state[parameter]
            gradient = by_id[id(parameter)]
            previous_m = state.get('exp_avg', torch.zeros_like(parameter))
            previous_v = state.get('exp_avg_sq', torch.zeros_like(parameter))
            step = float(state.get('step', 0))+1
            assert step == int(checkpoint['step'])+1
            next_v = b2*previous_v+(1-b2)*gradient.square()
            denominator = (next_v/(1-b2**step)).sqrt()+group['eps']
            scale = -learning_rate/(1-b1**step)/denominator
            vectors['gradient'][id(parameter)] = scale*(1-b1)*gradient
            vectors['momentum'][id(parameter)] = scale*b1*previous_m
            vectors['decay'][id(parameter)] = -learning_rate*group['weight_decay']*parameter.detach()
    directions = [[vectors[factor][id(p)] for p in parameters] for factor in FACTORS]
    return parameters, gradients, directions, optimizer, dict(gradient_norm=float(norm), clip_factor=clip, lr=learning_rate)


def calculate(row, step, validate=False):
    started = time.time()
    model, checkpoint, path = bridge.load_model(row, step)
    data = bridge.make_data()
    parameters, gradients, directions, optimizer, update_info = update_vectors(model, checkpoint, data)
    original = [p.detach().clone() for p in parameters]
    values = {}
    base_features = None
    final_features = None
    for subset in range(8):
        with torch.no_grad():
            for index, (parameter, initial) in enumerate(zip(parameters, original)):
                parameter.copy_(initial)
                for factor in range(3):
                    if subset & (1 << factor):
                        parameter.add_(directions[factor][index])
        metrics, features = measure(model, data)
        values[subset] = metrics
        if subset == 0:
            base_features = features
        elif subset == 7:
            final_features = features
    with torch.no_grad():
        predicted = [p.clone() for p in parameters]
        for parameter, initial, gradient in zip(parameters, original, gradients):
            parameter.copy_(initial)
            parameter.grad = gradient.clone()
    optimizer.step()
    parameter_error = max(float((p-expected).abs().max()) for p,expected in zip(parameters,predicted))
    assert parameter_error < 1e-11
    with torch.no_grad():
        for parameter, initial in zip(parameters, original):
            parameter.copy_(initial)
    attributions = {metric: {} for metric in METRICS}
    for metric in METRICS:
        for factor, name in enumerate(FACTORS):
            value = 0.
            for subset in range(8):
                if subset & (1 << factor):
                    continue
                size = bin(subset).count('1')
                weight = math.factorial(size)*math.factorial(2-size)/6
                value += weight*(values[subset | (1 << factor)][metric]-values[subset][metric])
            attributions[metric][name] = value
        error = sum(attributions[metric].values())-(values[7][metric]-values[0][metric])
        assert abs(error) < 1e-10*(1+abs(values[0][metric]))
    y = data['labels']
    _, projection, residual = centered_projection(base_features, y)
    _, delta_projection, delta_residual = centered_projection(final_features-base_features, y)
    balance = dict(rule_linear=float(2*(projection*delta_projection).sum()/3481),
        rule_quadratic=float(delta_projection.square().sum()/3481),
        within_linear=float(2*(residual*delta_residual).sum()/3481),
        within_quadratic=float(delta_residual.square().sum()/3481))
    delta_b = values[7]['rule_energy']-values[0]['rule_energy']
    delta_w = values[7]['within_energy']-values[0]['within_energy']
    denominator = values[0]['total_variance']*values[7]['total_variance']
    balance['fraction_from_rule_change'] = values[0]['within_energy']*delta_b/denominator
    balance['fraction_from_within_change'] = -values[0]['rule_energy']*delta_w/denominator
    balance_errors = [abs(delta_b-balance['rule_linear']-balance['rule_quadratic']),
        abs(delta_w-balance['within_linear']-balance['within_quadratic']),
        abs(values[7]['rule_fraction']-values[0]['rule_fraction']-balance['fraction_from_rule_change']-balance['fraction_from_within_change'])]
    assert max(balance_errors) < 1e-9
    log = next(record for record in checkpoint['history'] if int(record['step']) == step)
    logged_correct = int(log['test_correct'])
    replay_difference = values[0]['test_correct']-logged_correct
    assert abs(replay_difference) <= 3
    native_check = None
    if validate:
        ix = torch.tensor([0, 1, 59, 1771, 3480])
        hidden = bridge.first_layer(model, data['tokens'][ix], data['roles'][ix], 'TGLT')
        fast, _ = bridge.native_parts(model, hidden, 'TGLT')
        optimized_gradients = torch.autograd.grad(fast.square().mean(), parameters)
        native = ref.OpenLocalExponentialSweep(4, model.r, model.T, model.sweep.lambdas.tolist(), False).double()
        native.load_state_dict(model.sweep.state_dict())
        model.sweep = native
        for block in model.blocks:
            block.sweep = native
        slow = model(data['tokens'][ix], data['roles'][ix])
        native_gradients = torch.autograd.grad(slow.square().mean(), parameters)
        forward_error = float((slow-fast).abs().max())
        gradient_error = max(float((a-b).abs().max()) for a,b in zip(optimized_gradients,native_gradients))
        assert forward_error < 1e-9 and gradient_error < 1e-8
        native_check = dict(forward_error=forward_error, gradient_error=gradient_error)
    result = dict(run_id=row['run_id'], window=row['window'], seed=row['seed'], step=step,
        strict_grokking=row['strict_grokking'], reached90=row['reached90'], base=values[0], after=values[7],
        coalitions=values, attribution=attributions, structure_balance=balance, update=update_info,
        checks=dict(parameter_error=parameter_error, balance_max_error=max(balance_errors),
            logged_test_correct=logged_correct, replay_correct_difference=replay_difference, native=native_check),
        source=dict(path=str(path), sha256=bridge.sha(path)), seconds=time.time()-started)
    destination = HERE/('validation/smoke' if validate else 'records')/row['run_id']/('%07d.json'%step)
    write_json(result, destination)
    return result


def aggregate(roster):
    records = [json.loads(path.read_text()) for path in sorted((HERE/'records').glob('*/*.json'))]
    expected = {(row['run_id'], step) for row in roster for step in row['steps']}
    assert {(r['run_id'], r['step']) for r in records} == expected
    rows, coalition_rows, sources = [], [], []
    for record in records:
        base = record['base']
        row = {key: record[key] for key in ['run_id', 'window', 'seed', 'step', 'strict_grokking', 'reached90']}
        row.update(base)
        for metric in METRICS:
            row['delta_'+metric] = record['after'][metric]-base[metric]
            for factor in FACTORS:
                row[factor+'_delta_'+metric] = record['attribution'][metric][factor]
        row.update(record['structure_balance'])
        row.update(record['update'])
        rows.append(row)
        for subset, values in record['coalitions'].items():
            coalition_rows.append(dict(run_id=record['run_id'], step=record['step'], subset=int(subset), **values))
        sources.append(dict(run_id=record['run_id'], step=record['step'], **record['source']))
    bridge.write_csv(rows, HERE/'results/curves.csv')
    bridge.write_csv(coalition_rows, HERE/'results/coalitions.csv')
    bridge.write_csv(sources, HERE/'results/sources.csv')
    report = dict(runs=len(roster), points=len(records), coalitions=len(coalition_rows),
        max_parameter_error=max(r['checks']['parameter_error'] for r in records),
        max_balance_error=max(r['checks']['balance_max_error'] for r in records),
        replay_correct_difference_points=sum(r['checks']['replay_correct_difference'] != 0 for r in records),
        max_replay_correct_difference=max(abs(r['checks']['replay_correct_difference']) for r in records),
        CPU_only=True, new_training_runs=0)
    write_json(report, HERE/'validation/completion_checks.json')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    roster = prepare() if args.prepare or not (HERE/'roster.json').exists() else json.loads((HERE/'roster.json').read_text())
    if args.prepare:
        return
    if args.smoke:
        results = [calculate(next(r for r in roster if r['window'] == window and r['seed'] == 1), step, True)
                   for window,step in [(8, 0), (8, 50000), (12, 100000), (15, 50000)]]
        write_json(dict(cases=4, seconds=sum(r['seconds'] for r in results),
            parameter_error=max(r['checks']['parameter_error'] for r in results),
            native_forward_error=max(r['checks']['native']['forward_error'] for r in results),
            native_gradient_error=max(r['checks']['native']['gradient_error'] for r in results)), HERE/'validation/smoke_checks.json')
        print(json.dumps(json.loads((HERE/'validation/smoke_checks.json').read_text())), flush=True)
        return
    started = time.time()
    tasks = [(row, step) for row in roster for step in row['steps']
             if not (HERE/'records'/row['run_id']/('%07d.json'%step)).exists()]
    write_json(dict(state='running', pid=os.getpid(), remaining=len(tasks), completed=0), HERE/'status.json')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(calculate, row, step) for row,step in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            state = dict(state='running', pid=os.getpid(), completed=completed, total=len(tasks),
                latest=[result['run_id'],result['step']], seconds=time.time()-started)
            write_json(state, HERE/'status.json')
            print(json.dumps(state), flush=True)
    report = aggregate(roster)
    write_json(dict(state='complete', seconds=time.time()-started, **report), HERE/'status.json')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
