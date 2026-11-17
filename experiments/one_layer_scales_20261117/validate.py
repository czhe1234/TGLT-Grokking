import copy
import json
import torch
from common import CFG, ROOT, make_model, prepare_data, make_data, render, paramhash, sha, dump

def validate():
    d = prepare_data()
    regenerated = make_data()
    checks = {}
    assert len(d['train_pairs']) == 870 and len(d['test_pairs']) == 2611
    assert not set(d['train_pairs'].tolist()) & set(d['test_pairs'].tolist())
    assert sha(ROOT.parents[1] / 'src/exp2_sequence_local_grokking.py') == CFG['source_sha256']
    for name in ['train', 'test_familiar', 'seen_new', 'test_new']:
        ds = d[name]
        assert all(torch.equal(ds[k], regenerated[name][k]) for k in ds)
        x, y = ds['x'], ds['y']
        assert torch.equal(y, (x[:, 0] + x[:, 1]) % 59)
        assert torch.equal(ds['pair_id'], x[:, 0] * 59 + x[:, 1])
        assert (x[:, 3] > x[:, 4]).all() and (x[:, 3] <= 8).all()
        assert (x[:, 5] >= 8).all() and (x[:, 5] <= 23).all()
        tokens, roles = render(x)
        rows = torch.arange(len(x))
        assert torch.equal(tokens[rows, x[:, 5]] - 1, x[:, 1])
        assert torch.equal(tokens[rows, x[:, 5]-x[:, 3]]-1, x[:, 0])
        assert torch.equal(tokens[rows, x[:, 5]-x[:, 4]]-1, x[:, 2])
        assert (roles == 1).sum(1).eq(2).all() and (tokens != 0).sum(1).eq(3).all()
    train_geom = set(map(tuple, d['train']['x'][:, [4, 3]].tolist()))
    new_geom = set(map(tuple, d['test_new']['x'][:, [4, 3]].tolist()))
    assert not train_geom & new_geom
    counts = {}
    for seed in CFG['seeds']:
        models = [make_model(g, seed) for g in CFG['groups']]
        assert paramhash(models[0]) == paramhash(models[1])
        for group, model in zip(CFG['groups'], models):
            counts[group] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert len(model.blocks) == 1 and model.position.eq(0).all()
            assert not model.position.requires_grad and not model.sweep.raw_lambda.requires_grad
            model.double()
            local = copy.deepcopy(model)
            x = torch.cat([d['train']['x'][:12], d['test_new']['x'][:12]])
            x[:4, :3] = torch.tensor([[0,0,0],[58,58,58],[0,58,29],[58,0,29]])
            x[0, 5], x[1, 5] = 8, 23
            z, expected = model(x), local.forward_local(x)
            err = (z-expected).abs().max().item()
            assert err < 1e-10, err
            target = (x[:,0]+x[:,1]) % 59
            torch.nn.functional.cross_entropy(z, target).backward()
            torch.nn.functional.cross_entropy(expected, target).backward()
            grads = []
            for (n,p), (k,q) in zip(model.named_parameters(), local.named_parameters()):
                assert n == k
                if p.requires_grad:
                    assert p.grad is not None and q.grad is not None, n
                    grads.append((p.grad-q.grad).abs().max().item())
            ge = max(grads)
            assert ge < 1e-9, ge
            moved = x.clone()
            moved[:,5] = 8 + (moved[:,5] + 1) % 16
            assert (model.forward_local(moved)-expected).abs().max().item() < 1e-10
            assert (2*model.sweep.kernel.square().sum(1)-1).abs().max().item() < 1e-12
            checks[group+'_s'+str(seed)] = {'forward_max_error':err,'gradient_max_error':ge}
    assert len(set(counts.values())) == 1
    checks.update(passed=True,parameter_counts=counts,data_sha256=sha(ROOT/'data.pt'),
                  paired_initialization_identical=True,label_and_role_checks=True,
                  geometry_and_pair_split_disjoint=True,translation_invariance=True)
    dump(checks, ROOT/'validation/cpu_checks.json')
    print(json.dumps(checks,ensure_ascii=False))

if __name__ == '__main__':
    validate()
