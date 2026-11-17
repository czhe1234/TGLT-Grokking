import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'src'))
import exp2_sequence_local_grokking as ref
CFG = json.loads((ROOT / 'config.json').read_text())
torch.set_num_threads(1)

def dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def paramhash(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def write_history(rows, path):
    path = Path(path)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
    dump(rows, path.with_suffix('.json'))

class NormalizedSweep(ref.OpenLocalExponentialSweep):
    def __init__(self, lambdas):
        super().__init__(4, 2, 4, lambdas, False)
        self.register_buffer('kernel', torch.empty(4, 8))
        self.register_buffer('denominator', torch.empty(4))
        self.rebuild()

    def rebuild(self):
        lam = self.lambdas.detach()
        powers = torch.arange(1, 9, device=lam.device, dtype=lam.dtype)
        raw = lam[:, None].pow(powers[None])
        self.denominator = (2 * raw.square().sum(1)).sqrt()
        self.kernel = raw / self.denominator[:, None]

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.rebuild()
        return self

    def forward(self, source):
        right, left = super().forward(source)
        den = self.denominator.view(1, 1, 4, 1)
        return right / den, left / den

class OneLayerModel(ref.SequenceLocalModeModel):
    def __init__(self, group):
        lambdas = [math.exp(-1.0 / xi) for xi in CFG['groups'][group]]
        super().__init__(p=59, seq_len=32, query_pos=15, d=64, d_a=8,
                         M=4, r=2, T=4, layers=1, lambdas=lambdas,
                         learnable_lambda=False, local_coupling=1.0,
                         learned_position=False)
        self.position.zero_()
        self.sweep = NormalizedSweep(lambdas)
        self.blocks[0].sweep = self.sweep
        self.register_buffer('role_ids', torch.tensor([1, 1, 3]))
        self.register_buffer('gap_id', torch.tensor(0))
        self.register_buffer('input_order', torch.tensor([0, 2, 1]))

    def query_state(self, x, mode_mask=None):
        block = self.blocks[0]
        h = self.residue(x.index_select(1, self.input_order) + 1) + self.role(self.role_ids)[None]
        source = block.drive(h[:, :2]).reshape(-1, 2, 4, 8)
        gap = block.drive(self.residue(self.gap_id) + self.role(self.gap_id)).reshape(4, 8)
        far = self.sweep.kernel[:, x[:, 3] - 1].T[..., None]
        near = self.sweep.kernel[:, x[:, 4] - 1].T[..., None]
        mass = self.sweep.kernel.sum(1)[None, :, None]
        right = far * source[:, 0] + near * source[:, 1] + (mass - far - near) * gap
        left = (mass * gap).expand_as(right)
        state = torch.cat([right, left], -1)
        visible = torch.einsum('bd,mdo->bmo', h[:, 2], block.visible_factor)
        propagated = torch.einsum('bma,mao->bmo', state, block.mode_factor)
        gates = torch.softmax(block.mode_logits, 0)
        contribution = block.local_coupling * visible * propagated * gates[None, :, None]
        if mode_mask is not None:
            contribution = contribution * mode_mask[None, :, None]
        writes = F.linear(contribution, block.out.weight)
        updated = block.norm(h[:, 2] + writes.sum(1) + block.out.bias)
        return updated, writes, gates

    def forward(self, x, mode_mask=None):
        hidden, _, _ = self.query_state(x, mode_mask)
        return self.classifier(self.final_norm(hidden))

    def forward_local(self, x):
        tokens, roles = render(x)
        h = self.residue(tokens) + self.role(roles) + self.position[None]
        h = self.blocks[0](h)
        q = h[torch.arange(len(x), device=x.device), x[:, 5]]
        return self.classifier(self.final_norm(q))

def make_model(group, seed):
    ref.set_seed(seed)
    return OneLayerModel(group)

def render(x):
    tokens = torch.zeros(len(x), 32, dtype=torch.long, device=x.device)
    roles = torch.zeros_like(tokens)
    rows = torch.arange(len(x), device=x.device)
    a, b, c, df, dn, q = x.unbind(1)
    tokens[rows, q - df] = a + 1
    tokens[rows, q - dn] = c + 1
    tokens[rows, q] = b + 1
    roles[rows, q - df] = 1
    roles[rows, q - dn] = 1
    roles[rows, q] = 3
    return tokens, roles

def examples(pairs, geometry, queries, seed):
    gen = torch.Generator().manual_seed(seed)
    n = len(pairs) * len(geometry)
    ids = pairs.repeat_interleave(len(geometry))
    g = torch.tensor(geometry).repeat(len(pairs), 1)
    c = torch.randint(59, (n,), generator=gen)
    anchors = torch.tensor(queries)
    q = anchors[torch.randint(len(anchors), (n,), generator=gen)]
    a, b = ids // 59, ids % 59
    x = torch.stack([a, b, c, g[:, 1], g[:, 0], q], 1)
    return {'x': x, 'y': (a + b) % 59, 'pair_id': ids}

def make_data():
    train, test = ref.make_split(3481, .25, CFG['split_seed'])
    tg, hg = CFG['train_geometry'], CFG['heldout_geometry']
    tq, hq = CFG['train_queries'], CFG['heldout_queries']
    return {'train': examples(train, tg, tq, 1729),
            'test_familiar': examples(test, tg, tq, 1730),
            'seen_new': examples(train, hg, hq, 1731),
            'test_new': examples(test, hg, hq, 1732),
            'train_pairs': train, 'test_pairs': test}

def prepare_data():
    path = ROOT / 'data.pt'
    if not path.exists():
        ref.atomic_torch_save(make_data(), path)
    data = torch.load(path, map_location='cpu', weights_only=False)
    dump({'sha256': sha(path), 'sets': {k: len(v['y']) for k, v in data.items() if isinstance(v, dict)},
          'train_pairs': len(data['train_pairs']), 'test_pairs': len(data['test_pairs'])}, ROOT / 'data_manifest.json')
    return data

def optimizer(model, cuda_graph=False):
    norms = {id(p) for mod in model.modules() if isinstance(mod, nn.LayerNorm)
             for p in mod.parameters(recurse=False)}
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if id(p) in norms or name.endswith('.bias') else decay).append(p)
    lr = torch.tensor(CFG['lr'], device='cuda') if cuda_graph else CFG['lr']
    return torch.optim.AdamW([{'params': decay, 'weight_decay': CFG['wd']},
                              {'params': no_decay, 'weight_decay': 0.0}],
                             lr=lr, betas=tuple(CFG['betas']), eps=CFG['eps'],
                             fused=cuda_graph, capturable=cuda_graph)
