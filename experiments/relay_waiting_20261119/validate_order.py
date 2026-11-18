"""Reference derivative, permutation and finite step tests for the controlled rotation."""
import json
from pathlib import Path
import torch
from order import cycles,rotate,P

torch.manual_seed(159)
real,sham=cycles("cpu",20261119)
x=torch.randn(P,64,dtype=torch.float64,requires_grad=True)
errors=[]
for c in (real,sham):
    rho,analytic=c.measure(x,True)
    auto=torch.autograd.grad(rho,x)[0]
    errors.append(float((auto-analytic).abs().max()))
    assert errors[-1]<1e-12
phase=2*torch.pi*torch.arange(P,dtype=torch.float64)/P
ring=torch.stack((phase.cos(),phase.sin()),1)
assert abs(float(real.measure(ring))-1)<1e-12
x=x.detach()
d=torch.randn_like(x)*.001
stats={}
for arm in ("increase","decrease","shuffled"):
    changed,diag=rotate(x,(x+d).float(),real,sham,arm,15*torch.pi/180)
    stats[arm]=diag.tolist()
    assert diag[:3].abs().max()<1e-4
    if arm=="increase": assert diag[3]>0
    if arm=="decrease": assert diag[3]<0
assert abs(stats["shuffled"][3])<.08*stats["increase"][3]
out=dict(analytic_max_error=max(errors),normalization_controls=stats,circle_order=float(real.measure(ring)))
dest=Path(__file__).resolve().parent/"results"/"order_check.json"
dest.write_text(json.dumps(out,indent=2))
print(json.dumps(out))
