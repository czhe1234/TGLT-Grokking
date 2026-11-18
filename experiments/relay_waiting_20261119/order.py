"""Cyclic variation of numeral-vector similarities and a matched rotation of an AdamW step."""
import math
import torch

P=59


class Cycle:
    def __init__(self, permutation):
        self.order=permutation
        self.inverse=permutation.argsort()
        ids=torch.arange(P,device=permutation.device)
        self.i=ids[None].expand(P-1,-1)
        self.j=(self.i+ids[1:,None])%P

    def measure(self, embedding, gradient=False):
        x=embedding.reshape(P,-1)[self.order].double()
        x=x-x.mean(0)
        gram=x@x.T
        bands=gram[self.i,self.j]
        off=bands-bands.mean()
        means=off.mean(1)
        total=off.square().sum().clamp_min(1e-30)
        value=P*means.square().sum()/total
        if not gradient:
            return value
        residual=torch.zeros_like(gram)
        residual[self.i,self.j]=means[:,None]-value*off
        deriv=(4/total)*(residual@x)
        return value,(deriv-deriv.mean(0))[self.inverse]


def cycles(device,seed):
    g=torch.Generator().manual_seed(seed)
    return Cycle(torch.arange(P,device=device)),Cycle(torch.randperm(P,generator=g).to(device))


def unit(x):
    return x/x.norm().clamp_min(1e-30)


def drop_component(x,direction):
    return x-(x*direction).sum()*direction


@torch.no_grad()
def rotate(before, proposed, real, shuffled, condition, angle):
    """Return FP32 weights. Tests and diagnostics compare to the actual rounded copy."""
    raw=proposed.double()-before
    mean=raw.mean(0)
    centered=raw-mean
    radius=unit(before-before.mean(0))
    tangent=drop_component(centered,radius)
    tangent_dir=unit(tangent)
    _,grad=real.measure(before,True)
    true_dir=unit(drop_component(drop_component(grad,radius),tangent_dir))
    if condition=="shuffled":
        _,g=shuffled.measure(before,True)
        direction=unit(drop_component(drop_component(drop_component(g,radius),tangent_dir),true_dir))
    else:
        direction=true_dir if condition=="increase" else -true_dir
    rot=math.cos(angle)*tangent+math.sin(angle)*tangent.norm()*direction
    changed=(proposed.double()+rot-tangent).to(proposed.dtype)
    actual=changed.double()-before
    delta=actual-raw
    rawnorm=raw.norm().clamp_min(1e-30)
    checks=torch.stack(((actual.norm()-raw.norm()).abs()/rawnorm,
                        (delta*radius).sum().abs()/rawnorm,
                        delta.mean(0).norm()*math.sqrt(P)/rawnorm,
                        ((grad*delta).sum()), (grad*raw).sum()))
    return changed,checks
