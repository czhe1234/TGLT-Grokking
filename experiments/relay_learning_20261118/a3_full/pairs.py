"""A3 observables for every nonidentical first-operand replacement at fixed b."""
import torch
import torch.nn.functional as F

P = 59


def operand_readout(x, train, test, centered=True):
    features = x.reshape(P * P, -1)
    a = torch.arange(P * P, device=x.device) // P
    if centered:
        features = features - features[train[0]]
    one = F.one_hot(a[train], P).to(features.dtype)
    centers = (one.T @ features[train]) / one.sum(0)[:, None]
    if centered:
        logits = -(features[test,None,:] - centers[None,:,:]).square().sum(2)
    else:
        logits = 2 * features[test] @ centers.T - centers.square().sum(1)
    return int((logits.argmax(1) == a[test]).sum())


def mode_mean(z):
    return z.square().sum(-1).mean((0, 1))


def measurements(model, field, train, test):
    src = field["source"].reshape(P, P, 4, 8)
    rx = field["right7"].reshape_as(src)
    write = field["writes"].reshape(P, P, 4, -1)
    units = model.sweep.lambdas.pow(7)[None,None,:,None]
    rows, per_shift = [], []
    minimum, collisions, max_error = [], [], []
    correction = 2 * P / (P-1)
    for offset in range(1, P):
        # rolling only axis 0 pairs (a,b) with (a+offset mod 59,b) for all a,b.
        ds = src.roll(-offset, 0) - src
        dr = rx.roll(-offset, 0) - rx
        dw = write.roll(-offset, 0) - write
        squared = dr.square().sum(-1)
        minimum.append(squared.amin((0,1)))
        collisions.append((squared == 0).sum((0,1)))
        max_error.append((dr - units*ds).abs().amax((0,1,3)))
        # Pairwise variance identity averages over all 58 other b values after each
        # fixed-b contrast has been formed. The b+1 value isolates the change of a sampling.
        per_shift.append(torch.stack([
            mode_mean(ds), mode_mean(dr), mode_mean(dw),
            mode_mean(dw - dw.roll(-1, 1)),
            correction * mode_mean(dw - dw.mean(1, keepdim=True)),
        ], dim=1))
    values = torch.stack(per_shift)
    avg = values.mean(0)
    for grid, observed in ((src, avg[:,0]), (rx, avg[:,1])):
        expected = correction * mode_mean(grid - grid.mean(0, keepdim=True))
        torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-18)
    joint = write - write.mean(0, keepdim=True) - write.mean(1, keepdim=True) + write.mean((0,1), keepdim=True)
    torch.testing.assert_close(avg[:,4], correction**2 * mode_mean(joint), rtol=1e-9, atol=1e-18)
    minimum = torch.stack(minimum).amin(0).tolist()
    collisions = torch.stack(collisions).sum(0).tolist()
    max_error = torch.stack(max_error).amax(0).tolist()
    assert max(max_error) < 1e-10
    plain = values.tolist()
    read_all = operand_readout(rx, train, test)
    for j in range(4):
        old = plain[0][j]; full = avg[j].tolist()
        count = operand_readout(rx[:,:,j], train, test)
        rows.append(dict(mode=j+1, source_adj=old[0], source_all=full[0],
                         transport_adj=old[1], transport_all=full[1], all_over_adj=full[1]/old[1],
                         mixed_adj_adj=old[3], mixed_all_a_adj_b=full[3], mixed_all_a_all_b=full[4],
                         write_all=full[2], minimum_transport_change_sq=minimum[j],
                         indistinguishable_pairs=collisions[j], identity_max_abs=max_error[j],
                         uncentered_gpu_correct=operand_readout(rx[:,:,j],train,test,centered=False),
                         read_correct=count, read_pct=count*100/len(test),
                         combined_read_correct=read_all, pairs=P*P*(P-1)))
    by_offset=[]
    for k,modes in enumerate(plain,1):
        for j,entry in enumerate(modes,1):
            by_offset.append(dict(offset=k, mode=j, fixed_b_pairs=P*P,
                                  source=entry[0], transport=entry[1], write_change=entry[2],
                                  mixed_next_b=entry[3], mixed_all_b=entry[4]))
    return rows, by_offset
