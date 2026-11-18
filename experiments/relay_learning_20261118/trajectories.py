"""Trace stage-wise diagnostic readouts on the original saved trajectories."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import monotonic
import sys
from replay import *


@torch.no_grad()
def measure(job):
    rid, step = job
    cached = HERE/".trajectory_cache"/rid/("%07d.json"%step)
    if cached.exists():
        return read_json(cached)
    model, cp, native, path, checksum = load(rid,step)
    values = local_summary(model,capture(model))
    log = metadata(rid)[step]
    assert cp["history"][-1]["step"] == step
    for tag in ("train","test"):
        delta = values[tag+"_correct"]-int(log[tag+"_correct"])
        assert int(log[tag+"_correct"]) == cp["history"][-1][tag+"_correct"]
        values["original_"+tag+"_correct"]=int(log[tag+"_correct"])
        values["original_"+tag+"_ce"]=float(log[tag+"_ce"])
        values["replay_delta_"+tag] = delta
    if values["replay_delta_train"] or values["replay_delta_test"]:
        model.sweep=native
        for block in model.blocks:
            block.sweep=native
        model.float()
        z=torch.cat([model(DATA["tokens"][k:k+128],DATA["roles"][k:k+128]) for k in range(0,N,128)])
        values["native_fp32_test_correct"]=score(z)["correct"]
        values["native_fp32_train_correct"]=score(z,TR)["correct"]
        values["native_fp32_test_ce"]=score(z)["ce"]
    row=dict(run_id=rid,step=step,path=path,sha256=checksum,**values)
    write_json(cached,row)
    return row


def main():
    if len(sys.argv)>1 and sys.argv[1]=="smoke":
        t=monotonic()
        for rid,step in [(RUNS[0],107000),(RUNS[2],100000)]:
            print(measure((rid,step)),flush=True)
        print(monotonic()-t,flush=True)
        return
    jobs = [(rid,k) for rid in RUNS for k in
            (range(0,100001,1000) if rid.startswith("r2") else range(0,300001,500))]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(measure,j) for j in jobs]
        for done,f in enumerate(as_completed(futures),1):
            rows.append(f.result())
            if done%100==0:
                print("Completed",done,"of",len(jobs),flush=True)
    rows.sort(key=lambda x:(RUNS.index(x["run_id"]),x["step"]))
    table(OUT/"trajectories.csv",rows)
    print("Completed",len(rows),flush=True)


if __name__=="__main__":
    main()
