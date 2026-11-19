"""Record the four A3 trajectories in the existing local SwanLab runtime and check every scalar."""
import csv,json,math,sqlite3,sys,urllib.parse,urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent
LOG=HERE/"swanlab/logs"
RES=HERE/"results"
def rows(name):
    with (RES/name).open() as f: return list(csv.DictReader(f))
def main():
    points={}
    for r in rows("state_summary.csv"):
        run=r["run_id"];step=int(r["step"]);tag=r["stage"]
        v=points.setdefault(run,{}).setdefault(step,{})
        for k in ("transport_adj","transport_all","mixed_adj_adj","mixed_all_a_all_b","combined_read_pct"):
            v[tag+"/"+k]=float(r[k])
    for r in rows("full_forward.csv"):
        if r["condition"] in ("mask_00","mask_15","no7_joint","no_query"):
            points[r["run_id"]][int(r["step"])][r["stage"]+"/"+r["condition"]+"_test_pct"]=float(r["direct32_test_pct"])
    if "--verify" in sys.argv:
        db=sqlite3.connect(LOG/"runs.swanlab"); exp=db.execute("select id,name from experiment").fetchall()
        assert len(exp)==4 and {r for _,r in exp}==set(points)
        total=0
        for eid,name in exp:
            pp=points[name]
            for k in next(iter(pp.values())):
                url=f"http://127.0.0.1:16105/api/v1/experiment/{eid}/tag/"+urllib.parse.quote(k,safe="")
                actual={int(x["index"]):float(x["data"]) for x in json.load(urllib.request.urlopen(url))["data"]["list"]}
                assert actual.keys()==pp.keys() and all(math.isclose(actual[s],pp[s][k],rel_tol=1e-6,abs_tol=1e-9) for s in pp)
                total+=len(actual)
        (RES/"swanlab_validation.json").write_text(json.dumps(dict(runs=4,scalars=total,port=16105,mode="local"),indent=2))
        print("verified",total)
        return
    import swanlab
    if (LOG/"runs.swanlab").exists(): raise RuntimeError("Log already exists; use --verify.")
    for name,pp in points.items():
        swanlab.init(project="TGLT-Relay-Learning",experiment_name=name,mode="local",log_dir=str(LOG),
          config=dict(run_id=name,train_n=870,test_n=2611,donor_offsets=58,mask_count=16,observer="site7"),
          settings=swanlab.Settings(probe=dict(hardware=False,runtime=False,requirements=False,git=False,monitor=False)))
        for step,v in sorted(pp.items()): swanlab.log(v,step=step)
        swanlab.finish()
if __name__=="__main__": main()
