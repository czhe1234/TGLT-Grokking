"""Stream only this experiment's source histories into an isolated local SwanLab board."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="",SWANLAB_MODE="local",OMP_NUM_THREADS="1")
import sys,time,json,csv,fcntl
from pathlib import Path
B=Path(__file__).resolve().parent
def main(dest):
 import swanlab
 lock=(dest/"swanlab.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rec=dest/"swanlab.json"
 if rec.exists():
  old=json.loads(rec.read_text())
  if old.get("finished"):return
  raise RuntimeError("Interrupted logger: inspect its receipt before recovery")
 while not (dest/"config.json").exists():time.sleep(1)
 cfg=json.loads((dest/"config.json").read_text());w=cfg["window"]
 label=("带FFN_wd1" if cfg["group"]=="T1" else "纯Attention_wd0.1")+"_窗口%d"%w
 cfg.update(model=("带FFN Transformer" if cfg["group"]=="T1" else "无FFN Attention"),display_group=label,accuracy_unit="百分比",
            source_history=str(dest/"history.csv"))
 colors={7:"#6A3D9A",8:"#0072B2",12:"#D55E00",15:"#009E73",None:"#D62728"}
 from swanlab.deprecated.local import _ColumnInfo
 original=_ColumnInfo.__init__
 def grouped(self,key,kid,chart_type):
  original(self,key,kid,chart_type);family=key.split("/",1)[0]
  self.section_name=family;self.section_sort={"准确率":0,"损失":1,"分类间隔":2,"参数":3}.get(family,4)
 _ColumnInfo.__init__=grouped
 logdir=B/"swanlab/logs"
 with (logdir/".init.lock").open("a") as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  swanlab.init(project="TGLT-Transformer-Followup",experiment_name=label+"_seed%d"%cfg["seed"],
   group=cfg["exp_group"],color=({8:"#0072B2",12:"#D62728",15:"#00845A"} if cfg["group"]=="T1" else {8:"#7B3294",12:"#B35806",15:"#192F3B"})[w],config=cfg,mode="local",log_dir=str(logdir),
   description="原始模59加法；输入0、7，查询15。T1: FFN256/wd1/150k；T2: 无FFN/wd0.1/100k。",
   settings=swanlab.Settings(probe=dict(hardware=False,runtime=False,requirements=False,git=False,monitor=False)))
 mapping={"准确率/训练集（%）":("train_acc",100),"准确率/测试集（%）":("test_acc",100),
          "损失/训练交叉熵":("train_ce",1),"损失/测试交叉熵":("test_ce",1),
          "分类间隔/测试集平均":("test_margin_mean",1),"参数/参数范数":("weight_norm",1)}
 last=-1;finished=False
 def receipt(state):
  tmp=rec.with_suffix(".json.tmp");tmp.write_text(json.dumps(dict(pid=os.getpid(),last_step=last,finished=finished,
   state=state,version=swanlab.__version__,log_dir=str(logdir),metric_map=mapping),ensure_ascii=False,indent=2));tmp.replace(rec)
 try:
  while True:
   if (dest/"history.json").exists():
    rows=json.loads((dest/"history.json").read_text())
    for row in rows:
     if row["step"]>last:
      swanlab.log({k:float(row[src])*scale for k,(src,scale) in mapping.items()},step=int(row["step"]))
      last=int(row["step"])
   status=json.loads((dest/"status.json").read_text()) if (dest/"status.json").exists() else {}
   receipt(status.get("state","starting"))
   if status.get("state") in ["complete","failed"] and last>=status.get("step",0):finished=True;break
   time.sleep(5)
 finally:
  swanlab.finish();receipt(status.get("state","interrupted"))
if __name__=="__main__":main(Path(sys.argv[1]))
