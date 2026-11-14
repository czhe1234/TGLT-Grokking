"""Figures and tables from raw histories and common mechanism outputs; no training."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import sys,json,csv,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B));import analyze as a
O=B/"results";O.mkdir(exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family":"Noto Sans CJK JP","axes.unicode_minus":False,"font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":2,"savefig.facecolor":"white"})
BLUE="#0072B2";RED="#D62728";COLORS=[BLUE,"#D55E00","#00845A","#7B3294"];BLACK="#192F3B"
def label(w):return "全局" if w is None else "W=%d"%w
def save(fig,name,top=.89,bottom=.07):
    fig.tight_layout(rect=[.02,bottom,.99,top])
    fig.savefig(O/name,dpi=180,bbox_inches="tight");plt.close(fig)
def events(ax,lab):
    if lab["t10"] is not None and lab["t90"] is not None:
        ax.axvspan(lab["t10"]/1000,lab["t90"]/1000,color="#F5B943",alpha=.2,zorder=0)
def axes(ax,ylim=None):
    ax.grid(alpha=.18);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100])
    if ylim is not None:ax.set_ylim(*ylim)
def collect_saved():
    rows=json.loads((B/"results/run_summary.json").read_text())
    hist={r["run_id"]:pd.DataFrame(json.loads((B/"histories"/(r["run_id"]+".json")).read_text())) for r in rows}
    return rows,hist

def collect():
    roster=a.baseline_roster()+a.transformer_roster();rows=[];hist={}
    for row in roster:
        path=Path(row["path"])/"history.json"
        if not path.exists():continue
        h=json.loads(path.read_text());lab=a.classify(h)
        r=dict(row,**lab);r.update(history_sha256=a.sha(path),complete=h[-1]["step"]==100000,
            parameter_count=109051 if row["kind"]=="Transformer" else 62787)
        rows.append(r);hist[row["run_id"]]=pd.DataFrame(h)
    a.write_csv(rows,O/"per_run_summary.csv")
    a.json_write(rows,O/"run_summary.json")
    grouped=[]
    for kind in ["TGLT","Transformer"]:
        for w in ([7,8,12,15] if kind=="TGLT" else [7,8,12,15,None]):
            rs=[r for r in rows if r["kind"]==kind and r["window"]==w]
            if len(rs)!=3 or not all(r["complete"] for r in rs):continue
            widths=[r["width_10_90"] for r in rs if r["width_10_90"] is not None]
            grouped.append(dict(model=kind,window="global" if w is None else w,n=3,
                reached90=sum(r["reached90"] for r in rs),strict_grokking=sum(r["audit_fixed_grokking"] for r in rs),
                reached90_fraction=sum(r["reached90"] for r in rs)/3,
                strict_fraction=sum(r["audit_fixed_grokking"] for r in rs)/3,
                width_n=len(widths),width_median=None if not widths else float(np.median(widths)),
                width_min=None if not widths else min(widths),width_max=None if not widths else max(widths),
                final_train_mean=np.mean([r["final_train_acc"] for r in rs]),
                final_test_mean=np.mean([r["final_test_acc"] for r in rs]),
                final_test_std=np.std([r["final_test_acc"] for r in rs]),
                parameter_count=rs[0]["parameter_count"]))
    a.write_csv(grouped,O/"grouped_summary.csv")
    cn=[]
    for g in grouped:
        cn.append({"模型":g["model"],"范围":("全局" if g["window"]=="global" else ("H=" if g["model"]=="TGLT" else "W=")+str(g["window"])),
            "种子数":g["n"],"达到90%条数":g["reached90"],"严格Grokking条数":g["strict_grokking"],
            "100k训练准确率均值%":100*g["final_train_mean"],"100k测试准确率均值%":100*g["final_test_mean"],
            "测试准确率标准差_百分点":100*g["final_test_std"],"完整跃迁条数":g["width_n"],
            "10到90宽度中位数_步":g["width_median"],"10到90宽度最小_步":g["width_min"],
            "10到90宽度最大_步":g["width_max"],"参数量":g["parameter_count"]})
    a.write_csv(cn,O/"配置汇总.csv")

    return rows,hist
def accuracy(rows,hist):
    windows=[7,8,12,15,None];fig,axs=plt.subplots(5,3,figsize=(17,15),sharex=True,sharey=True)
    for row in rows:
        if row["kind"]!="Transformer":continue
        ax=axs[windows.index(row["window"]),row["seed"]];h=hist[row["run_id"]]
        ax.plot(h.step/1000,h.train_acc*100,color=BLUE);ax.plot(h.step/1000,h.test_acc*100,color=RED)
        events(ax,row);axes(ax,(-2,103))
        tag="严格 Grokking" if row["audit_fixed_grokking"] else ("达到 90%" if row["reached90"] else "未达 90%")
        ax.set_title("%s · seed %d · %s"%(label(row["window"]),row["seed"],tag),fontsize=11)
        if row["seed"]==0:ax.set_ylabel("准确率（%）")
        if row["window"] is None:ax.set_xlabel("训练步数（千步）")
    fig.suptitle("标准 Transformer：局部窗口扫描的全部 15 条曲线",fontsize=21,weight="bold",y=.99,color=BLACK)
    fig.legend(handles=[Line2D([0],[0],color=BLUE,label="训练准确率"),Line2D([0],[0],color=RED,label="测试准确率")],loc="upper center",bbox_to_anchor=(.5,.957),ncol=2,frameon=False)
    fig.text(.5,.015,"原始输入位置 0、7，查询 15；每 500 步记录。浅橙色仅标出首次完整 10%→90% 区间。",ha="center",fontsize=11)
    save(fig,"01_Transformer全部训练测试曲线.png",top=.925)
def paired_means(rows,hist):
    fig,axs=plt.subplots(4,2,figsize=(15,13),sharex=True,sharey=True);agg=[]
    for i,w in enumerate([7,8,12,15]):
        for j,kind in enumerate(["TGLT","Transformer"]):
            ax=axs[i,j];rs=sorted([r for r in rows if r["kind"]==kind and r["window"]==w],key=lambda r:r["seed"])
            if len(rs)!=3:continue
            tables=[hist[r["run_id"]].set_index("step") for r in rs]
            steps=sorted(set.intersection(*(set(t.index) for t in tables)))
            for key,c in [("train_acc",BLUE),("test_acc",RED)]:
                vals=np.array([t.loc[steps,key].values for t in tables])*100
                mu,sd=vals.mean(0),vals.std(0)
                ax.plot(np.array(steps)/1000,mu,color=c);ax.fill_between(np.array(steps)/1000,np.clip(mu-sd,0,100),np.clip(mu+sd,0,100),color=c,alpha=.16)
                for step,m,s in zip(steps,mu,sd):agg.append(dict(model=kind,window=w,step=step,metric=key,mean_percent=m,std_percent=s,n=3))
            axes(ax,(-2,103))
            ax.set_title("%s · %s=%d\n达到90%%：%d/3；严格Grokking：%d/3"%(kind,"H" if kind=="TGLT" else "W",w,sum(r["reached90"] for r in rs),sum(r["audit_fixed_grokking"] for r in rs)))
            if j==0:ax.set_ylabel("准确率（%）")
            if i==3:ax.set_xlabel("训练步数（千步）")
    fig.suptitle("相同输入与读出：TGLT 与局部窗口 Transformer",fontsize=21,weight="bold",y=.99,color=BLACK)
    fig.legend(handles=[Line2D([0],[0],color=BLUE,label="训练均值"),Line2D([0],[0],color=RED,label="测试均值")],loc="upper center",bbox_to_anchor=(.5,.954),ncol=2,frameon=False)
    fig.text(.5,.018,"每条均值始终包含 3 个 seed，浅色带为 ±1 标准差。TGLT：62,787 参数；Transformer（FFN=256）：109,051 参数。",ha="center",fontsize=10.5)
    save(fig,"02_TGLT与Transformer平均曲线.png",top=.92)
    a.write_csv(agg,O/"mean_std_curves.csv")
def mechanisms(rows,hist,saved=False):
    paths=[B/"mechanism"/r["run_id"]/"complete.json" for r in rows if r["window"] in [8,12]]
    if not saved:
        if len(paths)!=12 or not all(p.exists() for p in paths):return
        a.aggregate()
    df=pd.read_csv(B/"mechanism/curves.csv")
    ep=pd.read_csv(B/"mechanism/endpoint.csv")
    pos=pd.read_csv(B/"mechanism/positions.csv")
    # Same-seed representatives chosen from TGLT successes before inspecting head curves.
    reps=[]
    for w in [8,12]:
        rs=[r for r in rows if r["kind"]=="TGLT" and r["window"]==w]
        candidates=[r for r in rs if r["audit_fixed_grokking"]] or [r for r in rs if r["reached90"]] or rs
        candidates=sorted(candidates,key=lambda r:(float("inf") if r["t90"] is None else r["t90"],r["seed"]))
        seed=candidates[(len(candidates)-1)//2]["seed"]
        for kind in ["TGLT","Transformer"]:
            reps.append(next(r for r in rows if r["kind"]==kind and r["window"]==w and r["seed"]==seed))
    a.write_csv(reps,O/"representatives.csv")
    fig,axs=plt.subplots(3,4,figsize=(20,11),sharex=True,sharey="row")
    cmax=max(abs(df[["C_fixed100k%d"%j for j in range(1,5)]].min().min()),abs(df[["C_fixed100k%d"%j for j in range(1,5)]].max().max()))
    for col,row in enumerate(reps):
        z=df[df.run_id==row["run_id"]].sort_values("step");h=hist[row["run_id"]]
        axs[0,col].plot(h.step/1000,h.train_acc*100,color=BLUE)
        axs[0,col].plot(h.step/1000,h.test_acc*100,color=RED)
        for j in range(4):
            axs[1,col].plot(z.step/1000,z["C_fixed100k%d"%(j+1)],color=COLORS[j])
            axs[2,col].plot(z.step/1000,z["q%d"%(j+1)]*100,color=COLORS[j])
        tag="严格G" if row["audit_fixed_grokking"] else ("达到90%" if row["reached90"] else "未达90%")
        axs[0,col].set_title("%s · %s=%d · seed %d\n%s"%(row["kind"],"H" if row["kind"]=="TGLT" else "W",row["window"],row["seed"],tag),fontsize=12)
        for ri in range(3):axes(axs[ri,col]);events(axs[ri,col],row)
        axs[0,col].set_ylim(-2,103);axs[1,col].axhline(0,color="#888",lw=.8);axs[2,col].set_ylim(-2,103)
        axs[2,col].set_xlabel("训练步数（千步）")
    axs[0,0].set_ylabel("训练/测试准确率（%）")
    axs[1,0].set_ylabel("条件判别贡献")
    axs[2,0].set_ylabel("写回强度份额（%）")
    fig.suptitle("四个 Mode 与四个 Attention Head：同口径的完整轨迹",fontsize=21,y=.995,weight="bold",color=BLACK)
    handles=[Line2D([0],[0],color=COLORS[j],label="Mode / Head %d"%(j+1)) for j in range(4)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.955),ncol=4,frameon=False)
    fig.text(.5,.022,"上排蓝色为训练、红色为测试；下两排各颜色代表一条通道。错误类别固定为 100k 时最强错误类。Transformer 的条件分解包含 MLP 与 LayerNorm。\n代表 seed 按 TGLT 严格G优先、首次90%时间居中选择，再配同 seed Transformer；全部 seed 数据保留。",ha="center",fontsize=10.5)
    save(fig,"03_Mode与Head机制对照.png",top=.9,bottom=.1)
    # Retain every seed, including counterexamples.
    for w in [8,12]:
        fig,axs=plt.subplots(3,4,figsize=(20,10),sharex=True,sharey="col")
        for s in range(3):
            for j,kind in enumerate(["TGLT","Transformer"]):
                row=next(r for r in rows if r["kind"]==kind and r["window"]==w and r["seed"]==s)
                z=df[df.run_id==row["run_id"]].sort_values("step");h=hist[row["run_id"]]
                ax=axs[s,j];ax.plot(h.step/1000,h.train_acc*100,color=BLUE);ax.plot(h.step/1000,h.test_acc*100,color=RED);axes(ax,(-2,103));events(ax,row)
                ax.set_title("%s · seed %d"%(kind,s))
                ax=axs[s,j+2]
                for k in range(4):ax.plot(z.step/1000,z["C_fixed100k%d"%(k+1)],color=COLORS[k])
                axes(ax);events(ax,row);ax.axhline(0,color="#888",lw=.8);ax.set_title("%s：4路条件贡献"%kind)
        values=df[df.window==w][["C_fixed100k%d"%j for j in range(1,5)]].values
        low,high=float(values.min()),float(values.max())
        pad=max(.5,(high-low)*.05)
        for ax in axs[:,2:].flat:ax.set_ylim(low-pad,high+pad)
        for ax in axs[-1]:ax.set_xlabel("训练步数（千步）")
        axs[1,0].set_ylabel("准确率（%）");axs[1,2].set_ylabel("条件判别贡献")
        fig.suptitle("H / W = %d：三个 seed 全部保留"%w,fontsize=21,y=.99,weight="bold")
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.95),ncol=4,frameon=False)
        fig.text(.5,.018,"前两列：蓝色训练、红色测试。后两列：固定100k最强错误类的四路条件贡献，统一纵轴。浅橙色为首次完整10%→90%区间。",ha="center",fontsize=10.5)
        save(fig,"04_H%02d_全部seed的条件贡献.png"%w,top=.9,bottom=.09)
    # Position replacement maps, same data/procedure for both architectures.
    fig,axs=plt.subplots(4,2,figsize=(16,12),sharex=True)
    for i,row in enumerate(reps):
        for j,src in enumerate(["a","b"]):
            z=pos[(pos.run_id==row["run_id"])&(pos.source==src)]
            phase_order=["before","during","after"]
            mat=np.stack([z[z.phase==p].sort_values("position").influence_share.values*100 for p in phase_order])
            ax=axs[i,j];im=ax.imshow(mat,aspect="auto",vmin=0,vmax=100,cmap="Blues")
            steps=[int(z[z.phase==p].step.iloc[0]) for p in phase_order]
            ax.set_yticks([0,1,2]);ax.set_yticklabels(["%gk"%(s/1000) for s in steps])
            ax.set_title("%s %s=%d · seed %d · 改变输入%s"%(row["kind"],"H" if row["kind"]=="TGLT" else "W",row["window"],row["seed"],src))
            ax.set_xticks(range(16));ax.set_xticklabels(range(16),fontsize=9)
            for ri in range(3):
                for c in range(16):
                    if mat[ri,c]>=5:ax.text(c,ri,"%.0f"%mat[ri,c],ha="center",va="center",color="white" if mat[ri,c]>55 else "black",fontsize=9)
            if i==3:ax.set_xlabel("被替换的第1层位置")
    fig.suptitle("第1层的信息经过哪些位置影响最终预测？",fontsize=21,y=.995,weight="bold")
    fig.text(.5,.035,"每次只把一个位置换成另一道题（a或b加1）的第1层状态，随后重新计算。颜色为单位置干预强度归一化后的占比（%），不是可加的因果分解。\n无完整跃迁的 Transformer 参考同 seed TGLT 的 10%/50% 事件选最近保存点；若均无完整跃迁，取 25k、50k、100k。",ha="center",fontsize=10.5)
    fig.tight_layout(rect=[.02,.11,.92,.94])
    cax=fig.add_axes([.94,.22,.012,.57])
    fig.colorbar(im,cax=cax,label="单位置干预强度占比（%）")
    fig.savefig(O/"05_第1层反事实位置替换.png",dpi=180,bbox_inches="tight");plt.close(fig)
    conds=["native"]+["L2_only_%d"%j for j in range(1,5)]+["L2_none"]
    mat=[];labs=[]
    for w in [8,12]:
        for kind in ["TGLT","Transformer"]:
            z=ep[(ep.window==w)&(ep.kind==kind)]
            mat.append([z[z.condition==c].test_acc.mean()*100 for c in conds]);labs.append("%s · %s=%d"%(kind,"H" if kind=="TGLT" else "W",w))
    fig,ax=plt.subplots(figsize=(13,5.5));im=ax.imshow(mat,cmap="Blues",vmin=0,vmax=100,aspect="auto")
    ax.set_xticks(range(6));ax.set_xticklabels(["完整模型","仅保留第1路","仅保留第2路","仅保留第3路","仅保留第4路","全部关闭"])
    ax.set_yticks(range(4));ax.set_yticklabels(labs)
    for i in range(4):
        for j in range(6):ax.text(j,i,"%.2f%%"%mat[i][j],ha="center",va="center",color="white" if mat[i][j]>55 else "black",fontsize=13)
    fig.colorbar(im,ax=ax,label="测试准确率（%）",shrink=.85)
    ax.set_title("100k 单通道消融：只保留第二层一条 Mode / Head",fontsize=19,pad=20,weight="bold")
    fig.text(.5,.015,"均值包含全部3个seed。第一层保持完整；MLP、LayerNorm、注意力与读出按干预后的状态实际计算。",ha="center",fontsize=10.5)
    save(fig,"06_终点单Mode与单Head消融.png",top=.94,bottom=.1)
    # Chinese phase and endpoint result tables.
    a.write_csv([dict(模型=r.kind,窗口=int(r.window),种子=int(r.seed),干预=r.condition,测试准确率百分比=100*r.test_acc) for r in ep.itertuples()],O/"终点消融汇总.csv")
    relay=pos.groupby(["run_id","kind","window","seed","step","phase","source"],as_index=False).agg(relay_share=("relay_share","first"),whole_input_change_rms=("whole_input_change_rms","first"))
    relay.to_csv(O/"relay_summary.csv",index=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--saved",action="store_true",help="Redraw from delivered histories and mechanism CSVs, without checkpoints.")
    args=parser.parse_args()
    rows,hist=collect_saved() if args.saved else collect()
    accuracy(rows,hist);paired_means(rows,hist);mechanisms(rows,hist,saved=args.saved)
    print(json.dumps({"runs":len(rows),"complete":sum(r["complete"] for r in rows),"figures":[p.name for p in O.glob("*.png")]},ensure_ascii=False))
if __name__=="__main__":main()
