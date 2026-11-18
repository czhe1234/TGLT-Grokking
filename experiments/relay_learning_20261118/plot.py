"""Regenerate the two figures using saved result tables, without model checkpoints."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

OUT=Path(__file__).resolve().parent/"results"
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family":"Noto Sans CJK JP","font.size":10.5,"axes.unicode_minus":False,
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#bcc5cc",
    "xtick.color":"#445563","ytick.color":"#445563","axes.labelcolor":"#394c59","axes.titlecolor":"#1b323c"})
ink="#1b3541"
blue="#126b86"
brown="#bb531e"
purple="#755688"
gray="#788894"


def rows(name):
    with (OUT/name).open() as handle:
        return list(csv.DictReader(handle))


def grid(ax):
    ax.grid(axis="y",color="#e3e8eb",lw=.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=3,pad=4)


series=rows("trajectories.csv")
divergence=rows("functional_divergence.csv")
runs=["b0.25_wd0.03_s0","b0.25_wd0.03_s2","r2_T4_M4_seed1","r2_T4_M4_seed0"]
fig,axes=plt.subplots(2,2,figsize=(12.3,7.2))
fig.subplots_adjust(left=.09,right=.974,top=.78,bottom=.155,hspace=.47,wspace=.25)
fig.text(.065,.95,"两条轨迹从哪个计算环节拉开差距？",color=ink,weight="bold",fontsize=20)
fig.text(.065,.895,"相同架构与训练题，分组比较两次初始化。每个点都来自实际保存的模型。",color="#465b66",fontsize=11.5)
axes[0,0].set_title("原模型答对留出题的比例",fontsize=12,loc="left",pad=10)
axes[0,1].set_title("去掉指定交互后，两条轨迹损失的准确率相差多少",fontsize=12,loc="left",pad=10)
for i,width in enumerate((12,8)):
    faster,slower=([r for r in series if r["run_id"]==rid] for rid in runs[i*2:i*2+2])
    x=np.array([int(r["step"])/1000 for r in faster])
    def col(gr,k):
        return np.array([int(r[k])/26.11 for r in gr])
    left,right=axes[i]
    left.plot(x,col(faster,"original_test_correct"),color=blue,lw=1.9)
    left.plot(x,col(slower,"original_test_correct"),color=gray,lw=1.7)
    for k,color in (("joint_removed_correct",purple),("no_query_mod_correct",brown)):
        gain=(col(faster,"test_correct")-col(faster,k))-(col(slower,"test_correct")-col(slower,k))
        right.plot(x,gain,color=color,lw=1.8)
    check="no_query_mod_correct" if width==12 else "joint_removed_correct"
    clock=next(int(r["first_saved_step"])/1000 for r in divergence if int(r["H"])==width and r["operation"]==check and r["min_loss_and_gap_pp"]=="5")
    right.axvline(clock,color=gray,linestyle=":",lw=1.25)
    right.text(clock+.018*x[-1],72,"%g 千步"%clock,fontsize=10,color=ink)
    for ax in (left,right):
        ax.set_xlim(0,x[-1])
        ax.set_ylim(-8,104)
        ax.set_yticks([0,25,50,75,100])
        grid(ax)
    left.set_ylabel("H = %d\n准确率（%%）"%width)
    right.set_ylabel("差值（百分点）")
    right.axhline(0,color="#a1abb2",lw=.6,zorder=0)
    if i:
        left.set_xlabel("原训练步数（千步）")
        right.set_xlabel("原训练步数（千步）")
for x0,items in ((.081,[(blue,"较早泛化"),(gray,"较慢泛化")]),(.558,[(purple,"位置 7 的双输入交互"),(brown,"查询位置 15 的调制")])):
    fig.legend([Line2D([],[],color=c,lw=2.6) for c,_ in items],[n for _,n in items],loc="center left",bbox_to_anchor=(x0,.843),
               ncol=2,frameon=False,handlelength=1.7,columnspacing=1.1,fontsize=10)
fig.text(.065,.042,"竖线对应首次连续 3 个保存点，较快轨迹因干预损失至少 5 个百分点，且比较慢轨迹多损失至少 5 个百分点。",color="#51616b",fontsize=10)
fig.savefig(OUT/"A_训练轨迹.png",dpi=165,bbox_inches="tight",pad_inches=.17)
plt.close(fig)

fig,(ax,bx)=plt.subplots(1,2,figsize=(12.8,6.1),gridspec_kw={"width_ratios":[.8,1.2]})
fig.subplots_adjust(left=.105,right=.946,bottom=.18,top=.71,wspace=.7)
fig.text(.06,.946,"位置 7 传递了什么？删掉的功能能否补回来？",color=ink,weight="bold",fontsize=19)
fig.text(.06,.866,"每层可传播 8 格，两条轨迹均取第 10 万步。",color="#465b66",fontsize=11.5)
ax.set_title("只替换位置 7 后，最终输出跟着谁走？",loc="left",fontsize=11.5,pad=14)
p=[r for r in rows("patches.csv") if r["run_id"] in runs[2:] and r["step"]=="100000" and r["condition"]=="single_site" and r["position"]=="7" and r["split"]=="both_held"]
for i,rid in enumerate(runs[2:]):
    r=next(r for r in p if r["run_id"]==rid)
    for j,(k,c) in enumerate((("donor_prediction",blue),("donor_true",brown))):
        y=3-i*2-j*.56
        value=100*int(r[k])/int(r["n"])
        ax.hlines(y,0,value,color=c,alpha=.3,lw=2)
        ax.scatter(value,y,color=c,s=47,zorder=2)
        ax.text(value-1.8 if value>80 else value+2.6,y+.1,"%.1f%%"%value,ha="right" if value>80 else "left",fontsize=10,color=c)
ax.set_yticks([2.72,.72],["较早泛化\nseed 1","较慢泛化\nseed 0"])
ax.set_xlim(0,104)
ax.set_ylim(0,3.7)
ax.set_xticks([0,25,50,75,100])
ax.set_xlabel("占 1,941 对留出题的比例（%）")
ax.grid(axis="x",color="#e3e8eb",lw=.75)
ax.legend([Line2D([],[],color=c,marker="o",lw=0,ms=6) for c in (blue,brown)],
          ["跟随参照题的模型预测","给出参照题的正确答案"],loc="lower left",bbox_to_anchor=(-.13,-.45),frameon=False,fontsize=10)
bx.set_title("成功模型在各项干预后，2,611 道留出题答对几道？",loc="left",fontsize=11.5,pad=14)
rs={r["condition"]:int(r["test_correct"]) for r in rows("repairs.csv") if r["run_id"]==runs[2]}
settings=[("原模型","untouched",ink),("删第 9 方向","delete_u9",brown),("仅用本地 b 补回","operand_b_only",blue),
          ("用其余方向补回","surviving_write",blue),("在位置 8 补第 9 方向","site8_linear",blue),
          ("删位置 7 全部写回","remove_whole_write7",brown),("全删后用位置 8 的二次函数补回","whole7_repair_at8_quadratic",purple)]
for i,(_,key,color) in enumerate(settings):
    n=rs[key]
    bx.hlines(i,0,n,color=color,alpha=.25,lw=2)
    bx.scatter(n,i,s=47,color=color,zorder=2)
    bx.text(n+60,i,format(n,","),va="center",fontsize=10,color=color)
bx.set_yticks(range(len(settings)),[x[0] for x in settings],fontsize=9.6)
bx.invert_yaxis()
bx.set_xlim(0,3040)
bx.set_xticks([0,1000,2000,2611])
bx.grid(axis="x",color="#e3e8eb",lw=.75)
bx.set_xlabel("答对题数")
fig.text(.515,.025,"所有补偿器仅用原训练题拟合，在留出题上评估。",color="#51616b",fontsize=10)
fig.savefig(OUT/"B_位置7干预.png",dpi=165,bbox_inches="tight",pad_inches=.17)
plt.close(fig)
