"""Render the comparison at three original checkpoints and the time courses used to read it."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(),
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b9c4c5","axes.labelcolor":"#23363a",
                     "text.color":"#152d31","xtick.color":"#394e51","ytick.color":"#394e51",
                     "axes.unicode_minus":False,"savefig.facecolor":"white"})
INK = "#183f57"
GREEN = "#00786e"
GRAY = "#697882"
ORANGE = "#a75515"
MAGENTA = "#864367"
FORKS = ["fast_same_time","slow_same_time","slow_matched_stage"]
FORK_LABELS = ["快轨迹 · 10.55 万步","慢轨迹 · 10.55 万步","慢轨迹 · 25.60 万步"]
ARMS = ["native","increase","decrease","shuffled"]
ARM_LABELS = ["原样续训","加强数字排列","减弱数字排列","打乱顺序对照"]
STYLES = [(INK,"-","o"),(GREEN,"-","s"),(MAGENTA,"--","^"),(ORANGE,":","D")]


def read(name):
    with (OUT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def main_figure():
    states = {r["fork"]:r for r in read("matched_states.csv")}
    arms = {(r["fork"],r["arm"]):r for r in read("branch_outcomes.csv")}
    fig = plt.figure(figsize=(12.5,4.85))
    gs = fig.add_gridspec(1,2,left=.14,right=.985,top=.78,bottom=.24,width_ratios=[1.08,1.02],wspace=.20)
    left,right = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.text(.038,.936,"训练题上的改进，何时能帮模型算对新题？",fontsize=21,weight="bold")
    fig.text(.04,.86,"H12 模型  ·  870 道训练题  ·  2,611 道留出题",fontsize=11,color=GRAY)
    left.set_title("相同训练信号对新题正确答案分数的影响",loc="left",fontsize=12.2,pad=10,weight="bold")
    for i,f in enumerate(FORKS):
        for offset,col,key in [( .14,GREEN,"transfer"),(-.14,GRAY,"with_query_removed")]:
            val=float(states[f][key])
            left.barh(2-i+offset,val,height=.235,color=col,zorder=3)
            left.text(val+(.008 if val>=0 else -.008),2-i+offset,f"{val:+.2f}",
                      ha="left" if val>=0 else "right",va="center",fontsize=10,color=col,weight="bold")
    left.set_xlim(-.16,.39)
    left.set_ylim(-.52,2.5)
    left.set_yticks([2,1,0],FORK_LABELS)
    left.tick_params(axis="y",length=0,pad=11)
    left.axvline(0,color="#51646a",lw=1)
    left.grid(axis="x",lw=.6,color="#e2e8e9",zorder=0)
    left.spines["left"].set_visible(False)
    left.set_xticks([-.1,0,.1,.2,.3])
    left.set_xlabel("相对于训练题分数增量的比例")
    left.legend([Rectangle((0,0),1,1,color=GREEN),Rectangle((0,0),1,1,color=GRAY)],
                ["原模型","关闭位置 15 两个数的交互"],loc="lower left",bbox_to_anchor=(-.02,-.34),
                ncol=2,frameon=False,fontsize=9.5,handlelength=1.3)
    right.set_title("各分支达到早期泛化标准所需的步数",loc="left",fontsize=12.2,pad=10,weight="bold")
    for i,f in enumerate(FORKS):
        for j,a in enumerate(ARMS):
            raw=arms[(f,a)]["primary_onset"]
            cell_color={"1000":"#c4e8de","1500":"#d5ede6","2000":"#e3f1ed","2500":"#edf4f1","":"#f1f3f4"}[raw]
            right.add_patch(Rectangle((j+.025,2-i-.38),.95,.76,facecolor=cell_color,edgecolor="none"))
            right.text(j+.50,2-i,format(int(raw),",d") if raw else "未达",
                       ha="center",va="center",fontsize=13,color=GREEN if raw else GRAY,weight="bold" if raw else "normal")
    right.set_xlim(0,4);right.set_ylim(-.52,2.50)
    right.set_xticks(np.arange(4)+.5,["原样","加强排列","减弱排列","打乱对照"])
    right.tick_params(axis="x",length=0,pad=11)
    right.set_yticks([])
    for s in right.spines.values():
        s.set_visible(False)
    right.text(0,-.28,"每次只改前 1,000 步，随后原样续训 3,000 步。",transform=right.transAxes,color=GRAY,fontsize=9.5)
    fig.text(.04,.028,"早期泛化要求连续三次评估（间隔 500 步）中新题准确率达到 10%，且每次关闭位置 15 的交互均使准确率至少下降 5 个百分点。",fontsize=9,color=GRAY)
    fig.savefig(OUT/"waiting_mechanism.png",dpi=180)
    plt.close(fig)


def trajectories():
    original=read("original_embedding.csv")
    response=[r for r in read("training_response.csv") if r["scope"]=="all" and r["force"]=="true"]
    curves=read("branch_curves.csv")
    fig,axes=plt.subplots(2,3,figsize=(14.2,7.85))
    fig.subplots_adjust(left=.069,right=.98,top=.815,bottom=.15,wspace=.27,hspace=.69)
    fig.text(.045,.942,"两条原训练轨迹与三个检查点的短续训",fontsize=20,weight="bold")
    fig.text(.046,.878,"原轨迹",fontsize=13,weight="bold")
    for suffix,label,col in [("s0","快轨迹（seed 0）",GREEN),("s2","慢轨迹（seed 2）",ORANGE)]:
        rows=[r for r in original if r["run_id"].endswith(suffix)]
        steps=np.array([int(r["step"]) for r in rows])/1e4
        axes[0,0].plot(steps,[100*int(r["test_correct"])/2611 for r in rows],c=col,lw=1.25,label=label)
        axes[0,1].plot(steps,[100*float(r["order"]) for r in rows],c=col,lw=1.45)
        pts=[r for r in response if r["run_id"].endswith(suffix)]
        axes[0,2].plot([int(r["step"])/1e4 for r in pts],[float(r["transfer"]) for r in pts],lw=1.2,c=col,marker="o",ms=3)
    for ax,title in zip(axes[0],["留出题答对比例（%）","数字相似度中由模 59 的差解释的比例（%）","训练题分数提高时，新题分数的响应比"]):
        ax.set_title(title,fontsize=10.6,loc="left")
        ax.set(xlim=(0,30),xticks=[0,5,10,15,20,25,30],xlabel="原训练步数（万）")
        ax.grid(axis="y",color="#e1e6e6",lw=.6)
    for ax in axes[0,:2]:
        ax.set(ylim=(-3,103),yticks=[0,25,50,75,100])
    axes[0,2].axhline(0,color=GRAY,lw=.8,ls="--")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="center left",bbox_to_anchor=(.13,.89),
               ncol=2,frameon=False,fontsize=9.5)
    fig.text(.046,.461,"短续训",fontsize=13,weight="bold")
    fig.text(.144,.462,"浅色区域为施加干预的 1,000 步，虚横线为 10% 准确率。折线取每 500 步的实测值。",color=GRAY,fontsize=10)
    for ax,f,label in zip(axes[1],FORKS,FORK_LABELS):
        for a,desc,(col,ls,marker) in zip(ARMS,ARM_LABELS,STYLES):
            rows=[r for r in curves if r["fork"]==f and r["arm"]==a and int(r["relative_step"])%500==0]
            ax.plot([int(r["relative_step"]) for r in rows],[100*int(r["test_correct"])/2611 for r in rows],
                    c=col,ls=ls,marker=marker,ms=3.8,lw=1.7,alpha=.92,label=desc)
        ax.axhline(10,color="#859497",lw=.8,ls="--",zorder=0)
        ax.axvspan(0,1000,color="#eef3f0",zorder=-1)
        ax.set_title(label,loc="left",fontsize=11)
        ax.set(ylim=(0,19),yticks=[0,5,10,15],xlim=(-35,4035),xticks=[0,1000,2000,3000,4000],xlabel="从检查点起的续训步数")
        ax.grid(axis="y",lw=.5,color="#e1e6e6")
    axes[1,0].set_ylabel("留出准确率（%）")
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc="center",bbox_to_anchor=(.52,.043),ncol=4,frameon=False)
    fig.savefig(OUT/"waiting_trajectories.png",dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    main_figure()
    trajectories()
