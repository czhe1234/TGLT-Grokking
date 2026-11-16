# 无标签内部场观测：H=8，0–40k

只使用已有检查点和 CPU。5 条运行，共 325 个检查点；每个检查点均使用全部 3,481 组输入，不构造答案标签，不读取训练/测试划分，不调用答案读出层。不包含新训练。

原跨度 15：`r2_T4_M4_seed1` 为后续严格 Grokking，`seed0` 在 100k 未达测试 90%。跨度 11：三个 seed 后来均达 90%；seed0 为宽跃迁，seed1/2 为严格 Grokking。后续分类来自已冻结的旧 roster，仅用于命名，不参与观测量计算。它们的 t50 分别为 57k，以及 59.5k/69.5k/43.5k，均在本次窗口之后。

## 观测量

- 两层位置状态的中心化归一化关联。原位置计算 G(0,15)、G(0,7)，新位置计算 G(0,11)、G(0,5)。这些是输入分布上的关联，不能把共同输入产生的关联当成直接因果传播。
- 第二层查询状态对第 0 格嵌入主状态的完整 64×64 Jacobian。每个输入精确计算，保留 LayerNorm 的完整导数；曲线报告输入平均 Frobenius 范数，另保留均方根及分位数。它是输入响应，不是热平衡磁化率。
- 第二层查询位总实际写回的中心化协方差谱：自然对数谱熵 S，逆参与率 D=1/sum(p²)。单独保留 Mode3/4 谱作为补充。这里的谱熵不是热力学熵。
- Mode3/4 实际写回的中心化归一化重叠。它测共同涨落的对齐，不直接测耦合常数，也不是已经验证的自旋玻璃序参量。

输入涨落取全部输入上的样本平均，不是温度驱动的热涨落。全部输入包括训练和留出输入，所以是“不用标签”，并不是“不用留出输入”。

## 结果与限制

`results/` 有四张完整曲线、关键值和分叉筛选表。原配对中，成功运行的写回谱较集中：40k 的 S 为 2.179 对 2.327，D 为 6.823 对 8.459。最早 1k 就出现满足预设幅度、并保持至少 5k 的谱差异；若要求该幅度之后一直保持到 40k，S 从 30k、D 从 34k 才满足。此处阈值仅是描述性筛选，见 `protocol.json`，不是显著性检验或已校准预警线。

原配对的响应都大幅增强，40k 平均范数为 20.529 对 20.455；不能据此区分是否学会。Mode 重叠有差别但回落明显。跨度 11 后续成功的三个 seed 在 40k 的 S 为 2.847/2.826/2.956、D 为 12.232/11.236/14.766；低谱熵/低参与维数不能用同一绝对阈值跨几何判别成功。只有一对同配置成功/失败，不支持 seed 层面的显著性或预警可靠率。

## 复算

环境：服务器 transformer，`/home/chenzhe/.conda/envs/risk-exp4/bin/python`（torch 2.4.1，CPU float64），新图使用 Noto Sans CJK。从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' /home/chenzhe/.conda/envs/risk-exp4/bin/python experiments/label_free_physics_20261116/analyze.py --validate
CUDA_VISIBLE_DEVICES='' /home/chenzhe/.conda/envs/risk-exp4/bin/python experiments/label_free_physics_20261116/analyze.py --workers 5
CUDA_VISIBLE_DEVICES='' /home/chenzhe/.conda/envs/risk-exp4/bin/python experiments/label_free_physics_20261116/verify.py
CUDA_VISIBLE_DEVICES='' /home/chenzhe/.conda/envs/risk-exp4/bin/python experiments/label_free_physics_20261116/plot.py
```

检查点路径由 `roster.json` 和 `sources.csv` 指向原始服务器；另可设置 `TGLT_SOURCE_ROOT`。大权重未上传。仅重画图不需要检查点，只需已保存的 `curves.csv`。

原运行每 1,000 步保存，新运行每 500 步保存。本分析不补点、不外推。跨度 11 总配置里的顶层 r/T/query 字段沿用总任务模板；实际几何按 geometry 子任务、run_id/H 及已保存传播核核验，不能误当作开关任务的 r=3,T=4,query=15。325 个检查点均独立核对真实几何。

15 个代表检查点的精确 Jacobian 与自动微分最大误差 2.0e-15，中心有限差分最大误差 4.1e-10；几何不允许的一层端到端 Jacobian 严格为零。完整核验在 `validation/`。CPU 分析启动代码提交为 `9e0ba02`；每个原检查点均记录 SHA256。

## SwanLab

独立本地看板：http://127.0.0.1:16099/charts 。不混入训练和其他实验，不上传云端。

```bash
ssh -N -L 16099:127.0.0.1:16099 transformer
```

服务器日志位于本目录 `swanlab/logs`，必须先有 `runs.swanlab` 再启动 `.venv-swanlab/bin/swanlab watch <日志目录> -h 127.0.0.1 -p 16099`。`swanlab_export.py` 用现有 SwanLab 环境导入，`--verify 16099` 核对 2,600 个标量。图中颜色按 seed 固定，线型全部为实线。

物理术语参照 [MIT Statistical Mechanics II lecture 2](https://web.mit.edu/8.334/www/lectures/lec2.pdf)：标准热平衡磁化率与共轭外场及平衡涨落有关。本分析不假定网络满足热平衡涨落耗散关系。
