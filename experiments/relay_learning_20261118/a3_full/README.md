# A3 全替换与原网络完整前向

[实验 A3 与实验 B，两页中文结果及四张三线表](results/A3与实验B_机制结果.docx)

H12 主对照在 4,000、107,000、155,000、300,000 步取种子 0、2，严格中继 H8 在 3,000、51,000、58,000、100,000 步取种子 1、0。每点使用原始权重与上次保存的原设备 AdamW 一步后权重。末次检查点的下一步只检验更新方向。原始检查点与一步后缓存的哈希在 [sources.csv](results/sources.csv)。

## 测量

每次只改变第一个加数，在全部 59 个固定 b 条件下各检查 59 × 58 对不同 a。每个不同输入的激活来自原网络前向，同一激活可用于多对比较。四个模式分别报告接收信号的平均平方变化、源信号变化，以及局部写回对两个数的混合差分。双输入量同时列出两个数都加一、a 全替换且 b 加一、a 与 b 各自全替换三种口径。全 b 替换用固定 a、a′ 后的成对平方差与方差恒等式精确平均，并以去除两项主效应后的写回方差独立校验。每种模式和每个偏移的均值见 [replacement_offsets.csv](results/replacement_offsets.csv)，按状态汇总见 [state_summary.csv](results/state_summary.csv)。

身份读出只用 870 道训练题拟合，在 2,611 道留出题上评估。原实现的内积展开在极弱单模式上出现过数值抵消。新读出平移全部向量并直接计算到类中心的平方距离，同一分类规则下四个单模式和合并读出均为 100%。旧读出、未居中的 GPU 读出和稳定读出均见 [modes.csv](results/modes.csv)。

16 种模式组合均从原始 token 和角色开始前向。仅在第一层的传播器第一次调用时清零位置 7 指定模式的左右激活，无门控重归一化。原来的残差、投影偏置、各处归一化、整个第二层和分类器继续执行。四个机制对照另用原网络从头重算。旧法的逐题 59 类分数在相同权重、相同条件下重建，并校验旧档案中的训练与测试正确数、损失和间隔。[逐条件正确率、损失、间隔与分数误差](results/full_forward.csv) · [核验汇总](results/verification.json)

固定精度后，所有模式组合的 1,782,272 次预测与原等价算法完全一致。换回历史单精度，17 次逐题预测与旧双精度不同，每个条件最多差 1 道留出题。全替换改变传播强度，其四模式排序在全部 32 个状态下保持。64 个“模式 × 单步”对比中，两项微小增减反号，分别是 H12 种子 0 第 4,000 步和严格中继 H8 种子 0 第 58,000 步的模式 2。H12 位置 15 与 H8 位置 7 的关键干预结论保持。实验 B 的状态替换、方向删除与功能恢复采用上轮原始记录，[百分比与分母](results/experiment_B_percentages.csv) 可与父目录结果逐行对照。

## 复现

在 transformer 的本目录执行。CPU 参考用既有 risk-exp4 环境，GPU 用原 PyTorch 2.5.1+cu121。输入还包括父目录 .native_cache 中已经验证的原精度单步权重。如从原始检查点恢复该缓存，用父目录 native_step.py 在其指定设备运行。reference.py 的参考分数和 full_forward.py 的逐点输出写入 .work，可在同一版本和输入下续跑。复现不同版本时使用新的工作副本。

```sh
CUDA_VISIBLE_DEVICES='' /home/chenzhe/.conda/envs/risk-exp4/bin/python reference.py
CUDA_VISIBLE_DEVICES=GPU-70b6fceb-c7c4-9467-3c81-097c9284c160 /opt/anaconda3/bin/python full_forward.py b0.25_wd0.03_s0
CUDA_VISIBLE_DEVICES=GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009 /opt/anaconda3/bin/python full_forward.py b0.25_wd0.03_s2 r2_T4_M4_seed1 r2_T4_M4_seed0
python3 compare.py
```

SwanLab 采用本地记录，4 条曲线、288 项数值已对照结果表核验。[本地看板](http://127.0.0.1:16105/charts) · [核验记录](results/swanlab_validation.json)
