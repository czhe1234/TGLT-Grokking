# 内部指标口径

主要测量位置是第二层查询位置15。
`u_m(x)`表示该位置的一个mode写回，不含输出偏置，不含残差和LayerNorm。
`u(x)=Σ_m u_m(x)`为总写回。输入x=(a,b)。

## 实际写回强度与份额

\[
A_m=\frac1{59^2}\sum_{a,b}\|u_m(a,b)\|_2,\quad A=\sum_m A_m,\quad q_m=A_m/A.
\]

表中的s1至s4是A_m，q1至q4是份额。A不是耦合系数，也不是规则投影能量。
这些量使用实际输入和状态，不使用答案。它们测量mode对主状态作出的改动幅度。

## 写回关联的各向异性

先对每道题的总写回减去特征均值，再在完整a、b网格上扣除单输入分量：

\[
X=u-\operatorname{mean}_{feature}u,\quad
J=X-\mathbb E_a X-\mathbb E_b X+\mathbb E_{a,b}X.
\]

对非退化的J逐题归一化为z=J/‖J‖。
在模59输入平面中，60个方向各包含58个非零位移。
每个方向求沿该方向所有有效题对的z点乘平均，得到C_d。

\[
C_{sd}=\sqrt{\frac1{60}\sum_d(C_d-\overline C)^2}.
\]

C_sd上升表示不同输入变化方向的写回关联更不均匀。
方向属于输入变化空间，不是16个序列位置的几何方向。
C_sd不使用答案，但使用全部输入，包含测试输入。
C_add专门选择保持a+b不变的方向，因此它在解释上使用已知任务规则。
相对退化阈值、有效题对分母与FFT实现保存在历史measure_beta.py和geometry.py中。

## 规则写入强度D、交叉项I、相对增强率rho

固定该次完整前向的LayerNorm除数s(x)，定义每个mode在LayerNorm后的可加分量：

\[
v_m(x)=\gamma\odot[u_m(x)-\operatorname{mean}_{feature}u_m(x)]/s(x).
\]

该分量不是“仅保留此mode后重新计算LayerNorm”的输出。
按正确答案y分组，先取类条件均值，再减去全局均值：

\[
\mu_m(y)=\mathbb E[v_m\mid y]-\mathbb E[v_m],\qquad
S_{mn}=\frac1{59}\sum_y\langle\mu_m(y),\mu_n(y)\rangle.
\]

\[
D=\sum_m S_{mm},\quad I=\sum_{m\ne n}S_{mn},\quad \rho=I/D.
\]

D测量各mode分别携带的规则分量平方强度。I测量组合时增加或抵消的交叉项。
D+I为合并后的规则分量平方强度。rho不是夹角余弦，可以大于1。
这些规则量使用任务答案，不是无标签预警指标。
raw_D使用LayerNorm之前的u_m，不能和D混用。

## 历史表中的P

本批trajectories.csv.gz中的P是归一化有效耦合系数向量的平均向量长度：

\[
P=\left\|\frac14\sum_m\widehat{\mathcal B}_m\right\|_2.
\]

有效系数包含固定指数传播权重及两路方向；归一化时各mode等权。
P来自参数，不使用输入答案。它不是D+I。
历史不同报告曾重复使用P这个符号，本包以本表定义为准。
对应公式实现附在provenance中的firstgen_mode_direction_arrangement_20261102/measure.py。

## 单mode消融

保持第一层、第二层输入、残差、偏置和已训练读出不变。
在第二层查询位置保留一个mode的写回，另外三个置零；不重新归一化gate。
LayerNorm正常重新计算。
另提供均方根匹配的单mode对照。它将单mode写回乘一个全输入网格统一系数，使特征去均值后的均方根等于总写回。
因此它控制整体写回幅值，但不固定每道题的LayerNorm除数。
此干预测当前计算作用，不等于训练形成机制，也不是能量或磁化率测量。
