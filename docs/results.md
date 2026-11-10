# 结果与图片索引

## 已有结果

架构扫描有192条，100k内有106条达到过90%测试准确率，其中32条满足已固定的严格Grokking判据。
32条严格Grokking都位于8≤rT≤14的范围。
该范围内共72条，其中49条达到90%。
rT≤7的60条均未达到90%。rT≥15的60条有57条达到90%，但严格标签为0条。
这些统计描述当前输入位置、读出位置和训练协议，不是关于传播范围的一般定理。

β×wd扫描有105条。按各自实际观察终点，严格Grokking为58条；另外25条达到过90%；22条未达到90%。
其中78条观察到300k，27条只观察到100k。不能把105条写成全部完成300k。

写回方向关联差异在58条严格Grokking的原窗口内均净上升。
55条有足够后续观测，49条满足原协议的保持条件。
但25条非严格、达到90%的运行中也有24条在首次10%–90%窗口内上升。
所以它与学习形成的内部组织有关，不能直接叫作Grokking特有的相变序参量。

在58条严格运行的跃迁末端，完整模型均达到90%。
保持第一层和读出不变，仅保留第二层查询位置的一路mode写回，四路中最好的单路也没有一条达到90%。
将单路写回放大到完整写回的相同均方根后，有1条达到90%。
这里的“最佳单路”用测试准确率选择，属于给予单路的事后上界，不是预先选定的部署方法。
这一消融说明冻结模型当前预测通常依赖多路组合。它不证明多mode对训练得到Grokking是必要条件；原架构扫描的M=1也有严格Grokking。

## 严格标签与画图窗口

严格标签保留原实现：首次训练准确率≥99%；从该点起测试准确率连续≤10%至少10000步；之后首次达到90%，跃迁宽度为正且小于低平台长度的一半。
严格跃迁起点是该连续低平台的最后记录点。
另有“首次完整10%–90%”的绘图窗口。两者可能不同。
每张图和数值表保留原口径，不相互替换。
采样间隔500步。首次达到阈值的记录时刻不是连续时间的精确穿越时刻。

## 图片清单

| 文件（figures/） | 内容 | 对应数值 |
|---|---|---|
| 01_architecture_scan_192.png | 架构扫描全部192条训练/测试曲线 | experiments/architecture_scan；results/original_architecture_scan_192 |
| 02_architecture_reachable_132.png | 去掉传播不足的60条后的132条 | 同上，全集仍保留 |
| 03_beta_wd_scan_105_at_100k.png | 105条前100k曲线 | experiments/beta_wd_scan |
| 04_beta_wd_35_seed_means_t50.png | 35配置、3seed均值，满足条件时按t50平移对齐 | results/all35_seed_means |
| 05_beta_wd_35_seed_means_raw.png | 相同35配置按原训练步数平均 | 同上 |
| 06_architecture_transition_width.png | 架构扫描三个seed跃迁宽度 | 全部历史与原窗口脚本 |
| 07_beta_wd_transition_width_available_300k.png | 续训批次现有观察范围的跃迁宽度 | results/scan_atlases/width_300_by_seed.csv |
| 08_write_direction_anisotropy_35.png | 总写回的方向关联差异 | results/mode_order_strength_300k/trajectories.csv.gz |
| 09_write_strength_35.png | mode实际写回强度 | 同上 |
| 10_mode_shares_35.png | 四个mode的强度份额 | 同上 |
| 11_matched_controls.png | 同配置对照 | matched_component_curves.csv.gz、matched_case_summary.csv |
| 12_same_config_counterexample.png | 同配置反例 | trajectories.csv.gz及原分析脚本 |
| 13_single_mode_ablation_58.png | 58条按t50对齐的单mode消融 | mode_function_aligned_means.csv及individual_values.npz |

图04中26个配置的3seed均达到50%，各自平移后平均。9个配置不具备3个t50，按原训练步数平均。
每个平均点始终包含3个seed，只使用共同观察范围，不外推。浅色带为seed间样本标准差，不是置信区间。
图13每个平均点始终包含58条运行。

## 复核边界

本次整理复核原文件哈希、配置/划分/历史覆盖、续训前缀和已有标签；不重新跑训练。
内部轨迹来自此前52305个检查点的CPU分析，本次没有全部重放。
此前CPU双精度重放与历史预测存在459个测试正确数差异点，最大相差3题；误差记录随包保留。
所有精确计数与原协议以results目录下的原JSON/CSV为准。
