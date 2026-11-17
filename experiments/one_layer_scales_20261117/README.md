# 一层多尺度模加对照

一层、四个 mode，对比四种衰减尺度与四路共同长尺度。标签为 `(a+b)%59`；查询位含 b，远处为 a，近处为独立干扰 c。a/c 角色相同且在同侧，无绝对位置嵌入。

配置及用户请求见 config.json。870/2611 个有序数字组合先划分，再构造固定有限布局。四个训练相对距离组合与两个测试组合分离，查询位置训练为偶数、测试为奇数，所有位置保留完整双侧 H=8 邻域。训练数据不在线刷新。

common.py 复用原模型参数和主辅乘法，完整双侧有限传播核按 L2 范数归一化。训练使用单查询位置等价计算。validate.py 将前向与全部可训练梯度和原逐轮局域传播核对，并检查标签、角色、划分、平移不变性与两组初始化。检查失败返回非零退出码。

计算在 transformer 的 risk-exp4 环境执行；SwanLab 使用既有独立环境。每500步完整保存模型、优化器和随机状态。原始history与检查点在 runs/，权重不提交Git。

本次共同尺度为32，未扫描最佳共同尺度。未见数字组合在熟悉布局上的测试是主要算术泛化指标；另记录已见组合换布局、未见组合换布局。不把布局泛化单独称为算术Grokking。
## 在服务器复现

从仓库根目录运行以下命令。训练环境是 `risk-exp4`，监控环境是项目的 `.venv-swanlab`。`config.json` 保存完整配置和 GPU UUID。

```sh
/home/chenzhe/.conda/envs/risk-exp4/bin/python experiments/one_layer_scales_20261117/validate.py
```

该检查生成固定数据，验证前向、梯度和数据划分。失败信号是非零退出码。正式运行前，按 `pipeline.py` 中的三张卡分别执行短试跑：

```sh
/home/chenzhe/.conda/envs/risk-exp4/bin/python experiments/one_layer_scales_20261117/train.py --group multiscale --seed 0 --gpu GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009 --pilot
/home/chenzhe/.conda/envs/risk-exp4/bin/python experiments/one_layer_scales_20261117/train.py --group all_long --seed 1 --gpu GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0 --pilot
/home/chenzhe/.conda/envs/risk-exp4/bin/python experiments/one_layer_scales_20261117/train.py --group multiscale --seed 2 --gpu GPU-70b6fceb-c7c4-9467-3c81-097c9284c160 --pilot
```

每次试跑验证 CUDA graph 与普通 AdamW 更新，并记录 1,000 步测速。失败信号是非零退出码。试跑和正式运行均拒绝覆盖已有曲线；复现实验请在独立检出目录运行。

```sh
/home/chenzhe/.conda/envs/risk-exp4/bin/python experiments/one_layer_scales_20261117/pipeline.py
```

队列运行两组各三个 seed，每张卡先后运行同 seed 的两组。六条曲线均训练到 100,000 步。`runs/<组>_s<seed>/history.csv` 是原始曲线，`checkpoint_*.pt` 保存模型、优化器、随机状态、配置和已有历史。`summary.json` 同时给出 90% 达标、首次 10% 到 90% 宽度与原严格 Grokking 判据。执行或监控失败返回非零退出码，并在状态文件保留原因。

SwanLab 看板仅包含本实验，准确率显示百分比，其余指标保留原单位。

```sh
/home/chenzhe/work/tglt_experiments/.venv-swanlab/bin/swanlab watch experiments/one_layer_scales_20261117/swanlab/logs -h 127.0.0.1 -p 16100
```

Mac 通过 `ssh -N -L 16100:127.0.0.1:16100 transformer` 建立转发后，打开 `http://127.0.0.1:16100/charts`。SwanLab 失败不改变原始日志和模型文件。
