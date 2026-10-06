# 设计笔记：调研要点、关键取舍与数值核验

本文记录 LTEEG 背后的判断依据，供后续修改框架时参考。使用说明见 [README](../README.md)。

---

## 1. 问题定义

长程 EEG 癫痫检测的主流做法是**窗口级分类**：把记录切成几秒到几十秒的窗口，每个窗口输出一个“是否发作”的 logit，再把窗口序列拼接、平滑成事件。这个范式和图像分类同构，有三个问题：

1. 窗口内的时间结构只被用来产生一个标量，发作起止时间的分辨率被窗口长度/步长限制；
2. 为了得到连续标注，需要大量重叠推理和繁琐的后处理；
3. 标签粒度（窗口）与临床需求（起止时间）以及评估粒度（SzCORE 的 1 Hz sample 与带容差的 event）不一致。

LTEEG 采用**逐点（point-level）**形式：样本仍是窗口 `x ∈ R^{C×T}`，监督是 `y ∈ {0..K-1, ignore}^T`，模型输出 `R^{K'×T}`。这在形式上与三类成熟问题同构：

| 领域 | 对应 | 可借鉴之处 |
|---|---|---|
| 时间序列异常检测（TSAD） | 逐点标签、极度不平衡 | 逐点打分；但其常用的 point-adjust 评估已被证明会严重高估（随机分数也能得到接近满分的 F1，Kim et al., AAAI 2022），不应照搬 |
| 时序动作分割（TAS） | 未裁剪长序列的逐帧分类 | MS-TCN 的截断 MSE 平滑损失（抑制过分割/抖动）；以分段为单位的指标 |
| 睡眠分期稠密分割 | U-Time / U-Sleep：U 形网络逐点输出，再聚合 | 类别不平衡的处理（Dice、按类采样）；任意长度推理 |

SeizureTransformer 正是 U 形的逐点模型，它在 2025 SzCORE 挑战赛（65 名受试者、约 4360 小时、按事件级 F1 排名）中获得第一，因此作为 baseline。

## 2. 调研要点

### 2.1 SzCORE 评估（Dan et al., Epilepsia 2024）

- **事件级**：参考事件前扩 30 s、后扩 60 s，任意重叠即检出；间隔 < 90 s 的事件合并；长于 5 min 的事件切分；误报是未与任何已检出参考扩展区间重叠的预测事件；以每 24 小时误报数报告误报率。计算在 10 Hz 网格上进行。
- **sample 级**：在 **1 Hz** 网格上逐秒比较（不是原始采样率）。
- 官方实现 `timescoring` 的若干细节会影响计数：秒→采样点用 Python `round`（四舍六入五成双）、合并发生在切分之前、事件列表在重采样时不被重新量化。LTEEG 的 `evaluation/szcore.py` 逐行复刻了这些行为，并与官方包做随机对拍（见第 4 节）。
- 聚合方式：挑战赛按受试者平均（`avg_per_subject`），原训练脚本则把所有文件的计数相加（pooled）。两者都输出，并额外给出最差病人。

### 2.2 评分器的选择对数字影响极大

有人用同一组 SeizureTransformer 预测在 TUSZ 上比较了不同评分器：SzCORE 事件评分 8.59 次误报/24h（灵敏度 52.4%），NEDC OVERLAP 26.89 次/24h（45.6%），NEDC TAES 136.73 次/24h（65.2%）。同一个模型，误报率差 3 到 16 倍。因此：

- 报告时必须写明评分规则及其全部参数（LTEEG 的 `summary.md` 自动写出后处理参数，评分参数在配置的 `scoring.*` 中）；
- 不要把不同评分规则下的数字直接比较。

### 2.3 CHB-MIT 的已知问题

- 部分病人（如 chb12）中途更换过导联配置，部分文件不是双极导联；原始 EDF 中有虚拟通道（`--`）、重复的 `T8-P8`、极性相反的同名导联等。你提供的数据已统一为 18 路双极导联、256 Hz，LTEEG 仍然逐文件检查通道数，并在文件带通道名属性时按名字核对和重排。
- chb01 与 chb21 是同一受试者相隔 1.5 年的记录，必须在同一划分中（split 文件的 `groups` 强制检查）。
- 198 次发作中 chb12 一人占 40 次、chb15 占 20 次，pooled 指标会被少数病人主导，因此同时报告病人宏平均与最差病人，并提供 `sampling.group_by: patient`。

### 2.4 SeizureTransformer 的实现细节

- braindecode 社区移植时发现：原实现中 `self.transformer_encoder_layer` 被注册为子模块，但 `nn.TransformerEncoder` 会深拷贝传入的层，因此这个属性从未参与前向计算，却带着约 3.15M 参数（原 41.0M，去除后 37.85M）。它们会被优化器跟踪、被 weight decay 作用，在 DDP 下还需要 `find_unused_parameters`。
- 编码器在奇数长度时用 -1e10 右补再 max-pool，等价于 `MaxPool1d(ceil_mode=True)`；解码器的裁剪等价于“上采样后截到对应 skip 的长度”。按后者实现后，网络对任意输入长度成立。
- 原模型在 `forward` 内做 sigmoid，训练用 `F.binary_cross_entropy`；在 autocast 区域内 PyTorch 会拒绝这个组合，且概率接近 0/1 时数值不稳定。

## 3. 关键取舍

### 3.1 预处理缓存 + memmap，而不是物化窗口

原 `get_dataset.py` 把所有窗口（75% 重叠）先追加到 Python 列表、再下采样后存为一个大 `.npy`，内存峰值约为原始数据的 4 倍，CHB-MIT 规模下不可行；而且训练子集一旦固定就无法在每个 epoch 重抽背景、无法做时间平移增强。LTEEG 把每条记录预处理一次后存成 `.npy`，训练和推理都通过 memmap 按需读取窗口：内存占用与数据规模无关，随机访问一个窗口只是一次页缓存命中，背景重抽、抖动等采样策略都成为配置项。缓存目录名是所有影响数值的配置（导联、采样率、重采样、预处理流水线、dtype）的哈希，旁路 JSON 记录源文件大小和修改时间，任何变化都会触发对应记录的重建。

### 3.2 整条记录滤波，而不是逐窗滤波

原实现在每个窗口上独立做因果 IIR 滤波（零初始状态），每个窗口开头都会出现滤波器的启动瞬态：带通的 0.5 Hz 高通部分约 1 秒量级，1 Hz 陷波因为 Q=30（带宽仅 1/30 Hz）极点非常靠近单位圆，时间常数约 9.5 秒。训练和推理都受影响。整条记录滤波一次既消除了这个伪迹，也让任意起点的窗口（抖动、重叠推理）看到一致的信号，还能缓存。所用滤波器与原实现相同（3 阶 Butterworth 带通 + 两个 Q=30 的陷波），以 SOS 形式实现（b/a 形式在 0.5 Hz/128 Hz 这样的低归一化频率下条件数差），并把三个滤波器融合为一个级联。需要逐窗口复现原行为时，用 `configs/experiments/original_faithful.yaml`。

z-score 仍在滤波之前、按整条记录计算，与原实现一致。这意味着标准化用到了整条记录的统计量（离线标注场景没有问题，在线/流式部署需改为因果的滑动统计）。

### 3.3 采样器在主进程、按（种子, epoch）确定

窗口选择和顺序完全由 `numpy.random.default_rng([seed, epoch, ...])` 决定，通过 `torch.utils.data.Sampler` 把 `(记录, 起点)` 元组交给 Dataset。好处：与 worker 数量无关的可复现性；Windows 的 spawn 进程不需要同步采样状态；`persistent_workers` 下每 epoch 重抽依然生效；断点续训后数据顺序与不中断时一致。

### 3.4 验证即长程评估

验证不在切好的窗口集合上算准确率，而是对每条完整 dev 记录连续推理、后处理、SzCORE 打分，和最终报告用同一段代码。这样选出的 checkpoint 优化的就是要报告的量，也避免了“窗口级指标很好、长程误报很多”的错配。代价是每个验证 epoch 都要推理整个 dev（CHB-MIT 的 6 名 dev 病人约 200 多小时），可用 `train.val_every` 调节频率。

### 3.5 自研 SzCORE 实现

依赖 `timescoring` 的可编辑安装是原项目不能自包含的原因之一，而且需要在其基础上增加逐事件明细（检出延迟、覆盖率）和误报列表。因此重新实现，并用随机对拍保证与官方包计数一致；官方包只作为可选的测试依赖。

### 3.6 末窗对齐而非补零

补零的输入在 z-score 后表现为一段完全平直的信号，是训练中从未出现过的分布外输入，可能影响该窗口内真实部分的预测。默认在记录末尾追加一个右对齐的窗口，不需要补零；只有记录短于一个窗长时才补零，并且补零区域既不计入损失，也不计入评估。

### 3.7 严格校验而非容错

原代码中 `try/except: continue` 和“短于窗长就跳过”会让文件和标签在不报任何信息的情况下消失，评估结果也随之偏移。LTEEG 的原则是：能确定的问题就报错，并且一次列出所有问题；只有明确无害、可解释的情况（如发作结束时间超出记录末尾不到 1 秒）才自动修正，并记录告警。

### 3.8 有意保持原样的地方

为保证 baseline 可比，以下沿用原项目且默认不改：RAdam（L2 形式的 weight decay）、常数学习率、不裁剪梯度、无 AMP、100 epoch、batch 86、阈值 0.8、形态学核 5 个采样点（256 Hz 下仅约 20 ms，作用很小）、最短事件 2 s、按 pooled event F1 选模。`configs/experiments/st_robust_training.yaml` 汇集了常见的稳定化手段，作为对照实验的起点，它在真实数据上的效果尚未验证。

## 4. 数值核验记录

| 核验内容 | 方法 | 结果 |
|---|---|---|
| SeizureTransformer 移植 | 原 `model.py` 与移植版加载同一份权重，输入长度 15360 / 15000 / 3001（含奇数层长度），eval 模式比较 sigmoid 输出 | 最大绝对误差 1.2e-7；state dict 键完全匹配 |
| 参数量 | 18 通道、1 输出 | 原 41,000,929 → 37,848,545 |
| SzCORE 实现 | 与 `timescoring` 0.0.7 对拍，随机参考/预测掩码，记录长 1 分钟到 1 小时 | 300 组（测试中 150 组）TP/FP/参考数/样本数全部一致，fp/day 一致 |
| 预处理 | 融合 SOS 流水线 vs 原实现的 b/a `lfilter` 链（z-score→带通→陷波 1 Hz→陷波 60 Hz） | 相对误差 < 1e-6 |
| 形态学后处理 | 游程实现 vs `scipy.ndimage.binary_opening/closing`（内部区域），k = 2/3/5/8 | 完全一致 |
| 最短事件过滤 | 与原 `remove_short_events` 逐元素比较 | 完全一致 |
| 长程拼接 | 位置无关的“回声”模型，在 align/pad、重叠/不重叠、mean/hann 组合下拼接输出应等于逐点 sigmoid | 完全一致（误差 < 1e-6） |
| AUROC / AUPRC | 与 scikit-learn（含并列分数） | 一致 |

## 5. 后续可能的工作

- 记录级划分（同一病人前若干次发作训练、其余测试），支撑个体化研究；
- 逐类别的事件评估（多分类时）；
- 时序动作分割风格的分段指标（例如按 IoU 阈值的分段 F1）和起止点误差分布，作为对 SzCORE 容差评分的补充，更直接地衡量逐点模型的起止定位能力；
- 流式/因果推理模式（因果标准化、滑动缓存），用于在线监测场景；
- 多 GPU（DDP）训练。

## 6. 资料来源

- SeizureTransformer 论文与代码：arXiv:2504.00336；https://github.com/keruiwu/SeizureTransformer
- braindecode 中的 SeizureTransformer 移植及其核验讨论：https://github.com/braindecode/braindecode/pull/1236
- SzCORE 框架：https://onlinelibrary.wiley.com/doi/10.1111/epi.18113 ；arXiv:2402.13005
- 2025 挑战赛报告：arXiv:2505.18191
- 同一预测在不同评分器下的对比：https://github.com/Clarity-Digital-Twin/SeizureTransformer
- TSAD 评估中 point-adjust 的问题及后续指标的稳健性：arXiv:2607.11969（及其引用的 Kim et al., AAAI 2022）
- MS-TCN：https://openaccess.thecvf.com/content_CVPR_2019/papers/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.pdf
- U-Sleep：https://www.nature.com/articles/s41746-021-00440-5
- CHB-MIT 导联问题的讨论：https://github.com/PakkapanD/EEG-Research
