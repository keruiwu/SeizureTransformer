# LTEEG：长程脑电逐点癫痫检测框架

LTEEG 是一个研究框架，用来做**长程脑电（long-term EEG）的癫痫发作自动检测与标注**。

它和主流做法最大的不同在于输出的粒度：

- **主流做法**：把 EEG 切成几秒到几十秒的窗口，每个窗口判断一次"是否发作"，本质上和图像分类是同一个范式。
- **LTEEG**：样本仍然是窗口，但模型对窗口里**每一个采样点**都输出一个发作 logit（逐点 / point-level）。标签形式和时间序列异常检测的逐点标签一致，也直接对应临床上"标注发作起止时间"的需求。

训练好的模型不在切好的样本集合上评估，而是**放回每个病人完整的长程记录上连续推理**，再按领域通用的 SzCORE 规则做事件级打分，结果逐病人呈现。

框架内置的 baseline 是 **SeizureTransformer**（Wu et al., 2025），它是 2025 年 SzCORE 癫痫检测挑战赛的第一名。移植后与原实现逐点数值一致，最大误差约 1e-7。

LTEEG 完全自包含，不依赖原项目的任何代码和路径，只依赖 numpy、scipy、h5py、pyyaml、torch，可以在 Windows 下运行。

> 本文按"一段 EEG 在框架里怎么流动"的顺序讲思路。逐个配置项、逐个文件的说明见 [docs/reference.md](docs/reference.md)；设计取舍、调研笔记和数值核验记录见 [docs/design_notes.md](docs/design_notes.md)。

---

## 目录

1. [五分钟上手](#五分钟上手)
2. [全景：一段 EEG 从磁盘走到分数](#全景一段-eeg-从磁盘走到分数)
3. [第一步：读取数据](#第一步读取数据)
4. [第二步：切窗口、做采样](#第二步切窗口做采样)
5. [第三步：模型](#第三步模型)
6. [第四步：训练](#第四步训练)
7. [第五步：在 dev 上评估](#第五步在-dev-上评估)
8. [怎么读结果](#怎么读结果)
9. [和原项目相比改进了什么](#和原项目相比改进了什么)
10. [配置怎么改](#配置怎么改)
11. [怎么接入自己的网络和数据集](#怎么接入自己的网络和数据集)
12. [命令速查](#命令速查)
13. [目录结构](#目录结构)
14. [已知限制](#已知限制)
15. [建议的第一批实验](#建议的第一批实验)
16. [参考文献](#参考文献)

---

## 五分钟上手

### 安装

```bash
conda create -n lteeg python=3.10 -y
conda activate lteeg
# 先到 https://pytorch.org 选择与本机 CUDA 匹配的 torch，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu121
cd LTEEG
pip install -e .          # 安装 lteeg 包本身及 numpy / scipy / h5py / pyyaml
pip install -e .[dev]     # 可选：pytest、timescoring、scikit-learn，用来跑测试
python -m pytest -q       # 74 个测试，约半分钟，应全部通过
```

要求 Python ≥ 3.9，torch ≥ 2.0。

### 没有真实数据时，先用合成数据跑通

```bash
python -m lteeg make-synthetic --out D:/tmp/syn
python -m lteeg train --set data.root=D:/tmp/syn data.split_file=D:/tmp/syn/split.json model.name=tcn "model.params={}" train.epochs=6 train.batch_size=16
```

合成数据和真实数据的目录与文件格式完全一样，CPU 上几分钟就能跑完整条流程：训练、长程验证、出报表。

### 在 CHB-MIT 上

```bash
python -m lteeg inspect        # 第一件事：校验全部数据和标注，打印每个病人的统计
python -m lteeg train          # 用默认配置训练 SeizureTransformer baseline
```

默认配置是 `configs/chbmit.yaml`，数据路径默认为 `F:/EEG/CHB-MIT`。换路径可以改 yaml，也可以在命令后加 `--set data.root=D:/你的路径`。

**第一次接触真实数据，一定先跑 `inspect`。**任何标签、导联、采样率上的问题都会在这一步暴露出来，并指出具体文件和行号。

---

## 全景：一段 EEG 从磁盘走到分数

```
  磁盘上的 h5 + 标注文件
          │
          ▼
  ① 读取数据        清点、核对，预处理一次存成可随机读取的缓存
          │
          ▼
  ② 切窗口、采样    在训练记录上铺 60 s 窗口，按配额挑出每个 epoch 用的几千个
          │
          ▼
  ③ 模型            输入 (B, 18, 15360)，输出每个采样点一个 logit
          │
          ▼
  ④ 训练            逐点损失、反向传播；每个 epoch 结束做一次 ⑤
          │
          ▼
  ⑤ 评估            dev 记录整段推理 → 概率曲线 → 事件 → SzCORE 打分 → 逐病人报表
```

贯穿全局的三条原则：

1. **数据集相关的知识只放在配置里。**导联、采样率、文件布局、标注格式、病人划分都写在 `configs/` 里，模型、训练循环、推理、打分都不认识 CHB-MIT。以后换数据集不用改框架代码。
2. **能确定的问题就报错，绝不静默跳过。**标签出错会污染所有结论，所以宁可停下来。
3. **选模型用的量，就是最终报告的量。**训练中的验证和最终评估是同一套整段推理和打分代码。

---

## 第一步：读取数据

### 要解决的问题

- **数据太大**：CHB-MIT 900 多小时、18 导联、256 Hz，存成 float32 约 65 GB，内存放不下。
- **标签在另一个文件里**：信号在 `.h5`，发作时间在 `chbXX_annotations.txt`，两边只靠文件名对应。文件名写错一个字母，这次发作就可能被悄悄丢掉。
- **训练要随机取窗口**：每个 batch 要从几百条记录里随意抽 60 秒的片段。

所以数据读取做三件事：**清点、核对、做成可以随手翻到任意一页的形式。**

### 1. 清点：只看目录，不读信号

把每个 h5 文件想成一本书。清点只看封面和目录：有几个通道、多少个采样点、采样率是多少、属于哪个病人、标注说第几秒到第几秒有发作。

清点的结果是一份很小的**清单**，每条记录一行：

```
chb01/chb01_03   256 Hz   921600 个采样点   发作: [2996 s, 3036 s]
chb01/chb01_04   256 Hz   921600 个采样点   发作: [1467 s, 1494 s]
chb01/chb01_05   256 Hz   921600 个采样点   发作: 无
```

发作只以"第几秒到第几秒"的形式保存，**不存逐点的 0/1 序列**。需要标签时再现场画出来，几乎不占空间。

### 2. 核对：对不上就报错

清点的同时逐项对账，并且把所有问题一次性列出来：

| 检查 | 说明 |
|---|---|
| 标注 ↔ 文件 | 标注里的每个文件名，都必须恰好对应一个 h5 文件 |
| 采样率 | 必须是 256 Hz（可以配置；也支持自动重采样） |
| 导联 | 必须是 18 路。如果 h5 里存了通道名，会按名字核对，顺序不对就自动重排 |
| 发作时间 | 不能超出记录长度，同一条记录里的发作不能重叠 |
| 划分 | chb01 和 chb21 是同一个人，必须在同一侧 |
| **发作总数** | 训练集必须正好 159 次，dev 正好 39 次 |

最后一条是总保险：前面无论漏掉什么问题，只要有一次发作丢了，总数就对不上，程序会报错，并打印每个病人各算出了几次。

### 3. 预处理一次，存成可以随手翻页的缓存

核对通过后，每条记录完整读一遍，做 z-score、0.5–120 Hz 带通、1 Hz 和 60 Hz 陷波，然后存成 `.npy`。这一步只做一次。

之后训练和推理都用 **memmap** 打开缓存：文件用起来像内存里的数组，但只有被取用的那一小段才会真正从硬盘读进来。

和原项目对比：

| | 原项目 | LTEEG |
|---|---|---|
| 存什么 | 先把所有窗口切好，堆成一个大数组 | 存整条记录 |
| 体积 | 窗口重叠 75%，约为原始数据的 4 倍 | 等于原始数据，放在硬盘上 |
| 训练时怎么取 | 只能用最初切好的那一批 | 给出（第几条记录，从哪开始），现场切 |

缓存会记住生成它的预处理参数，以及源文件的大小和修改时间。参数或源文件一变，就自动重建，不会读到过期的结果。

> **为什么整条记录滤波，而不是像原项目那样每个窗口单独滤？**每个窗口单独滤波时，滤波器每次都从零状态启动。其中 1 Hz 陷波的时间常数约 9.55 秒，相当于每个 60 秒窗口的前十几秒都没有被正确滤波。整条记录滤一次就没有这个问题，用的滤波器完全相同。需要完全复现原做法时，用 `configs/experiments/original_faithful.yaml`。

**这一步的产出**：一份核对过的清单，加上一套可以按任意位置取片段的缓存。此时还没有"窗口"或"样本"。

代码：`lteeg/data/store.py`（清点与核对）、`lteeg/data/annotations.py`（标注解析）、`lteeg/data/cache.py`、`lteeg/data/preprocess.py`（预处理与缓存）。

---

## 第二步：切窗口、做采样

### 1. 铺网格，列出所有候选窗口

在每条训练记录上，从第 0 秒开始，每隔 15 秒放一个 60 秒的窗口，相邻窗口重叠 45 秒：

```
记录（1 小时）: |=============================================...|
窗口 1:          [------60s------]
窗口 2:             [------60s------]
窗口 3:                [------60s------]
```

这一步不复制任何数据，只记录每个窗口的地址：（第几条记录，从第几个采样点开始）。训练集约 700 小时，大约能铺出 **17 万个候选窗口**。

### 2. 分类：数一数窗内有多少发作

```
发作:            ░░░░░░░░████████████████████████░░░░░░░░░
background 窗:   [ 全是背景 ]                                ← 0 个发作采样点
boundary 窗:            [ 背景 | 发作 ]                      ← 部分发作，包含起点或终点
full 窗:                       [ 全是发作 ]                  ← 发作长于 60 s 时才会出现
```

发作总时长不到全部数据的 1%。如果直接随机抽窗口，模型几乎只见得到背景，学会"永远输出 0"就能拿到很低的损失。

### 3. 按配额挑选（沿用原项目的思路）

- **boundary 窗全部保留**：它们最稀缺，而且同时包含背景和发作，正好用来教逐点模型"从哪开始、到哪结束"；
- **full 窗**取 boundary 数量的 0.7 倍；
- **background 窗**取 boundary 数量的 3 倍；
- 都不超过实际可用的数量，随机种子固定为 0。

粗略估计：一次约一分钟的发作会产生约 8 个 boundary 窗，159 次发作就是约 1300 个，**每个 epoch 合计约五六千个窗口**，发作采样点的占比从不到 1% 提升到两三成。准确数字以 `inspect` 在你的数据上打印的为准。

### 4. 每个 epoch 怎么用

- 每个 epoch 的顺序只由（种子，epoch 号）决定。换机器、换数据加载进程数、断点续训之后，顺序都完全一样。
- **默认每个 epoch 用同一批窗口**，与原项目一致。但这意味着 17 万个背景窗里，模型始终只见过固定的约 4000 个（2% 左右），而长程测试里的误报恰恰来自没见过的各种背景。可选开关如下：

| 开关 | 作用 |
|---|---|
| `sampling.redraw_every_epoch: true` | 每个 epoch 重新抽一批背景窗，100 个 epoch 下来能见到的背景多得多 |
| `sampling.group_by: patient` | 在每个病人内部各自按配额抽，避免 chb12（40 次发作）、chb15（20 次）这样的病人主导训练 |
| `sampling.jitter_sec` | 把选中的窗口随机前后平移几秒，相当于时间平移增强 |

### 5. 取出一个样本

拿到地址（第 7 条记录，从第 3840 个采样点开始）之后：

1. 从缓存读出这 60 秒，得到信号 `x`，形状 (18, 15360)；
2. 根据清单里的发作秒数，**现场画出**逐点标签 `y`，形状 (15360,)：背景为 0，发作为 1；
3. 以下位置标为 **−1，表示不计入损失**：
   - 窗口超出记录末尾、被补零的部分；
   - （可选，`task.ignore_boundary_sec`）发作起点和终点前后各几秒。专家标注的起止本来就有几秒的不确定性。

> **注意：以上切分和采样只用于训练集。**dev 的记录在评估时会从头到尾完整推理，不做任何挑选，也不做类别平衡（见第五步）。

代码：`lteeg/data/windows.py`（铺网格、分类、配额、每个 epoch 的顺序）、`lteeg/data/datasets.py`（取出样本、画标签）。

---

## 第三步：模型

框架对模型只有一个约定：

```python
Model(in_channels, in_samples, num_outputs, **model.params)
forward(x)              # x: (B, C, T)，已预处理的 float32
  -> logits             # (B, num_outputs, T)，每个采样点一个 logit
```

只要满足这个约定，任何网络都可以接入。几个灵活之处：

- **输出可以比输入短**：例如 patch / token 级的模型只在 1/8 分辨率上输出，框架会自动把 logits 线性插值回逐采样点。内置的 TCN 就是这种情况。
- **可以有多个输出和附加损失**：返回 `ModelOutput(logits, aux_logits=[...], losses={...})`。多阶段或深监督网络把中间输出放进 `aux_logits`；异常检测式的重构损失这类附加项放进 `losses`。
- **可以拿到病人信息**：在类上设 `wants_meta = True`，`forward(x, meta)` 就能收到病人编号、记录编号和窗口位置，可用于病人条件化或测试时自适应。
- **二分类和多分类**：二分类输出 1 个通道，经 sigmoid 得到发作概率；多分类输出 K 个通道，经 softmax 后用 1 − p(背景) 作为"任意发作"概率。所以评估不需要为多分类单独改代码。

### 内置的 SeizureTransformer

这是一个 U 形网络：

- **编码器**：5 级卷积，每级长度减半，15360 → 480；
- **瓶颈**：7 个残差卷积块，加上 8 层 Transformer，对 480 个 token 做全局注意力；
- **解码器**：5 级上采样，加上跳跃连接，恢复到 15360；
- **输出**：每个采样点一个 logit。

移植时做了这些改动，在原配置下计算结果都不变：

- 输出 logits 而不是概率，以便使用数值稳定的 `BCEWithLogits`，也才能开混合精度；
- 删掉了一层从未参与计算、却带着 315 万参数的冗余 Transformer 层（41.0M → 37.85M）；
- 支持任意长度 ≥ 32 的输入，改窗长做实验不用改模型；
- 参数名保持不变，原作者的权重可以直接加载。

代码：`lteeg/models/`。

---

## 第四步：训练

一次训练迭代：

```
取一个 batch（后台线程已经提前读好）
  → 搬到 GPU，可选数据增强
  → 前向（可选混合精度），logits 对齐到逐点
  → 逐点损失（−1 的位置不计入）
  → 反向（可累积多个 batch）
  → 可选梯度裁剪，记录梯度范数
  → 更新参数，学习率调度器前进一步，可选 EMA 更新
```

每个 epoch 结束：在 dev 上做完整评估（第五步）→ 指标提升就保存 `best.pt` → 每个 epoch 都保存 `last.pt`，可以断点续训。

**默认超参数与原项目一致**：RAdam，学习率 1e-4，weight decay 2e-5，100 个 epoch，batch 86，常数学习率，不裁剪梯度，不开混合精度，按 dev 上的 event F1 选模型。这是为了先得到一个可复现、可比较的 baseline。

可选的工程能力（默认关闭）：

| 能力 | 配置 |
|---|---|
| 混合精度 bf16 / fp16 | `train.amp` |
| 梯度累积 / 梯度裁剪 | `train.accum_steps` / `train.grad_clip` |
| EMA 权重 | `train.ema_decay` |
| 学习率预热 + 余弦 / 阶梯衰减 | `scheduler.*` |
| 早停 | `train.early_stopping`、`train.patience` |
| 数据增强（幅度缩放、符号翻转、左右半球导联互换、噪声、通道丢弃、时间遮挡） | `train.augment` |
| 其他损失：focal、dice、tmse 平滑损失（抑制会变成误报的输出抖动） | `loss.terms` |

另外几项是默认就开着的保险：

- 损失出现 NaN 时自动跳过该 batch，连续出现太多次就报错；
- 检查点原子写入；
- `train --resume 运行目录` 可以完整续训；
- Windows 下 `num_workers=0` 时用后台线程预取数据，不让 GPU 空等。

代码：`lteeg/engine/`、`lteeg/losses.py`。

---

## 第五步：在 dev 上评估

一句话：**像临床一样，把病人的整段记录从头到尾交给模型，再看它报出的发作事件和医生的标注对不对得上。**

### 1. 整段推理：把记录变成一条概率曲线

用 60 秒的窗口把每条 dev 记录从头铺到尾（默认不重叠），把每个窗口输出的逐点概率拼回去：

```
记录:   |--------------------------------------------- 1 小时 ---------|
窗口:   [60s][60s][60s][60s] ...                              [60s]
输出:   p(t) ________/‾‾‾‾‾‾‾\____________________/\__________________
                    ↑ 一次发作                    ↑ 一个可疑尖峰
```

最后一个窗口对齐到记录末尾，不补零，因为补零的平直信号是模型从没见过的输入。这里**不做任何挑选和平衡**：dev 里发作不到 1%，模型必须在这 99% 的背景里不乱报。这才是真实场景的难度。

### 2. 后处理：把曲线变成"发作事件"

1. 阈值：概率 > 0.8 记为发作；
2. 形态学开、闭运算，去掉零星的碎点（默认核长 5 个采样点，约 20 ms，作用很小）；
3. 删掉短于 2 秒的片段。

最终得到的是一个事件列表：[12:03 → 12:51]，[47:10 → 47:40]，……

### 3. SzCORE 打分

**event 级**（最重要，挑战赛就按它排名）按"次"计数，规则贴近临床直觉：

```
真实发作:              [====]
放宽后的范围:      [----====--------]        往前放宽 30 s，往后放宽 60 s
预测 A:              [==]                    → 有重叠：检出 ✓
预测 B:                                  [==]  → 在范围外：这次发作算漏检，B 算一次误报
```

- 预测与放宽后的范围有重叠：**检出（TP）**；
- 一个预测都没碰到：**漏检（FN）**；
- 预测不挨着任何已检出的发作：**误报（FP）**；
- 间隔不到 90 秒的事件合并为一次，所以碎成几段的预测只算一次；
- 长于 5 分钟的事件切成几段，所以长发作只报开头会被扣分。

由此得到四个指标：

| 指标 | 含义 |
|---|---|
| sensitivity | 检出的发作 / 全部发作：漏没漏 |
| precision | 对的报警 / 全部报警：报得准不准 |
| F1 | 两者的综合，是选模型的主指标 |
| 每 24 小时误报数 | 临床最在意的数字 |

**sample 级**（辅助指标）按秒逐个比对，注意 SzCORE 的"sample"是 **1 秒**，不是 1/256 秒。另外还会计算**不依赖阈值的 AUPRC**，直接衡量概率曲线本身的质量。

打分逻辑是自己实现的，在 300 组随机样例上与官方 `timescoring` 包的计数完全一致。

### 4. 汇总：为什么要按病人看

- **pooled**：所有记录的计数直接相加再算指标。记录长、发作多的病人权重大。
- **macro**：先算每个病人自己的指标，再对病人求平均，同时给出**最差病人**。

举个编造的例子：chb24 有 16 次发作、表现很好，chb14 几乎全错。pooled 会被 chb24 拉高，chb14 的失败被稀释；按病人看就一目了然。"总有一两个病人效果奇差"这个现象，在报表里体现为最差病人这一项。

### 关于 dev 的数字

dev 同时承担两件事：**选模型**（往往还顺带选阈值），以及**报告结果**。在同一份数据上挑出最好的再报告，数字一定偏乐观。开发阶段这样迭代没有问题；最终写论文的数字，建议用**病人级交叉验证**（split 文件支持 `folds`，配合 `--set data.fold=k`），或**留出一组只测一次的 test 病人**。

代码：`lteeg/inference/`（整段推理、后处理）、`lteeg/evaluation/`（打分、汇总、报表、阈值扫描、多实验对比）。

---

## 怎么读结果

每次训练会生成一个运行目录：

```
runs/chbmit/<实验名>_<时间戳>/
├── config.yaml              本次运行的完整配置（路径都已转为绝对路径）
├── env.json                 Python / torch / CUDA / GPU / git 版本，以及命令行
├── data_summary.json        每个病人的记录数、时长、发作数、发作占比
├── sampling_summary.json    各类窗口可用多少、选了多少
├── train.log / train_log.csv   每个 epoch 的损失、梯度范数、学习率、全部验证指标
├── checkpoints/last.pt      完整训练状态（续训用）
├── checkpoints/best.pt      最佳模型权重
└── eval/dev_best/
    ├── summary.md           逐病人表 + POOLED + MACRO，先看这个
    ├── patients.csv         每个病人的指标
    ├── records.csv          每条记录的指标
    ├── events.csv           每次真实发作：是否检出、提前或延迟几秒、覆盖了多少
    ├── false_alarms.csv     每次误报：在哪、持续多久
    ├── probs/               每条记录的逐点概率曲线（float16）
    └── sweep.csv            不同阈值下的结果
```

建议的阅读顺序：

1. `summary.md`：先看整体和每个病人；
2. 找出最差的病人；
3. 在 `events.csv` 里看它漏了哪些发作；
4. 在 `false_alarms.csv` 里看误报集中在哪里；
5. 结合 `probs/` 里的概率曲线，看模型在那些时刻输出了什么。

换阈值、换最短事件长度不需要重新跑模型：

```bash
python -m lteeg sweep runs/chbmit/<run>/eval/dev_best --thresholds 0.5 0.6 0.7 0.8 0.9 --min-durations 2 5 10
```

多个实验并排比较：

```bash
python -m lteeg compare runs/chbmit
```

---

## 和原项目相比改进了什么

默认情况下，**模型结构和训练超参数都与原项目保持一致**。改进集中在数据、正确性、评估和工程四个层面；凡是可能改变结果的改动，都留了开关可以退回原行为。

| 方面 | 原项目 | LTEEG |
|---|---|---|
| 数据组织 | 所有窗口提前切好堆进内存，约为原始数据的 4 倍，CHB-MIT 放不下 | 整条记录缓存到硬盘，按需现场取窗口；因此才能每个 epoch 换背景、随机平移、按病人平衡 |
| 出错处理 | 读文件出错就跳过；比整窗短的记录也跳过 | 一律报错；短记录补零后照常评估 |
| 标签核对 | 无 | 标注与文件一一对应、导联、采样率、发作越界与重叠、发作总数 159/39 |
| 秒 → 采样点 | 训练用截断，评分用四舍五入，可能差一个点 | 统一用四舍五入 |
| 平坦通道 | z-score 除以 0，产生 NaN | 置零并记录 |
| 滤波 | 每个窗口单独滤，每个窗口开头都有约 10 秒的滤波瞬态 | 整条记录滤一次；`original_faithful.yaml` 可复现原做法 |
| 推理末尾 | 补零 | 对齐记录末尾，不补零；`inference.tail: pad` 可复现原做法 |
| 模型 | 输出概率；带一层 315 万参数的冗余层；只能输入固定长度 | 输出 logits；去掉冗余层；任意长度；计算结果逐点一致 |
| 评估输出 | 只打印 pooled 的 8 个数字 | 逐病人 / 记录 / 发作 / 误报的报表，宏平均与最差病人，AUPRC，概率存档，阈值扫描，多实验对比 |
| 打分 | 依赖外部本地安装的 timescoring | 自己实现，并与官方结果核对一致 |
| 训练工程 | 基础训练循环 | 混合精度、梯度累积与裁剪、EMA、学习率调度、早停、NaN 保护、断点续训、线程预取（默认关闭或不影响结果） |
| 配置 | 散落在各脚本里的 argparse，路径写死 | 一份 yaml，带类型检查，写错键名会提示正确写法；数据集相关内容全在配置里 |

---

## 配置怎么改

所有设置都在 `configs/chbmit.yaml` 里，每一项都有注释。有三种修改方式：

**1. 命令行临时改**：

```bash
python -m lteeg train --set train.lr=3e-4 train.amp=bf16 sampling.redraw_every_epoch=true
```

**2. 写一个实验配置，只写和默认不同的地方**：

```yaml
# configs/experiments/my_exp.yaml
_base_: ../chbmit.yaml
experiment: {name: st_redraw}
sampling: {redraw_every_epoch: true}
```

```bash
python -m lteeg train --config configs/experiments/my_exp.yaml
```

**3. 直接改 `chbmit.yaml`**（不推荐，基线会跟着变）。

配置是严格的：写错键名（例如 `train.batchsize`）会直接报错，并提示 "did you mean 'batch_size'"。

Windows 路径请写成 `F:/EEG/CHB-MIT`，或者用单引号括起来写成 `'F:\EEG\CHB-MIT'`。

已经准备好的实验配置：

| 文件 | 用途 |
|---|---|
| `original_faithful.yaml` | 完全复现原项目的逐窗滤波与末窗补零，用来量化框架默认值带来的差异 |
| `tcn_debug.yaml` | 小模型，在真实数据上快速打通流程 |
| `st_robust_training.yaml` | 打开常用的稳定化手段（每轮重抽背景、AdamW + 余弦、bf16、梯度裁剪、EMA、增强），作为对照实验的起点，**尚未在真实数据上验证** |

---

## 怎么接入自己的网络和数据集

### 新网络

```python
# my_models/unet.py
import torch.nn as nn

class MyUNet(nn.Module):
    def __init__(self, in_channels, in_samples, num_outputs, depth=4):
        super().__init__()
        ...

    def forward(self, x):       # x: (B, C, T)
        ...
        return logits           # (B, num_outputs, T)
```

不需要修改框架，在配置里写 `模块路径:类名` 即可：

```bash
# 先自检：检查输出形状、测速度和显存，并在一个 batch 上过拟合，确认损失能降下去
python -m lteeg check-model --set "model.name=my_models.unet:MyUNet" "model.params={depth: 5}" --overfit-steps 50
# 再训练
python -m lteeg train --set "model.name=my_models.unet:MyUNet" "model.params={depth: 5}"
```

新的损失函数、预处理算子、数据增强也可以用同样的方式接入，模板见 [docs/reference.md](docs/reference.md)。

### 新数据集

只需要新的 yaml 和 split 文件，框架代码不用改：

```yaml
# configs/siena.yaml
_base_: chbmit.yaml
experiment: {name: st_siena, output_dir: runs/siena}
data:
  root: D:/EEG/Siena
  split_file: siena_split.json
  channels: [FP1-F7, F7-T7, ...]     # 新数据集的导联与顺序
  fs: 512.0                          # 文件里的采样率
  resample_to: 256.0                 # 统一到 256 Hz，标签按秒换算，自动对齐
  cache_dir: cache/siena
```

如果原始数据不是 h5，先转换成同样的布局：每个病人一个目录，目录里是 `signals (通道, 采样点)` 加 `fs` 属性的 h5 文件，以及一个带 `[seizures]` 段的标注文件。

---

## 命令速查

| 命令 | 作用 |
|---|---|
| `python -m lteeg inspect` | 校验数据与标注，打印每个病人的统计和可用训练窗数 |
| `python -m lteeg prepare --jobs 4` | 多进程预先构建缓存（可选，`train` 会自动补齐） |
| `python -m lteeg train` | 训练；每个 epoch 做长程验证；结束后用最佳模型正式评估 |
| `python -m lteeg train --resume <运行目录>` | 断点续训 |
| `python -m lteeg train --dry-run` | 准备好数据、缓存和模型后退出，用来检查配置 |
| `python -m lteeg evaluate <best.pt> --splits dev` | 单独评估某个检查点 |
| `python -m lteeg sweep <eval目录>` | 在存好的概率上换阈值重新打分 |
| `python -m lteeg compare runs/chbmit` | 多个实验并排比较 |
| `python -m lteeg predict <best.pt> <h5 文件或目录> --out preds/` | 对新记录输出发作起止时间（TSV） |
| `python -m lteeg check-model` | 新网络接入自检 |
| `python -m lteeg make-synthetic --out <目录>` | 生成同格式的合成数据 |

---

## 目录结构

```
LTEEG/
├── README.md                   本文件
├── configs/
│   ├── chbmit.yaml             全部默认配置（与代码中的默认值逐项一致，有测试保证）
│   ├── chbmit_split.json       病人划分：train 18 人 / dev 6 人，chb01 与 chb21 同组，发作数 159 / 39
│   └── experiments/            继承默认配置、只写差异的实验配置
├── lteeg/
│   ├── config.py               配置的加载与校验
│   ├── data/                   第一、二步：读取、核对、缓存、窗口、采样、样本
│   ├── models/                 第三步：模型约定、SeizureTransformer、TCN
│   ├── losses.py               逐点损失
│   ├── engine/                 第四步：训练循环、优化器、检查点
│   ├── inference/              第五步：整段推理、后处理
│   ├── evaluation/             第五步：SzCORE 打分、汇总、报表、阈值扫描、对比
│   └── cli.py, workflows.py    命令行入口
├── docs/
│   ├── reference.md            详细参考手册：所有配置项、每个模块、扩展模板
│   └── design_notes.md         设计取舍、调研笔记、数值核验记录
└── tests/                      74 个测试
```

---

## 已知限制

- **尚未在真实 CHB-MIT 上运行过。**开发环境没有真实数据和 GPU，完整流程只在同格式的合成数据上测试过。请先用 `inspect` 检查你的数据，再跑训练。
- **CUDA 相关功能没有实际运行过**，包括 fp16 混合精度、`torch.compile`、`pin_memory`。bf16 只在 CPU 上测试过。
- **Windows 多进程数据加载是在 Linux 上模拟验证的**，没有在真正的 Windows 机器上跑过。
- **显存**：60 秒窗口、batch 86 的 SeizureTransformer 对显存要求较高，具体用量未实测。可以先跑 `check-model --batch-size 86` 看一下；不够时减小 `train.batch_size`，并用 `train.accum_steps` 补回等效的 batch。
- **缓存体积**：约 65 GB（float32）。设置 `data.cache_dtype: float16` 可降到约 33 GB，精度损失可以忽略。
- **加载原作者的比赛权重时注意导联顺序**：那份权重是在 TUSZ / Siena 上训练的，导联顺序可能与这里的 CHB-MIT 不同，需要先核对并重排。

---

## 建议的第一批实验

1. **跑通 baseline**：默认配置，看 dev 上 pooled 和 macro 的 event F1、每 24 小时误报数，以及最差病人是谁。
2. **量化框架默认值本身的影响**：用 `original_faithful.yaml` 再跑一次，和第 1 个实验对比。
3. **背景覆盖**：打开 `sampling.redraw_every_epoch`，看误报是否下降。
4. **病人平衡**：打开 `sampling.group_by: patient`，看最差病人是否改善。
5. **输出平滑**：在损失里加上 `tmse`，看碎片化误报是否减少。
6. **阈值**：对上面每个实验用 `sweep` 选工作点，再用 `compare` 汇总。注意，在 dev 上选出的阈值迁移到 test 时才是公平的数字。

---

## 参考文献

- K. Wu, Z. Zhao, B. Yener. *Large EEG-U-Transformer for Time-Step Level Detection Without Pre-Training* (SeizureTransformer). arXiv:2504.00336, 2025.
- J. Dan et al. *SzCORE: Seizure Community Open-Source Research Evaluation framework for the validation of EEG-based automated seizure detection algorithms*. Epilepsia, 2024.
- J. Dan et al. *SzCORE as a benchmark: report from the seizure detection challenge at the 2025 AI in Epilepsy and Neurological Disorders Conference*. arXiv:2505.18191.
- A. Shoeb. *Application of machine learning to epileptic seizure onset detection and treatment*. PhD thesis, MIT, 2009（CHB-MIT 数据集）。
- Y. A. Farha, J. Gall. *MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation*. CVPR 2019（tmse 平滑损失）。
- M. Perslev et al. *U-Sleep: resilient high-frequency sleep staging*. npj Digital Medicine, 2021.
- S. Kim et al. *Towards a Rigorous Evaluation of Time-series Anomaly Detection*. AAAI 2022.
