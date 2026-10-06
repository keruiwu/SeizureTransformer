# LTEEG：长程脑电逐点癫痫检测框架

LTEEG 是一个面向长程脑电（long-term EEG）的**逐采样点（point-level）癫痫检测**研究框架。
样本仍是窗口，但模型对窗口内**每一个采样点**输出 logit（二分类或多分类），标签形式与时间序列异常检测的逐点标签、临床上标注发作起止时间的需求一致。
训练之后，模型直接在**完整的长程记录**上连续推理，拼接出逐点概率，再用 SzCORE 体系做 sample 级与 event 级打分，并逐病人呈现结果。

- 内置 baseline：**SeizureTransformer**（Wu et al., 2025，2025 SzCORE 癫痫检测挑战赛第一），已移植并与原实现做过逐元素数值核对（最大误差 1.2e-7）。
- 完全自包含：不依赖原项目任何代码和路径；依赖只有 numpy / scipy / h5py / pyyaml / torch。
- 数据集相关的一切（导联、采样率、文件布局、标注格式、病人划分、窗长）都在配置里；模型、训练循环、长程推理和事件评分是通用的。
- Windows 可用：默认 `num_workers=0` 加后台线程预取，多进程 worker 也已按 spawn 方式验证。

> 设计取舍、调研笔记和数值核验记录见 [docs/design_notes.md](docs/design_notes.md)。

---

## 目录

1. [目录结构](#目录结构)
2. [安装](#安装)
3. [快速开始](#快速开始)
4. [数据约定与校验](#数据约定与校验)
5. [配置系统](#配置系统)
6. [训练流程](#训练流程)
7. [长程推理与评估](#长程推理与评估)
8. [接入新模型](#接入新模型)
9. [扩展：新数据集、多分类、新组件](#扩展新数据集多分类新组件)
10. [与原项目的差异](#与原项目的差异)
11. [面向后续研究方向的预留](#面向后续研究方向的预留)
12. [已知限制](#已知限制)
13. [参考文献](#参考文献)

---

## 目录结构

```
LTEEG/
├── configs/
│   ├── chbmit.yaml               # 全部默认值；与 lteeg/config.py 的 dataclass 默认值逐项一致（有测试保证）
│   ├── chbmit_split.json         # 病人级划分、同一受试者分组、预期发作数
│   └── experiments/              # 用 _base_ 继承 chbmit.yaml、只写差异的实验配置
│       ├── original_faithful.yaml    # 逐窗滤波 + 末窗补零，完全复现原项目流程
│       ├── st_robust_training.yaml   # 打开常用稳定化手段的对照实验（未在真实数据上验证）
│       └── tcn_debug.yaml            # 小模型，在真实数据上快速打通流程
├── lteeg/
│   ├── config.py                 # 带类型的配置：严格校验未知键、类型转换、跨字段检查
│   ├── registry.py               # 模型/损失/预处理/增强的注册表（也支持 "pkg.module:Class" 动态导入）
│   ├── task.py                   # 二分类/多分类：logits -> 概率 -> “任意发作”概率
│   ├── cli.py, workflows.py      # 命令行与各命令的实现
│   ├── synthetic.py              # 与真实数据同格式的合成数据（测试/冒烟用）
│   ├── data/
│   │   ├── annotations.py        # 标注解析与校验，秒 -> 采样点（与评分器同一舍入规则）
│   │   ├── store.py              # 读取 h5、校验导联/采样率/标注对应关系，产出 Recording 列表
│   │   ├── split.py, loading.py  # 病人级划分（支持交叉验证 folds）与按 split 加载
│   │   ├── preprocess.py         # 预处理算子（z-score、带通、陷波……），相邻线性滤波融合为一个 SOS 级联
│   │   ├── cache.py              # 预处理结果的磁盘缓存（memmap 读取，内存占用与数据量无关）
│   │   ├── windows.py            # 窗口索引、background/boundary/full 分类、采样器
│   │   ├── datasets.py           # 训练窗 Dataset（逐点标签，-1 为忽略）
│   │   └── augment.py            # GPU 上的批量增强
│   ├── models/                   # 模型契约 + SeizureTransformer + 轻量 TCN
│   ├── losses.py                 # bce / ce / focal / dice / tmse，可加权组合
│   ├── engine/                   # Trainer、优化器与调度、EMA、检查点、预取
│   ├── inference/                # 整段记录连续推理与拼接、后处理
│   └── evaluation/               # SzCORE 评分、AUROC/AUPRC、报表、阈值扫描、多实验对比
├── docs/design_notes.md
└── tests/                        # 73 个测试（约 25 秒，CPU 即可）
```

## 安装

```bash
conda create -n lteeg python=3.10 -y
conda activate lteeg
# 先按 https://pytorch.org 选择与本机 CUDA 匹配的 torch，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu121
cd LTEEG
pip install -e .          # 安装 lteeg 及 numpy/scipy/h5py/pyyaml
pip install -e .[dev]     # 可选：pytest、timescoring、scikit-learn（用于交叉核对评分实现）
python -m pytest -q       # 全部应通过
```

要求 Python ≥ 3.9、torch ≥ 2.0。所有命令都可以写成 `python -m lteeg <命令>`，安装后也可以直接用 `lteeg <命令>`。

## 快速开始

### 0. 没有真实数据时：合成数据冒烟测试（CPU，几分钟）

```bash
python -m lteeg make-synthetic --out D:/tmp/syn
python -m lteeg train --set data.root=D:/tmp/syn data.split_file=D:/tmp/syn/split.json ^
    model.name=tcn "model.params={}" train.epochs=6 train.batch_size=16
```

（PowerShell 里续行符是反引号 `` ` ``，cmd 里是 `^`，也可以写成一行。）合成数据和真实数据格式完全相同：每个病人一个目录、h5 文件和 `[seizures]` 标注文件。

### 1. CHB-MIT

```bash
python -m lteeg inspect                 # 校验全部数据与标注，打印逐病人/逐 split 统计和训练窗数量
python -m lteeg prepare --jobs 4        # 一次性构建预处理缓存（可选，train 会自动补齐）
python -m lteeg train                   # SeizureTransformer baseline，默认配置即 configs/chbmit.yaml
python -m lteeg train --config configs/experiments/original_faithful.yaml
python -m lteeg evaluate runs/chbmit/<run>/checkpoints/best.pt --splits dev
python -m lteeg sweep runs/chbmit/<run>/eval/dev_best --thresholds 0.5 0.6 0.7 0.8 0.9 --min-durations 2 5 10
python -m lteeg compare runs/chbmit     # 多个实验并排比较（pooled、病人宏平均、最差病人、逐病人）
python -m lteeg predict runs/chbmit/<run>/checkpoints/best.pt F:/EEG/new_patient --out preds/
python -m lteeg check-model --set model.name=my_net --overfit-steps 50   # 新网络接入自检
```

不带 `--config` 时使用 `configs/chbmit.yaml`。任何配置项都能用 `--set 键=值` 覆盖，例如
`--set train.lr=3e-4 train.amp=bf16 sampling.redraw_every_epoch=true`。

**第一次在真实数据上运行，请先跑 `inspect`。** 它会检查下文列出的全部约定并打印每个病人的记录数、时长、发作次数、发作时长占比和可用训练窗数量；任何不一致都会以错误退出并指出具体文件和行号。

### 2. 运行产物

```
runs/chbmit/<name>_<时间戳>/
├── config.yaml            # 本次运行的完整配置（路径已转为绝对路径）
├── env.json               # Python/torch/CUDA/GPU/git 版本、命令行
├── split.json, data_summary.json, sampling_summary.json
├── train.log, train_log.csv   # 每个 epoch 的训练损失、梯度范数、学习率、全部验证指标
├── best.json, result.json
├── checkpoints/last.pt    # 完整训练状态（断点续训用）
├── checkpoints/best.pt    # 仅权重（+EMA）与配置
└── eval/dev_best/
    ├── summary.md / summary.json   # 逐病人表 + POOLED + MACRO
    ├── patients.csv, records.csv   # 病人级、记录级指标
    ├── events.csv                  # 每个参考发作：是否检出、检出延迟、覆盖率
    ├── false_alarms.csv            # 每个误报事件的起止与时长
    ├── probs/<病人>/<记录>.npy      # 逐点概率（float16），供 sweep/可视化
    ├── manifest.json               # 参考事件与记录元数据（sweep 不需要原始数据）
    └── sweep.csv                   # 阈值扫描
```

## 数据约定与校验

默认布局（全部可在 `data.*` 中配置）：

```
<data.root>/<病人>/<记录>.h5             数据集 signals: (通道数, 采样点数)；文件属性 fs: 采样率
<data.root>/<病人>/<病人>_annotations.txt
```

标注文件中 `[seizures]` 段每行是制表符分隔的 `文件名  起始秒  结束秒`，表头行以 `file` 开头。其他段、`#` 注释、空行会被忽略；UTF-8 BOM 和 Windows 换行都能处理。

`H5Store` 在扫描阶段只读元数据，逐项检查；**所有问题汇总后一次性报错，绝不静默跳过**：

| 检查 | 失败时 |
|---|---|
| split 中的病人目录存在；同一 split 内无重复；病人不跨 split；`groups` 中同一受试者（chb01/chb21）不跨 split | 报错 |
| 每个病人都有标注文件、存在 `[seizures]` 段，每行能解析出数值 | 报错并给出行号 |
| 每条标注对应**恰好一个** h5 文件（`annotation_match: exact` 要求完全同名；若只是扩展名不同会提示改用 `stem`） | 报错 |
| `signals` 存在、二维、数值类型；形状像是转置了 `(采样点, 通道)` 会专门提示 | 报错 |
| 采样率属性存在且等于 `data.fs`（除非设置了 `data.resample_to`） | 报错 |
| 通道数等于 `data.channels`；若文件里存有通道名属性（`data.channel_attr`，或自动识别 `channels`/`ch_names` 等），按名字匹配并**自动重排**为配置顺序，缺通道或重名报错 | 报错 |
| 没有通道名属性时只能检查数量、无法检查顺序 | 明确告警 |
| 发作 `start < end`、`start ≥ 0`、在记录时长之内（允许末尾超出 `end_tolerance_sec`，裁剪并告警） | 报错 |
| 同一记录内发作重叠（`overlapping_events: merge` 时同类合并并告警） | 报错 |
| 各 split 的发作总数与 split 文件中的 `expected_seizures`（train 159 / dev 39）一致，不一致时打印逐病人计数 | 报错 |
| 读取信号时出现 NaN/Inf | 报错并列出通道 |
| z-score 时遇到平坦通道（标准差≈0）：置零而不是产生 NaN，并记录在缓存元数据里 | 记录 |
| 缓存与源文件不一致（大小或修改时间变化、预处理配置变化） | 自动重建 |

秒到采样点的换算统一使用 `round(t·fs)`（与 timescoring 相同），训练标签与评估参考逐采样点一致。CHB-MIT 的病人划分、`chb01/chb21` 同一受试者分组和 159/39 的发作数都写在 `configs/chbmit_split.json` 中，并在加载时强制校验。

## 配置系统

- **单一事实来源**：`lteeg/config.py` 中的 dataclass 默认值；`configs/chbmit.yaml` 逐项列出同样的值并附注释，`tests/test_config.py` 保证二者不会走样。
- **严格**：拼错的键会报错并给出“did you mean”；值按声明类型转换（`1e-4` 这类 PyYAML 不认的浮点写法也能正确解析）；跨字段约束（窗长是否为整数个采样点、步长不超过窗长、滤波频率低于 Nyquist、对称导联对是否存在等）在加载时检查。模型、损失、预处理、增强的参数名也会核对，拼错不会被静默忽略。
- **继承**：实验配置写 `_base_: ../chbmit.yaml`，只列出差异。`model.params`、`label_map` 整体替换，其余字典逐层合并，列表整体替换。
- **命令行覆盖**：`--set a.b=值`，值按 YAML 解析，例如 `--set "train.augment=[{name: sign_flip}]" inference.hop_sec=null`。
- **路径规则**：`data.split_file` 相对于定义它的 yaml 文件；`data.root`、`data.cache_dir`、`experiment.output_dir` 相对于当前工作目录。训练开始时都会转成绝对路径写进运行目录的 `config.yaml`，因此之后在任何目录下都能评估。
- **Windows 路径**：写 `F:/EEG/CHB-MIT`，或用单引号 `'F:\EEG\CHB-MIT'`；不要用双引号包反斜杠（YAML 会当作转义）。

## 训练流程

### 窗口与采样

训练窗以 `windows.train_stride_sec` 为步长铺在每条训练记录上，按窗内发作采样点数分为三类：**background**（0 个）、**full**（全部）、**boundary**（部分，含起止点）。默认 `balanced` 采样器沿用原项目 `get_dataset.py` 的思路：boundary 全部保留，full 取 boundary 数的 0.7 倍，background 取 3 倍，均不超过实际可用数量，随机种子为 0。`inspect` 和训练日志会打印每类可用/选中数量以及选中集合里发作采样点的比例。

可选项（默认关闭，保持 baseline 可复现）：

| 选项 | 作用 |
|---|---|
| `sampling.redraw_every_epoch: true` | 每个 epoch 重新抽 full/background。CHB-MIT 训练集背景窗约十几万个，固定子集只用到其中几千个；重抽能让模型见到多得多的背景，通常有助于降低长程误报 |
| `sampling.group_by: patient` | 在每个病人内部按比例抽样，避免发作多的病人（如 chb12、chb15）主导训练 |
| `sampling.jitter_sec` | 对选中的窗口随机平移，作时间增强 |
| `sampling.name: all` | 使用全部窗口，交给损失函数处理不平衡 |

窗口选择只取决于（种子, epoch）和按自然序排列的记录列表，与机器和 DataLoader worker 数无关。采样器在主进程里运行，每个 epoch 的顺序也就确定下来，`persistent_workers` 下每 epoch 重抽同样生效。

### 标签与损失

- 逐点标签为 int64，0 是背景，`-1` 表示忽略（补零区域、以及 `task.ignore_boundary_sec > 0` 时发作起止点附近的不确定区间）。
- `task.mode: binary`：模型输出 1 个通道，sigmoid；`multiclass`：K 个通道，softmax，`1 - p(背景)` 作为“任意发作”概率用于事件评估。
- 损失由若干项加权组合（`loss.terms`），所有项都会排除忽略位置：

| 名称 | 说明 |
|---|---|
| `bce` | 带 logits 的二元交叉熵（原项目用 sigmoid 后的 BCE，数值上不稳定且不能用于混合精度）；`pos_weight`、`label_smoothing` |
| `ce` | 多分类交叉熵，`class_weights` |
| `focal` | Focal loss（二分类/多分类） |
| `dice` | 在整个 batch 上计算的 soft Dice（不含发作的窗口也有定义） |
| `tmse` | MS-TCN 的截断 MSE 平滑损失：惩罚相邻采样点间对数概率的跳变，抑制阈值化后变成误报事件的“抖动” |

### 优化与工程特性

| 功能 | 配置 | 默认 |
|---|---|---|
| 优化器 Adam/AdamW/RAdam/SGD，可对 norm/bias 免衰减 | `optimizer.*` | RAdam，lr 1e-4，wd 2e-5（与原项目一致） |
| 学习率：线性 warmup + 常数/余弦/阶梯，按 step 更新 | `scheduler.*` | 常数（与原项目一致） |
| 混合精度 bf16 / fp16（fp16 自动配 GradScaler） | `train.amp`, `inference.amp` | 关闭 |
| 梯度累积、梯度裁剪 | `train.accum_steps`, `train.grad_clip` | 1、关闭 |
| EMA 权重（验证和评估自动用 EMA） | `train.ema_decay` | 关闭 |
| 非有限损失保护：跳过该 batch，连续出现则报错并给出建议 | `train.max_nonfinite_steps` | 20 |
| 每个优化 step 记录梯度范数，epoch 汇总均值和最大值 | — | 开启 |
| 早停 | `train.early_stopping`, `train.patience` | 关闭，patience 12 |
| `torch.compile`（不可用时自动回退） | `train.compile` | 关闭 |
| 原子写入的检查点、完整断点续训（模型/优化器/调度器/scaler/EMA/RNG） | `train --resume <运行目录>` | — |
| 确定性模式 | `experiment.deterministic` | 关闭 |

训练 batch 默认 86、100 个 epoch、`num_workers=0`。最后一个不足一个 batch 的残批会被丢弃（避免出现单样本 BatchNorm）。60 秒窗、batch 86 的 SeizureTransformer 对显存要求较高（未在本环境实测）；显存不足时可减小 `train.batch_size` 并用 `train.accum_steps` 保持等效 batch（BatchNorm 统计量会随 micro-batch 变小而变噪），或开启 `train.amp: bf16`。先用 `check-model --batch-size 86` 在目标 GPU 上看峰值显存。

### 验证就是长程评估

每个验证 epoch 都对**每条完整 dev 记录**做连续推理、后处理和 SzCORE 打分，与最终评估完全相同，因此用来选模型的量就是最终报告的量。日志会打印逐病人的验证表格，`train_log.csv` 记录每个 epoch 的全部指标。选模指标由 `train.selection_metric` 指定，默认 `event_f1_pooled`（与原训练脚本一致），可改为下文列出的任一指标，例如阈值无关的 `auprc_pooled` 或 `nll_pooled`（配合 `selection_mode: min`）。

## 长程推理与评估

### 连续推理

一条 N 个采样点的记录被训练窗长 W 的窗口覆盖，窗口间隔 `inference.hop_sec`（默认等于窗长，即原项目的不重叠推理）。重叠时各窗口的概率按 `mean` 或 `hann`（窗口中心权重大，边缘上下文不足处权重小）加权平均。记录末尾默认 `tail: align`，即追加一个恰好结束在 N 的窗口；`pad` 则像原项目那样补零，而补零对模型而言是分布外输入。输出逐点概率，内存只与记录长度成正比。

### 后处理

阈值（严格大于，默认 0.8）→ 形态学开运算再闭运算（核长 5 个采样点）→ 去掉短于 2 秒的事件 →（可选）合并间隔小于 `merge_gap_sec` 的事件。一维形态学以游程方式精确实现：开运算去掉短于 k 的阳性游程，闭运算填补短于 k 的内部空隙；在内部与 scipy 完全一致，同时避免了 scipy `border_value=0` 在记录首尾“啃掉”事件的边界效应。

### SzCORE 评分（自研实现，与 timescoring 逐项一致）

- **sample 级**：参考与预测都栅格化到 1 Hz（`scoring.sample_fs`），逐秒比较。注意 SzCORE 的“sample”是 1 秒粒度；如需原始采样率粒度可设 `scoring.sample_fs=256`。
- **event 级**：间隔 < 90 s 的事件合并，长于 300 s 的事件切分；参考事件向前扩 30 s、向后扩 60 s，与任一预测有重叠即算检出；没有和任何“已检出参考”的扩展区间重叠的预测事件算一次误报。全部在 10 Hz 网格上进行。
- 指标：sensitivity、precision、F1、每 24 小时误报数（`fp_per_day`），另有每个检出发作的延迟（`latency_sec`，负数表示提前）和覆盖率。
- 实现与官方 `timescoring` 包在 300 组随机样例上计数完全一致（见 `tests/test_scoring.py`），不需要安装该包。

### 结果聚合与呈现

每条记录单独打分，然后：

- **POOLED**：所有记录的 TP/FP/参考数相加后计算（原训练脚本的做法，长记录、发作多的病人权重大）；
- **MACRO**：每个病人内部先汇总，再对病人取平均（SzCORE 的 `avg_per_subject`），同时给出标准差、中位数和**最差病人**（`*_macro_min`）；
- 逐病人、逐记录、逐发作、逐误报的表格都会写出。“总有一两个病人效果奇差”这种现象在 pooled 数字里会被掩盖，在这里一眼可见。

阈值无关指标：把逐点概率平均到 1 Hz 后计算 AUROC 和 AUPRC（极端不平衡下 AUPRC 更有信息量），以及逐点的平均负对数似然 `nll`。

可用的指标名（`summary.json`、`train.selection_metric`、`sweep --metric`）：`{event,sample}_{sensitivity,precision,f1,fp_per_day}_{pooled,macro,macro_std,macro_min,macro_median}`、`auroc_*`、`auprc_*`、`nll_*`、`event_latency_median_sec`。

### 关于评估的严谨性

目前只有 train/dev，dev 同时承担选模（选 checkpoint，往往还要选阈值和后处理参数）和报告两种角色，dev 上的数字因此**偏乐观**；报表里会注明这一点。建议：

1. **开发期**沿用现有 8:2 划分迭代，看 POOLED、MACRO 和逐病人结果，尤其关注最差病人。
2. **选模尽量用阈值无关指标**（`auprc_pooled`、`nll_pooled`），把阈值和后处理参数的选择留到最后，以减少对 dev 的过拟合。
3. **最终报告**用病人级交叉验证：split 文件写成 `{"folds": [{"train": [...], "dev": [...]}, ...], "test": [...]}`，用 `--set data.fold=k` 逐折训练，`chb01/chb21` 用 `groups` 保证同折；或者留出若干病人作 `test`（只在训练结束后用最佳 checkpoint 评估一次，`evaluation.splits: [dev, test]`）。
4. **阈值迁移**：在 dev 上用 `sweep` 选工作点，再把它固定下来去评估 test，而不是在 test 上扫。
5. 多随机种子（≥3）重复，用 `compare` 汇总均值和波动。
6. 不同评分器给出的数字差距很大（同一组预测在 TUSZ 上，SzCORE 与 NEDC OVERLAP 的误报率能差 3 倍，详见设计笔记），报告时注明评分规则和全部参数（`summary.md` 会写出后处理参数）。

## 接入新模型

模型就是一个 `nn.Module`，构造签名与输入输出约定如下：

```python
# my_models/unet.py
import torch.nn as nn
from lteeg.registry import MODELS

@MODELS.register("my_unet")            # 或不注册，配置里写 model.name: "my_models.unet:MyUNet"
class MyUNet(nn.Module):
    output_stride = 1                  # 输出时间分辨率比输入低 r 倍时设为 r（如 patch/token 级模型）

    def __init__(self, in_channels, in_samples, num_outputs, depth=4, width=32):
        super().__init__()
        ...

    def forward(self, x):              # x: (B, C, T) 已预处理的 float32
        return logits                  # (B, num_outputs, T) 或 (B, num_outputs, T // r)
```

- `num_outputs` 由任务决定（二分类为 1，多分类为类别数），`in_channels`/`in_samples` 来自配置，其余参数来自 `model.params`，名字拼错会直接报错并列出可接受的参数。
- 输出分辨率低于输入时，框架会在计算损失和推理前把 logits 线性插值到逐采样点。
- 需要多阶段/深监督或额外损失（例如异常检测式的重构损失）时，返回 `lteeg.models.ModelOutput(logits, aux_logits=[...], losses={"recon": ...})`：每个 `aux_logits` 用同样的主损失计算后相加，`losses` 中的项直接加到总损失上并分别记录。
- 需要病人/位置信息的模型（病人条件化、测试时自适应等）设 `wants_meta = True`，`forward(x, meta)` 会收到 `patient`、`rec`、`start` 张量；推理时病人未知，`patient` 为 -1。
- 接入后先跑 `python -m lteeg check-model --set model.name=my_unet --overfit-steps 50`：检查输出形状、前后向耗时与显存，并在单个 batch 上过拟合，确认损失能下降。
- 自定义模块所在的包需在 `PYTHONPATH` 中；若用装饰器注册，需在 `lteeg/models/__init__.py` 中 import 它，或直接用 `"包.模块:类名"` 形式。

## 扩展：新数据集、多分类、新组件

- **新数据集**：只需新的 yaml（导联列表、采样率或 `resample_to`、文件模式、h5 键名、标注文件名与段名、对称导联对）和 split 文件；如果原始格式不是 h5，转换成同样的 h5 布局即可，框架其余部分不变。
- **多分类**：标注行增加一列类别名，配置 `data.label_column`、`data.label_map`、`task.mode: multiclass`、`task.class_names`，损失换成 `ce`/`focal`。事件评估对“任意发作”进行；逐类评估可在 `evaluation/runner.py` 中按类调用同一套评分函数。
- **新的损失、预处理算子、增强**：分别用 `LOSSES`、`PREPROCESSORS`、`AUGMENTATIONS` 注册（见 `lteeg/registry.py`），配置里按名字引用。
- **新的采样策略**：在 `lteeg/data/windows.py` 的 `WindowSampler._select` 中添加分支；采样器只需要返回窗口索引数组。

## 与原项目的差异

| 方面 | 原项目（time_step_level） | LTEEG 默认 | 复现原行为 |
|---|---|---|---|
| 模型输出 | `forward` 内 sigmoid | 输出 logits，用 `BCEWithLogits`（数值稳定、可用 AMP） | 数学上等价，无需改动 |
| 模型参数 | 41.0M，其中约 3.15M 属于从未使用的 `transformer_encoder_layer` | 37.85M（删除死参数），计算结果逐元素相同 | 原权重可经 `load_original_state_dict` 直接加载 |
| 输入长度 | 只能等于 `in_samples`，位置编码上限 6000 token | 任意长度 ≥ 32 | — |
| 训练集构造 | 一次性把所有窗口物化为 numpy 数组再存盘，内存占用随数据量线性增长（75% 重叠时约 4 倍于原始数据） | 只存预处理后的整条记录（memmap），窗口按索引即时读取 | — |
| 滤波 | 每个窗口独立做因果 IIR 滤波，每个窗口开头都有滤波器启动瞬态 | 整条记录滤波一次后缓存（同一组滤波器，SOS 形式） | `configs/experiments/original_faithful.yaml` |
| 末窗处理 | 补零 | 对齐记录末尾，无补零 | `inference.tail: pad` |
| 读取失败 | `try/except: continue` 静默跳过文件；短于一个窗的记录静默跳过 | 一律报错；短记录补零后照常评估 | — |
| 平坦通道 | 除以 0 产生 NaN | 置零并记录 | — |
| 秒 → 采样点 | 标签用 `int()` 截断，评分器用 `round()` | 统一 `round()` | — |
| 选模 | 每 epoch 在 dev 上长程评估，按 pooled event F1（阈值 0.8）保存最佳 | 相同，且可换指标、可早停、记录完整历史 | 默认即如此 |
| 评估输出 | 只打印 pooled 的 sample/event 各 4 个指标 | 逐病人/逐记录/逐发作/逐误报报表 + 宏平均 + 阈值无关指标 + 概率存档 | — |
| 依赖 | `epilepsy2bids`、`timescoring` 的本地可编辑安装 | 仅 numpy/scipy/h5py/pyyaml/torch | — |

## 面向后续研究方向的预留

这些方向现在都不需要解决，但框架没有把它们堵死：

- **极度不平衡**：采样比例、按病人分组抽样、每 epoch 重抽背景、`sampling.name: all` + 加权损失（`pos_weight`、focal、dice）、起止点忽略区间都已经是配置项；评估同时给出 AUPRC 和每日误报数这两个不平衡下更有意义的量。
- **患者特异性**：所有结果按病人呈现并给出最差病人；`wants_meta` 让模型拿到病人编号；`group_by: patient` 平衡各病人贡献；split 文件支持 folds，可做病人级交叉验证；`model.init_checkpoint` 可加载已训练权重做个体化微调（目前划分粒度是病人级，若要做“同一病人前几次发作训练、后几次测试”，需在 split 中加入记录级划分，接口位置在 `data/split.py` 与 `data/loading.py`）。
- **长程测试**：训练期验证和最终评估都是整段记录的连续推理和事件级评分；`predict` 可对任意新记录输出发作起止时间（TSV），`false_alarms.csv` 便于分析误报来源。
- **时间序列异常检测式的范式**：逐点标签与逐点输出本身就是 TSAD 的形式；模型可以通过 `ModelOutput.losses` 加入重构等自监督损失；评估刻意没有使用 TSAD 中常见但已被证明会严重高估性能的 point-adjust，而是使用带固定容差的事件评分和 AUPRC。
- **多分类**：标签、损失、推理、评估链路都已按 K 类实现。

## 已知限制

- 开发环境无法访问真实 CHB-MIT 数据，也没有 GPU。真实数据的读取路径只用同格式的合成数据走通过；在你的机器上请先运行 `inspect`。CUDA 专属路径（fp16 GradScaler、`pin_memory`、`torch.compile`）没有实际运行过；bf16 autocast 只在 CPU 上测过。
- Windows 下的多进程 DataLoader 是在 Linux 上用 spawn 方式模拟验证的，没有在真正的 Windows 上运行过。
- 若要加载原作者的比赛权重（`model.init_checkpoint`），注意它是在 TUSZ/Siena 上用 epilepsy2bids 的双极导联顺序训练的，该顺序很可能与这里的 CHB-MIT 导联顺序不同，需先核对并按名称重排通道（T7/P7/T8/P8 分别对应旧命名 T3/T5/T4/T6）。
- 缓存体积：CHB-MIT 约 980 小时，18 通道 float32 约 65 GB（`data.cache_dtype: float16` 约 33 GB；z-score 之后 float16 的精度损失可以忽略）。

## 参考文献

- K. Wu, Z. Zhao, B. Yener. *Large EEG-U-Transformer for Time-Step Level Detection Without Pre-Training* (SeizureTransformer). arXiv:2504.00336, 2025.
- J. Dan et al. *SzCORE: Seizure Community Open-Source Research Evaluation framework for the validation of EEG-based automated seizure detection algorithms*. Epilepsia, 2024. arXiv:2402.13005.
- J. Dan et al. *SzCORE as a benchmark: report from the seizure detection challenge at the 2025 AI in Epilepsy and Neurological Disorders Conference*. arXiv:2505.18191.
- A. Shoeb. *Application of machine learning to epileptic seizure onset detection and treatment*. PhD thesis, MIT, 2009（CHB-MIT 数据集）。
- Y. A. Farha, J. Gall. *MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation*. CVPR 2019（`tmse` 平滑损失）。
- M. Perslev et al. *U-Time* (NeurIPS 2019) / *U-Sleep* (npj Digital Medicine 2021)（时间序列稠密分割）。
- T.-Y. Lin et al. *Focal Loss for Dense Object Detection*. ICCV 2017.
- S. Kim et al. *Towards a Rigorous Evaluation of Time-series Anomaly Detection*. AAAI 2022（point-adjust 的问题）。
- J. Paparrizos et al. *Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection*. VLDB 2022.
