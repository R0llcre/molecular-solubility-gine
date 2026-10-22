# gine_desc：置信度校准、可解释性与性能提升（ESOL，2026-10-17 至 10-18）

本报告的所有数字都来自 2026-10-17 至 10-18 在本机（Apple M5，CPU，PyTorch 2.14.1 / PyG 2.8.1 / RDKit 2026.09.1，Python 3.12 venv `~/.venvs/gnn-transformer`）的真实运行。共三轮：
- **第一轮**（第 1–4 节）：高斯 NLL 校准 + 积分梯度可解释性。
- **第二轮**（第 5–8 节）：深度集成、conformal、归因定量验证、新颖度分析，以及第一次按事先规则调参、锁定后揭开测试集。
- **第三轮**（第 9 节）：继续提升性能，用全新划分做确认，原始测试集第二次查看。

`ChemPath-GINE/reports/summary_findings.md` 未改动。原始结果文件见第 12 节，局限与披露见第 10 节。

## 0. 结论先行

**校准与可解释性**
- **NLL 头，random split**：换成高斯 NLL 输出头（同时预测 μ 和 log σ²）后，RMSE 没有退化（0.558 vs 同协议 MSE 基线 0.556）。**ECE 从 0.070 降到 0.033**，5/5 个 seed 都下降。90% 区间实际覆盖 0.880，宽度 1.64 logS；用验证集缩放后覆盖 0.900，宽度 1.73，比常数误差条的 1.86 窄。
- **NLL 头，scaffold split**：单模型过度自信，90% 覆盖只有 0.827。**5 个模型的深度集成**不做任何缩放就把 90% 覆盖恢复到 0.899，但 ECE（0.056）仍高于常数误差条（0.032）（第 5 节）。
- **σ 有信息量**：σ 最小的一半分子 RMSE 0.41，最大的一半 0.67（random）。
- **σ 与"没见过的分子"**：单模型的 σ 基本不随离训练集远近变化；集成的模型间分歧在 random 上会变大（最陌生的四分之一是最熟悉的 1.5–1.8 倍），但在 scaffold 测试集里反而略低。最陌生的那部分分子仍偏过度自信，90% 覆盖 0.74–0.86（第 7 节）。
- **积分梯度比注意力更可靠，但注意力也不是没用**（第 6 节）：
  - 逐个原子遮挡时，|IG| 和预测变化幅度的分子内相关是 0.47 / 0.50，注意力是 0.30 / 0.24。
  - 遮掉排第一的原子，IG 选的原子让预测变化达到随机遮挡的 2.2–2.7 倍，注意力选的是 1.9–2.1 倍。
- **化学读法**（第 4、6 节）：
  - 羟基 O（每原子 +1.26 logS）、羰基 O（+0.69）、芳香 N（+0.38）等极性基团在 5/5 个 seed、两种划分上都提高预测溶解度，碳骨架平均降低。
  - 和 Crippen 经验原子贡献只是弱一致（原子级相关约 0.2）。
  - 卤素的效应可能被分子量描述符吸收（证据有混淆），原子级颜色不按字面读。

**性能**（6 描述符单模型是简历上的版本）

| 原始测试集 RMSE | random | scaffold |
|---|---:|---:|
| 简历上的数字（6 描述符单模型，存档） | 0.559 | 0.929 |
| 同一模型 CPU 重跑（配对基线） | 0.556 | 0.919 |
| + Crippen logP（desc7），单模型，原训练入口 | 0.564 | 0.869 |
| desc7 GINE × 5 集成（第二轮锁定） | 0.543 | 0.847 |
| desc7 GINE × 5 + D-MPNN × 5，只用训练集训练 | 0.541 | 0.840 |
| **最终系统**：同上，再用训练集 + 验证集重训 | **0.531** | **0.813** |

- **相对配对基线**：最终系统好 −0.025（95% CI [−0.050, −0.001]）和 −0.106（[−0.178, −0.033]）。区间按分子做 bootstrap，两种划分都是 5/5 个 seed 更好。
- **全新确认划分上**（random seed 6–10、5 个随机骨架划分；没参与模型结构选择，但重训的决定是在上面做的）：
  - 只用训练集的系统比基线好 −0.038 [−0.057, −0.019] / −0.103 [−0.130, −0.077]。
  - 第三轮相对第二轮 GINE × 5 的增量：加 D-MPNN −0.009 / −0.017，再加重训合计 −0.022 / −0.042，区间都在 0 以下。
- **真正有用的是四件事**：
  1. 把 Crippen logP 加回描述符：这是 Delaney 溶解度方程里最重要的一项，主要帮 scaffold。
  2. 深度集成：集成带来的提升大部分来自这一步。
  3. 加入 D-MPNN 做跨架构集成：增量小，主要在 scaffold。
  4. 用训练集 + 验证集重训。
- **其余约 30 组改动没带来稳定提升**：dropout、EMA、cosine 组合、sum 池化、加宽、加深、学习率、batch、weight decay、更多描述符、其他融合方式。16 个描述符通过了验证集规则，但不如只加 logP。
- **披露**：原始测试集看过两次（第二轮锁定后、第三轮冻结后），第二轮之前还提前看过一些测试数字（第 10 节）。第三轮的增量在原始测试集上方向基本一致，但区间碰到或跨过 0；在确认集上是显著的。

## 1. 实验设置

| 项目 | 取值 |
|---|---|
| 代码库 | `ChemPath-GINE`（简历上 0.56 / 0.93 两个数就来自这里的 `ablation_runs/esol_fusion_{random,scaffold}_5seed`；MolFuse-GINE 的 random split 实现不同（按重复 SMILES 分组），换过去就不能和那两个数直接比） |
| 模型 | `gine_desc`：3 层 GINE（hidden 128）→ GraphPool（注意力池化 + mean + max）→ 门控加法融合描述符 → MLP 头 |
| 配置 | `configs/esol_fusion_ablation_{random,scaffold}.yaml`，与存档一致：lr 3e-4，batch 64，最多 120 epoch，patience 18，dropout 0.10，grad clip 5 |
| 描述符（6 个） | MinimumDegree, MolecularWeight, HBondDonors, RingCount, RotatableBonds, PolarSurfaceArea（**不含 logP**） |
| 数据 | ESOL 1,128 个分子，train/val/test = 789/169/170（scaffold 为 791/169/168） |
| 划分 | random：seed 决定划分和初始化；scaffold：确定性划分，5 个 seed 只改变训练随机性 |
| 选 checkpoint | 只看验证集 loss；测试集只在最终 checkpoint 上评一次 |
| 标准化 | 描述符和目标的 scaler 只在训练集上拟合 |
| 设备 | CPU（每个 run 约 20 秒，40 个 run 并行约 3 分钟） |

### 对比的四种训练方式

| 模式 | 均值怎么训 | 方差怎么来 | 选 checkpoint |
|---|---|---|---|
| `mse`（基线） | MSE，**直接调用未改动的原训练入口** `chempath_gine.training.train.main` | 常数 σ = 验证集 RMSE（朴素的同方差不确定度） | val MSE |
| `nll`（主方法） | 高斯 NLL：½·(log σ² + (y−μ)²/σ²)，μ 和 log σ² 共用一个头的两路输出 | 网络输出 | val NLL |
| `beta_nll`（敏感性检验，事先定好） | β-NLL（Seitzer et al. 2022，β = 0.5） | 网络输出 | val NLL |
| `faithful`（事后补的） | MSE（和基线完全一样）；方差头读 stop-gradient 后的表示和 μ，用 NLL 训练（Stirn et al. 2023） | 网络输出；选出 checkpoint 后冻结均值路径，单独把方差头训到收敛（第二阶段） | 均值：val MSE；方差头：val NLL |

`faithful` 是看到 scaffold **验证集**上 NLL 早停过早之后加的，前后修了两次：
1. 第一次实现只对 μ 做了 stop-gradient，方差梯度仍会流进共享表示。改成同时 detach 表示。
2. 独立代码评审发现，按 val MSE 选出的 checkpoint 上，方差头还远没收敛：它和均值共用一个被 val MSE 驱动衰减的学习率，σ 明显偏大（第二阶段之前，random 上验证集的 σ 是实际误差的 1.5–2.4 倍）。于是加了第二阶段：冻结均值路径（eval 模式），只在训练集残差上训练方差头（Adam，lr 1e-3，按 val NLL 早停，patience 50）。均值预测不受影响。

两次都重跑了全部 10 个 run，表里只报最终版本。

## 2. 指标怎么算

所有指标都在原始单位（log10 mol/L）上算。μ、σ 由标准化空间反变换：μ = μ̂·s_y + m_y，σ = exp(½·log σ̂²)·s_y，其中 (m_y, s_y) 是训练集上的目标 scaler。

| 指标 | 定义 |
|---|---|
| RMSE | √mean((y−μ)²) |
| NLL | mean[½·log(2πσ²) + (y−μ)²/(2σ²)]，含常数项，越低越好 |
| 覆盖率 cov(p) | 测试分子中 y 落在中心区间 μ ± z₍₁₊ₚ₎/₂·σ 内的比例 |
| **ECE** | 名义覆盖率 p ∈ {0.05, 0.10, …, 0.95}（19 档）上 \|cov(p) − p\| 的平均（区间版回归校准误差，Kuleshov et al. 2018 的思路） |
| 90% 覆盖率 / 宽度 | cov(0.90)；宽度 = mean(2 × 1.645 × σ)，单位 logS |
| 验证集缩放 s | s² = mean_val[((y−μ)/σ)²]，即让验证集 NLL 最小的单一 σ 缩放系数（闭式解）；"重新校准"指测试时用 s·σ |
| Spearman(σ, \|err\|) | σ 能否给误差排序 |
| 选择性 RMSE | 按 σ 从小到大排序，前一半 / 后一半各自的 RMSE |

常数 σ 基线的 σ 本来就是用验证集拟合的（σ = val RMSE，即同方差高斯在验证集上的 MLE），所以它不再做二次缩放。

## 3. 校准结果

### 3.1 主表（测试集，5 seed 均值 ± 标准差）

| 划分 | 方法 | RMSE | NLL | ECE | 90% 覆盖率 | 90% 区间宽度 |
|---|---|---:|---:|---:|---:|---:|
| random | MSE 基线 + 常数 σ | **0.556 ± 0.042** | 0.839 ± 0.068 | 0.070 ± 0.035 | 0.902 ± 0.028 | 1.86 ± 0.08 |
| random | NLL 头，原始 σ | 0.558 ± 0.032 | 0.763 ± 0.076 | **0.033 ± 0.011** | 0.880 ± 0.018 | **1.64 ± 0.11** |
| random | NLL 头，σ × 验证集缩放（s = 1.06 ± 0.03） | 0.558 ± 0.032 | **0.761 ± 0.064** | 0.037 ± 0.028 | **0.900 ± 0.026** | 1.73 ± 0.16 |
| scaffold | MSE 基线 + 常数 σ | **0.919 ± 0.018** | **1.338 ± 0.019** | 0.032 ± 0.014 | 0.874 ± 0.014 | 2.84 ± 0.07 |
| scaffold | NLL 头，原始 σ | 0.968 ± 0.053 | 1.393 ± 0.036 | 0.102 ± 0.027 | 0.827 ± 0.015 | 2.60 ± 0.26 |
| scaffold | NLL 头，σ × 验证集缩放（s = 1.33 ± 0.11） | 0.968 ± 0.053 | 1.360 ± 0.046 | 0.033 ± 0.006 | 0.939 ± 0.020 | 3.45 ± 0.11 |

存档里同配置的 `gine_desc` RMSE：random 0.559 ± 0.046，scaffold 0.929 ± 0.024（`ablation_runs/esol_fusion_*_5seed/ablation_aggregate.csv`）。本次重跑的 MSE 基线（0.556 / 0.919）与存档一致：差 0.003（random）/ 0.010（scaffold），都小于 seed 间标准差（0.042 / 0.018–0.024）；划分包含相同的分子、相同的顺序。存档很可能是在 CUDA 上跑的（同批次的日志显示 device=cuda，但存档本身没记录设备），本次在 CPU 上跑（2 线程），单个 seed 的 RMSE 最多差 0.028。CPU 上 GINE 的结果在线程数固定时逐位可复现；独立审查用 1/2/4/8 线程重跑同一个 MSE 基线 seed，得到 0.580 / 0.561 / 0.559 / 0.558，所以单 seed 的小差异不必解读。配对比较一律用本次重跑的 CPU 基线。

### 3.2 和基线的配对比较（同 seed 配对，n = 5）

| 划分 | 指标 | NLL − MSE 基线 | NLL 更好的 seed 数 | 配对 t 检验 p |
|---|---|---:|---:|---:|
| random | RMSE | +0.002 ± 0.028 | 2/5 | 0.90 |
| random | NLL（原始 σ） | −0.076 ± 0.079 | 4/5 | 0.10 |
| random | ECE（原始 σ） | −0.038 ± 0.025 | 5/5 | 0.03 |
| scaffold | RMSE | +0.049 ± 0.069 | 2/5 | 0.18 |
| scaffold | ECE（原始 σ） | +0.070 ± 0.021 | 0/5 | 0.002 |
| scaffold | ECE（验证集缩放后） | +0.001 ± 0.014 | 1/5 | 0.91 |

结论：random split 上"没有明显退化"成立。scaffold split 上 RMSE 平均差 0.05，5 个 seed 的方差很大（单个 seed 从 0.911 到 1.038），统计上不显著，但不能说"完全没有退化"。

### 3.3 不确定度能不能给误差排序

| 划分 | 方法 | Spearman(σ, \|err\|) | σ 最小一半的 RMSE | σ 最大一半的 RMSE |
|---|---|---:|---:|---:|
| random | NLL 头 | 0.34 ± 0.05 | 0.415 ± 0.036 | 0.671 ± 0.049 |
| scaffold | NLL 头 | 0.27 ± 0.04 | 0.794 ± 0.008 | 1.115 ± 0.088 |

常数 σ 基线无法排序（所有分子 σ 相同）。这是异方差头相对基线最实在的好处：**它知道哪些分子没把握**。

### 3.4 另外两种训练方式（敏感性检验）

| 划分 | 方法 | RMSE | ECE 原始 → 缩放后 | 90% 覆盖率（缩放后） | 90% 宽度（缩放后） | Spearman(σ,\|err\|) | 平均最佳 epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| random | β-NLL | 0.571 ± 0.035 | 0.040 → 0.033 | 0.891 | 1.75 | 0.34 | 27.6 |
| random | faithful | 0.563 ± 0.046 | 0.071 → 0.061 | 0.918 | 1.95 | 0.18 | 43.6 |
| scaffold | β-NLL | 1.083 ± 0.102 | 0.073 → 0.026 | 0.927 | 3.61 | 0.27 | 7.8 |
| scaffold | faithful | 0.932 ± 0.037 | 0.029 → 0.020 | 0.905 | 3.14 | 0.12 | 33.2 |

- β-NLL 在这里没有帮助，scaffold 上早停得更早（第 ~8 epoch），RMSE 更差。
- faithful 把 μ 的精度保住了（scaffold 0.932，与基线 0.919 在噪声内，配对 p = 0.56）。scaffold 上不缩放就已经校准得不错：原始 ECE 0.029，90% 覆盖 0.906，宽度 3.22 logS。但它的 σ 给误差排序的能力弱得多：Spearman 0.18（random）/ 0.12（scaffold），NLL 头是 0.34 / 0.27。random 上它的 ECE（0.071）也不比常数 σ 基线好。原因是方差头只能读一个为均值训练、被冻结的表示，学不到多少"哪里难"。
- 所以这是一个**取舍**：联合训练 NLL 的 σ 更有信息量（能排序误差），但在分布偏移下可能拖累 μ（scaffold +0.049，p = 0.18，不显著），原始 σ 也过度自信；把均值和方差解耦能保住 μ 和整体覆盖率，但 σ 主要只剩整体尺度。

### 3.5 图

![可靠性图](figures/calibration_interpretability_2026-10-17/reliability_diagram.png)

`figures/calibration_interpretability_2026-10-17/reliability_diagram.png`（另有 `.svg`）：横轴是名义覆盖率，纵轴是测试集实际覆盖率，阴影是 5 个 seed 的最小–最大范围。scaffold 上 NLL 原始 σ 明显在对角线下方，即过度自信；乘上验证集缩放系数后回到对角线附近。

## 4. 可解释性结果

### 4.1 方法

- **积分梯度（IG）**：对预测均值 μ 求 IG。输入取三组，在同一条路径上联合积分：原子嵌入 h₀ = AtomEncoder(原子特征)、键嵌入 e₀ = EdgeEncoder(键特征)、标准化后的 6 个描述符；基线全为 0（对描述符而言 0 就是训练集均值）。路径 x(α) = α·x，128 步中点黎曼和。原子归因 = Σ_hidden (h₀ ⊙ 平均梯度)；键归因 = 两条有向边之和；最后乘 s_y 换算成 logS 单位。由完备性，原子 + 键 + 描述符 ≈ μ(x) − μ(基线)。
  - 完备性实测：random 850 个分子样本，|误差| 中位数 0.0086、最大 0.051 logS，相对于 |μ(x) − μ(基线)| 的中位数 1.44 logS。scaffold 840 个样本，中位数 0.0069、最大 0.038。
  - **对称性修正**：GraphPool 的 max-pool 分支对化学等价原子（如蒽醌的两个羰基 O）会出现精确并列，梯度只流向 torch.max 选中的那一个，IG 会在等价原子之间任意分配功劳。修正方法是在 RDKit 对称类（`CanonicalRankAtoms(breakTies=False)`）内取平均，键在"两端原子类 + 键型"的类内取平均。每个分子的总和不变，完备性仍成立。未修正的原子值保存在 `atom_attributions.csv` 的 `ig_raw` 列。
- **注意力**：`gine_desc` 没有 Transformer 注意力层，模型里唯一的注意力是 GraphPool 的注意力池化（对最后一层节点表示做 softmax，权重和为 1）。图里画的"readout attention"就是它。它表示读出时更看重哪些原子的表示，不等于"这个原子让溶解度变高/变低"。在单个分子内，它和 |IG| 的 Spearman 相关平均只有 0.31（random）/ 0.27（scaffold）。
- **画图**：RDKit `MolDraw2D` 高亮原子和键。IG 用红–蓝发散色，蓝色表示提高预测 logS（更易溶），红色表示降低。颜色按每个分子的 max|IG| 归一化，圆的半径随 |IG| 线性增大（0.25 + 0.30·|IG|/max|IG|）。注意力用单色（橙）深浅表示。
- **选分子**：展示用模型取 random split 下 NLL 头测试 RMSE 为中位数的那个 seed（seed 3，RMSE 0.579），不挑最好的。在它的测试集里取 |误差| 最小的 4 个（要求重原子 ≥ 8，避免选到甲烷这种画不出东西的小分子）和 |误差| 最大的 4 个（无过滤）。

### 4.2 数据集级的化学读法（每原子平均 IG，单位 logS；5 个 seed 汇总）

| 原子环境 | random：平均 IG | random：均值为正的 seed 数 | random：原子样本数 | scaffold：平均 IG | scaffold：均值为正的 seed 数 |
|---|---:|---:|---:|---:|---:|
| O–H（羟基 / 羧酸 OH） | **+1.26** | 5/5 | 303 | +0.72 | 5/5 |
| O=（羰基 C=O） | **+0.69** | 5/5 | 481 | +0.74 | 5/5 |
| O=（S=O / P=O） | +0.52 | 5/5 | 57 | +0.46 | 5/5 |
| O（醚 / 酯） | +0.42 | 5/5 | 333 | +0.44 | 5/5 |
| N（芳香） | +0.38 | 5/5 | 296 | +0.42 | 5/5 |
| N–H（胺 / 酰胺） | +0.37 | 5/5 | 261 | +0.19 | 5/5 |
| N（叔胺 / 腈 / 其他） | +0.43 | 5/5 | 163 | +0.05 | 4/5 |
| C（芳香） | **−0.16** | 0/5 | 4,187 | −0.09 | 0/5 |
| C（脂肪 sp3/sp） | **−0.19** | 0/5 | 3,453 | −0.07 | 1/5 |
| C（sp2，非芳香） | −0.29 | 0/5 | 739 | −0.19 | 0/5 |
| N（硝基，C–NO₂） | −0.49 | 0/5 | 44 | −0.35 | 0/5 |
| S | −0.41 | 0/5 | 115 | −0.16 | 1/5 |
| Cl | −0.06 | 2/5 | 509 | −0.01 | 3/5 |
| F | +0.04 | 3/5 | 99 | +0.05 | 4/5 |
| Br | +0.48 | 5/5 | 50 | −0.05 | 2/5（仅 5 个原子） |
| I | +0.84 | 4/4（seed 1 的测试集里没有 I） | 15 | （测试集中没有） | — |
| P | +0.66 | 5/5 | 31 | +0.72 | 5/5（仅 10 个原子） |
| O（芳香环） | +0.32 | 4/5 | 18 | +0.74 | 5/5（70 个原子） |

"x/5" 是在有这类原子的 seed 里，平均 IG 为正的 seed 数。硝基 O、N-氧化物 / 硝酸酯这几类只放在 CSV 里（硝基 O 的两个氧原子在分子图里带不同电荷，符号相反，合在一起看意义不大）。分组规则：硝基按 SMARTS `[#6][N+](=O)[O-]` 或中性写法 `[#6]N(=O)=O` 识别；双键连 C 的 O 算羰基，双键连 S/P 的 O 单独一类。

完整表：`ChemPath-GINE/calibration_runs/interpretability/esol_{random,scaffold}/atom_group_ig_summary.csv`。

**描述符的归因（random，mean |IG|，logS）**：MolecularWeight 1.15 > RingCount 0.54 > PolarSurfaceArea 0.46 > RotatableBonds 0.38 > HBondDonors 0.07 > MinimumDegree 0.03。图和描述符各占多少，取决于怎么算：按每个分子"图分支归因总和"与"描述符归因总和"的绝对值之比，图占 39%（scaffold 41%）；但图里正负原子会互相抵消，如果改成把每个原子的 |IG| 加起来，原子占 63%（scaffold 57%）。所以只能说：分子级净效应里描述符分量更大，而图分支在原子层面做了大量正负修正。

**卤素为什么不按字面解读**：含 Cl（不含 Br/I）的测试分子，MolecularWeight 描述符的平均归因是 −0.82 logS（random，n = 179）/ −2.09（scaffold，n = 150）。不含任何卤素的分子是 +0.44（n = 594）/ −0.60（n = 655）（`halogen_mw_check.csv`）。这符合"卤素变重、变疏水的效应被 MW 描述符记账"的解释，但证据有混淆：含 Cl 的分子本身也更难溶（平均 logS −4.36 vs −2.63），MW 归因本来就会更负；而含 Br/I 的分子 MW 归因只有 −0.39，比含 Cl 的还小，解释不了 Br/I 的正值。Br/I 原子样本也很少（50 和 15 个）。所以只能说"可能被吸收"。面试时要主动说这是联合归因 + 相关特征带来的局限。

### 4.3 展示分子（random split，seed 3）

| 标签 | 分子 | 真值 | 预测 μ ± σ | \|误差\| | 主要归因（IG，logS） | 一句话解读 |
|---|---|---:|---:|---:|---|---|
| best1 | 2,4,6-三氯苯酚 | −2.34 | −2.34 ± 0.36 | 0.00 | 酚羟基 O **+2.78**；对位 Cl −0.54；环碳 −0.57/−0.49 | 酚羟基能和水形成氢键，是唯一明显的增溶位点；芳环和对位 Cl 是疏水面，降低溶解度。 |
| best2 | 抗蚜威（pirimicarb） | −1.95 | −1.94 ± 0.52 | 0.01 | 嘧啶环 N +0.75；二甲氨基 N +0.63；氨基甲酸酯 N +0.63、C=O +0.52；芳香碳平均 −0.12，甲基碳正负相抵（平均 0.00） | 多个氮原子和羰基是氢键受体，拉高溶解度；碳原子贡献很小。 |
| best3 | 4-溴苯酚 | −1.09 | −1.10 ± 0.38 | 0.01 | 羟基 O **+2.20**；与 OH 相连的环碳 −0.86 | 同样由酚羟基主导。注意力池化最看重 Br（0.40），IG 却在 OH 上：注意力 ≠ 归因的直观例子。 |
| best4 | 苯腈 | −1.00 | −0.99 ± 0.34 | 0.01 | 腈基 N +0.41；描述符合计 +1.97（MW +1.46） | 分子小是它相对易溶的主要原因（描述符分支），极性腈基再加一点。 |
| worst1 | 磷胺（Dimecron / phosphamidon） | **+0.52** | −1.26 ± 0.60 | 1.78 | P +0.99、酰胺 N +0.75、P=O 的 O +0.74、酰胺 C=O +0.50；Cl −0.56 | 磷酸三酯和酰胺都是强极性基团，方向是对的，但幅度远远不够（实际几乎能和水混溶）。训练集里有机磷酸酯不多：seed 3 训练集有 5 个磷酸三酯，包括结构相近的久效磷（logS +0.65），但它和训练集最近邻的 Tanimoto 只有 0.26，处在测试集最低的 8%。σ 0.60 只略高于该测试集 σ 的 75 分位（0.59），误差却是 2.97σ，属于"自信地预测错了"。 |
| worst2 | 溴敌隆（bromadiolone） | −4.45 | −6.13 ± **1.02** | 1.69 | 两个 OH +1.30/+0.67、内酯 C=O +0.52；描述符 −4.71（MW −3.32，RingCount −1.60） | 35 个重原子、5 个环，描述符分支把"大而多环 → 不溶"外推过头了。σ = 1.02 是展示集里最大的，模型知道自己没把握。 |
| worst3 | 1-十八醇 | −8.40 | −6.73 ± 0.88 | 1.67 | 端羟基 O **+2.18**；链上 18 个碳平均 −0.27（−0.56 ~ +0.15）；RotatableBonds −3.12 | 模型给 OH 的功劳和苯酚里差不多大。一种可能的解释是没学到"一个 OH 的增溶作用会被 C18 长链稀释"，所以高估了溶解度。注意归因是相对全零基线的，不等于因果：描述符这里两个方向都有很大的贡献（RingCount +1.14，RotatableBonds −3.12）。 |
| worst4 | 邻苯二甲酸二辛酯 | −5.11 | −6.70 ± 0.94 | 1.58 | 两个酯羰基 O 各 +0.61、酯 O +0.56；MW −2.12、RotatableBonds −2.59 | 极性的酯基方向正确，但两条辛基链经描述符（分子量、可旋转键）被重罚，预测偏不溶。 |

这个测试集里 σ 的中位数 0.47、90 分位 0.71。四个误差最大的分子 σ 在 0.60–1.02，四个最准的在 0.34–0.52。

**图的位置**（目录 `GNN-Transformer/reports/figures/calibration_interpretability_2026-10-17/esol_random_interpretability/`）：
- 总览：`overview_ig_attention.png`（8 个分子 × [IG, 注意力]，带色条）
- 单个分子：`best{1..4}_ig.svg|png`、`best{1..4}_attention.svg|png`、`worst{1..4}_ig.svg|png`、`worst{1..4}_attention.svg|png`
- 每个分子的原子级数值：`ChemPath-GINE/calibration_runs/interpretability/esol_random/showcase_molecules.csv` 和 `atom_attributions.csv`

![IG 和注意力总览](figures/calibration_interpretability_2026-10-17/esol_random_interpretability/overview_ig_attention.png)

## 5. 第二轮（2026-10-18）：深度集成 + conformal 区间

**做法**：对每个外层 seed（同一个划分），用不同初始化再训 4 个成员（init_seed = 1000·seed + m），共 5 个成员。集成用混合矩：μ = 成员均值，σ² = 成员 σ² 的平均（数据噪声，aleatoric）+ 成员 μ 的方差（模型不确定性，epistemic）。

每种方法都按三种方式评估：原始 σ、σ × 验证集缩放、**split conformal**。conformal 的做法是：在验证集上取 |y−μ|/σ 的 ⌈(n+1)p⌉ 分位数 q，区间取 μ ± q·σ；σ 恒定时，它就退化成普通的残差 conformal。

**结果**（测试集，5 个重复的均值 ± 标准差；6 个描述符的模型）：

| 划分 | 方法 | RMSE | 原始 σ：ECE / 90% 覆盖 / 宽度 | conformal：90% 覆盖 / 宽度 | Spearman(σ,\|err\|) |
|---|---|---:|---|---|---:|
| random | MSE 单模型 + 常数 σ | 0.556 ± 0.042 | 0.070 / 0.902 / 1.86 | 0.892 / 1.83 | — |
| random | NLL 单模型 | 0.558 ± 0.032 | 0.033 / 0.880 / 1.64 | 0.892 / 1.72 | 0.34 |
| random | **NLL 5 成员集成** | **0.553 ± 0.024** | 0.035 / **0.900** / 1.65 | 0.891 / **1.64** | **0.40** |
| scaffold | MSE 单模型 + 常数 σ | 0.919 ± 0.018 | 0.032 / 0.874 / 2.84 | 0.885 / 2.91 | — |
| scaffold | NLL 单模型 | 0.968 ± 0.053 | 0.102 / 0.827 / 2.60 | 0.933 / 3.40 | 0.27 |
| scaffold | **NLL 5 成员集成** | 0.956 ± 0.045 | 0.056 / **0.899** / 2.95 | 0.932 / 3.24 | 0.26 |
| scaffold | faithful 5 成员集成 | 0.914 ± 0.018 | 0.033 / 0.929 / 3.36 | 0.907 / 3.02 | 0.11 |

要点：
- **集成恢复了 scaffold 上的 90% 覆盖**：NLL 单模型只有 0.827，5 成员集成不做任何缩放就到了 0.899。但它的 ECE（0.056）仍高于常数 σ（0.032）和 faithful 集成（0.033），所以说"恢复了 90% 覆盖"，不说"解决了校准"。
- **random 上集成也正好 0.900**：σ 和误差的相关从 0.34 升到 0.40；σ 最小的一半 RMSE 0.378，最大的一半 0.684。
- **random 上，异方差 σ 让区间更窄**：同样大约 89% 的覆盖，NLL 集成的 conformal 区间宽 1.64，常数 σ 的残差 conformal 宽 1.83，窄了 0.18 logS（4/5 个 seed，配对 p = 0.025）。变窄来自 σ 能区分难易，不是来自 conformal 本身（NLL 集成原始 σ 已经是 1.65）。
- **scaffold 上反过来**：NLL 集成的 conformal 区间比常数 σ 宽（3.24 vs 2.91，5/5 个 seed）。
- **集成对 RMSE 帮助很小**：NLL 集成比单模型只好 0.005（random）/ 0.012（scaffold）。同架构成员的测试残差相关约 0.90（两种划分都是），大部分误差是成员共有的，平均掉不了。
- **conformal 的保证只是近似的**：同一个验证集还用来选 checkpoint、早停方差头和选配置，验证残差偏乐观。random 上 conformal 覆盖 0.88–0.89，略低于 0.90，差距在约 1 个标准误以内。scaffold 上验证集和测试集是不同骨架，可交换性本来就不成立。

图：`figures/calibration_interpretability_2026-10-17/reliability_ensemble_conformal.png`

## 6. 第二轮：积分梯度的定量验证

两种检验（脚本 `scripts/validate_attributions.py`，NLL 模型，5 个 seed × 2 种划分）。

**(a) 删除检验**：把原子嵌入置 0（即 IG 的基线），描述符保持不变，看预测变化 |Δμ|（logS）。

| 划分，k | 遮 IG 最大的 k 个 | 遮注意力最大的 k 个 | 随机遮 k 个 | IG 胜过随机的分子比例 |
|---|---:|---:|---:|---:|
| random，k=1 | **0.70** | 0.62 | 0.32 | 84% |
| random，k=3 | 0.85 | 0.86 | 0.58 | 69% |
| scaffold，k=1 | **0.46** | 0.36 | 0.17 | 89% |
| scaffold，k=3 | 0.80 | 0.75 | 0.30 | 84% |

- **k=1 时 IG 更好，但注意力也明显强于随机**：IG 选的原子让 |Δμ| 达到随机的 2.23 / 2.75 倍，注意力是 1.95 / 2.13 倍。k ≥ 3 时，random 上注意力和 IG 打平甚至略高，因为 IG 选出的 top-k 里正负原子会互相抵消。
- **方向对**：遮掉 IG 最正的原子，μ 下降的分子占 80–86%；遮掉 IG 最负的原子，μ 上升的占 96–98%。
- **逐原子遮挡，同口径比较**：|IG| 与 |Δμ| 的分子内 Spearman 为 0.47（random）/ 0.50（scaffold），注意力与 |Δμ| 为 0.30 / 0.24。带符号的 IG 与 −Δμ 相关更高（0.70 / 0.74），这说明 IG 的正负方向也对。
- **局限**：置 0 不等于真正删除原子，它仍通过键嵌入传消息。random 上随机遮挡本身会让 μ 平均上移 0.08–0.19；scaffold 上基本不变或略下移。

**(b) 与 Crippen 经验原子贡献比较**：RDKit Crippen logP 的原子贡献，H 的贡献并到相连的重原子上（求和等于 MolLogP，误差 1e-14），取负号后和 IG 比。

| 指标 | random | scaffold |
|---|---:|---:|
| 原子级合并 Spearman | 0.23 ± 0.08 | 0.22 ± 0.03 |
| 分子级 Spearman(Σ 原子 IG, −MolLogP) | 0.41 ± 0.11 | 0.45 ± 0.15 |
| 原子组内 Spearman（去掉组均值后） | −0.04 | −0.11 |

结论是**只是弱一致**。按原子组的平均值看符号，两种划分各有 10 组一致：
- **random 上不一致的**：P、Br、F、羰基 O、硝基 O、硝基 N、非芳香 sp2 碳。
- **scaffold 上不一致的**：F、芳香环 O、羰基 O、硝基 O、非芳香 sp2 碳。
- **Cl 两种划分上都一致**，所以不能笼统说"卤素不一致"。
- **可能的原因**：卤素和 P 的效应部分被分子量描述符吸收；Crippen 预测的是 logP 而不是溶解度；两者在 C=O、硝基上分配功劳的方式也不同。

图：`figures/calibration_interpretability_2026-10-17/attribution_validation.png`

## 7. 第二轮：σ 会不会在"没见过的分子"上变大

对每个测试分子，计算它和本 run 训练集的最大 Tanimoto 相似度（Morgan r=2，2048 位），按相似度四分位分箱（脚本 `scripts/sigma_vs_similarity.py`）。

| 模型 | Spearman(相似度, σ) | Spearman(相似度, 模型间分歧 σ) | Spearman(相似度, \|err\|) | 最陌生 1/4 的 90% 覆盖 |
|---|---:|---:|---:|---:|
| NLL 单模型，random | −0.11 | — | −0.22 | 0.82 |
| NLL 集成，random | −0.16 | **−0.27** | −0.23 | 0.83 |
| faithful 集成，random | −0.18 | **−0.39** | −0.18 | 0.86 |
| NLL 单模型，scaffold | +0.01 | — | −0.17 | 0.74 |
| NLL 集成，scaffold | +0.04 | +0.12 | −0.18 | 0.86 |

- **单模型的 σ 几乎不随陌生程度变大**，误差却会：最陌生的四分之一，误差是最熟悉的 1.57 倍，σ 只大 1.10 倍。单模型 σ 主要反映数据噪声，而且跟着预测值走：预测越不溶，σ 越大。
- **random 上，集成的模型间分歧能识别陌生分子**：最陌生四分之一的分歧 σ 是最熟悉的 1.46 倍（NLL）/ 1.83 倍（faithful）。
- **scaffold 测试集内部方向反过来**：这里全部是没见过的骨架，相似度越低，分歧 σ 反而略小（Spearman +0.12，5/5 个 seed；最陌生四分之一是最熟悉的 0.78 倍）。
- **最陌生那部分仍偏过度自信**：90% 覆盖 0.74–0.86，要如实讲。

图：`figures/calibration_interpretability_2026-10-17/sigma_vs_similarity_nll_scaffold.png`（单模型，两种划分）

## 8. 提升性能，第二轮：只按验证集选，锁定后揭开一次测试集

**流程**：
- **方案**：几个 agent 分别从特征、模型/训练、集成/系统三个角度提方案，再由一个评审 agent 汇总，定下选择规则。
- **规则**：(i) 两种划分平均的验证集配对提升 ≤ −0.010；(ii) 任一划分不能变差超过 0.005；(iii) 至少 70% 的划分-seed 对有提升。
- **执行**：约 20 组方案、约 230 个 run，只看验证集。
- **锁定与揭晓**：决定写进 `calibration_runs/locked/lock.json` 后，用 `scripts/reveal_locked.py` 一次性揭开测试集。

**验证集筛选**（相对 6 描述符基线的配对变化）：

| 方案 | random | scaffold | 结果 |
|---|---:|---:|---|
| **+ Crippen logP（desc7）** | −0.015（4/5） | −0.084（5/5） | **通过并选中** |
| + logP + 芳香原子比例（desc8） | +0.006 | −0.094 | random 变差超过 0.005，拒绝 |
| 16 个描述符 | −0.010 | −0.055 | 通过，但两种划分之和不如 desc7 |
| dropout 0.2 / 0.3、权重 EMA、4 层、宽度 256、sum 池化、cosine 组合* | — | — | 全部拒绝 |

desc7 上的 2×2×2 组合（sum 池化 × 宽度 256 × cosine 组合*）按两种划分平均的主效应是 +0.007 / −0.002 / −0.005，全部拒绝。

\* "cosine 组合"同时改了 warmup+cosine 学习率、峰值学习率 1e-3、固定 100 epoch、不早停，所以测的是这一揽子改动，不能单独说明 cosine 调度没用。

**锁定后的测试集**（每种划分 5 个 seed，逐 seed 配对；集成成员是 faithful 模式，即均值用 MSE 训练）：

| 划分 | 模型 | 测试 RMSE | 相对参照的配对变化（更好的 seed 数，p） |
|---|---|---:|---|
| random | 6 描述符，单模型（原训练入口） | 0.556 ± 0.042 | 参照 |
| random | desc7，单模型（原训练入口） | 0.564 ± 0.047 | +0.007（1/5）：不比基线好 |
| random | desc7，单模型（faithful） | 0.549 ± 0.041 | 相对 6 描述符 faithful −0.013（4/5，p=0.07） |
| random | **desc7，5 成员集成（faithful）** | **0.543 ± 0.042** | 相对原训练入口基线 −0.014（5/5，p=0.01） |
| scaffold | 6 描述符，单模型（原训练入口） | 0.919 ± 0.018 | 参照 |
| scaffold | **desc7，单模型（原训练入口）** | **0.869 ± 0.015** | −0.050（5/5，p=0.02） |
| scaffold | desc7，单模型（faithful） | 0.860 ± 0.014 | 相对 6 描述符 faithful −0.072（5/5） |
| scaffold | **desc7，5 成员集成（faithful）** | **0.847 ± 0.008** | 相对原训练入口基线 −0.071（5/5） |

- **scaffold 上 logP 的提升真实迁移到了测试集**：只改一个描述符、用原训练入口，0.919 → 0.869。
- **random 上提升很小**：单模型在两种训练方式下结论相反，属于噪声。5 成员集成逐 seed 5/5 更好（−0.014），但按分子做的 bootstrap 区间是 [−0.037, +0.009]，跨 0（第 9.5 节）。
- **对照组**：
  - desc8 被拒，只是因为 random 变差 0.006，超过 0.005 门限。它在 scaffold 测试上不如 desc7（0.887 vs 0.860），但仍好于 6 描述符。
  - 16 个描述符：验证集 −0.010 / −0.055，测试集只有 −0.002 / −0.020，都不显著。验证集上的提升没有完整迁移。
- **混合模型**（desc7 集成 + 梯度提升树，权重在验证集上定为 0.15 / 0.10）：测试 0.539 / 0.842，只比集成好 0.004–0.005。它的特征组合在探索阶段被看过测试集，所以只算探索性结果，不作为主结论。
- **desc7 模型的归因**（faithful 模式；和第 4 节的 NLL 模式不完全可比，因为训练方式也变了）：
  - MolLogP 成为最重要的描述符（平均 |IG| 0.99 / 1.21），分子量降到 0.59 / 0.76。
  - 羟基 O、羰基 O、芳香 N 仍在 5/5 个 seed 上为正（random：+1.03 / +0.46 / +0.26）。
  - 碳原子的归因在 scaffold 上降到约 0（芳香 C +0.01、脂肪 C +0.01）。random 上脂肪 C 仍是 −0.06，非芳香 sp2 C 仍是 −0.14。
  - 一种解释是疏水骨架的作用部分被 logP 描述符接管了。

## 9. 第三轮：继续提升性能（选择只用验证集，再用全新划分确认）

**为什么流程要变**：原始测试集在第二轮已经看过一次。如果再调几轮还用它来挑最好的方案，数字会越来越乐观，等于对测试集调参。第三轮的规则：
- **选择**：只用验证集。
- **验证集判断不了的决定**（比如训练集 + 验证集重训）：用**全新的确认划分**来判断。一种是 random seed 6–10 的 5 个新划分；另一种是 5 个 chemprop 风格的随机骨架划分（`scripts/make_confirmation_splits.py`）。
- **原始测试集**：最后只再看一次，并标明是"第二次查看"。

**规则落盘的时间要说清楚**：规则写进 `calibration_runs/locked/round3_plan.json` 的时间是 2026-10-18T03:22。
- 9.1 的 17 组单模型搜索是在这之前跑完的，沿用第二轮 `lock.json` 的规则，plan 只记录了它的结果。
- 9.2 的候选集成和选择规则，是在一个只看验证集的 D-MPNN / AttentiveFP 试跑之后定的。那次试跑已经显示含 D-MPNN 的集成在验证集上更好。所以验证集选择的独立性有限，选择是否正确，要以确认集上和 GINE × 5 的比较为准（9.4）。
- 重训规则和混合规则，都是在各自的结果出来之前定的。

### 9.1 单模型调参已到头

在 desc7 上又搜了 17 组（170 个 run，只看验证集，和 desc7 逐 seed 配对），**没有一组通过规则**：
- 最好的是残差加融合，平均验证集提升 0.007，门槛是 0.010；
- concat 融合、2 层、更长训练、weight decay、更小 dropout 都在 ±0.005 以内；
- 再加描述符（MolMR、FractionCSP3、氢键受体、芳香环数、BertzCT）都让 random 变差 0.016–0.019；
- 学习率 1e-3 / 1e-4、batch 32、宽度 64 都更差，FiLM 融合明显更差。

结论：在"GINE + 描述符"这个框架里，desc7 已经接近局部最优。

### 9.2 跨架构集成

移植了 D-MPNN（有向边消息传递，排除反向边，来自 MolFuse）和 AttentiveFP（PyG 的原子阶段）作为可替换的编码器（`chempath_gine/extensions/encoders.py`），输入特征和读出头都和 GINE 相同。

| 单模型（验证 RMSE，成员 0–4 平均） | random | scaffold |
|---|---:|---:|
| GINE（desc7） | 0.538 | 0.786 |
| D-MPNN | 0.541 | 0.768 |
| AttentiveFP | 0.551 | 0.783 |

3 个候选集成，按两种划分验证 RMSE 之和选：

| 候选（验证 RMSE） | random | scaffold | 和 |
|---|---:|---:|---:|
| **GINE × 5 + D-MPNN × 5** | 0.510 | 0.749 | **1.259（选中）** |
| GINE + D-MPNN + AttentiveFP 各 5 个 | 0.513 | 0.748 | 1.261 |
| GINE × 10 | 0.513 | 0.760 | 1.273 |

- **为什么混合集成更好**：
  - random 上，不同架构之间的验证残差相关更低（GINE 与 D-MPNN 0.873，GINE 之间 0.914）。
  - scaffold 上几乎没有这种去相关（0.922 vs 0.931），这里的提升主要来自 D-MPNN 本身在 scaffold 上更准（0.768 vs 0.786）。
- **原始测试集上不明显**：GINE × 10 和 GINE × 5 + D-MPNN × 5 分别是 0.539 vs 0.541（random）、0.844 vs 0.840（scaffold）。确认集上没有 GINE × 10 这一组。
- **不加梯度提升树**：在选中的集成上混合梯度提升树，验证集权重只有 0.10 / 0.05，改善 0.0015 / 0.0007，最终方案不包含它。

### 9.3 训练集 + 验证集重训

每个成员先在训练集上训练，用验证集选出最佳 epoch。然后在训练集 + 验证集上重新训练同样多的 epoch（训练数据多 21%），**逐 epoch 重放原来的学习率序列**（`run_calibration.py --refit-from`），scaler 仍只在训练集上拟合。

验证集判断不了这个决定，所以在确认划分上判断。事先规则：两类确认集平均都更好，且至少 7/10 个划分更好。

| 确认集 | 只用训练集 | 重训 | 差值 | 重训更好 |
|---|---:|---:|---:|---:|
| random seed 6–10 | 0.529 | 0.516 | −0.013 | 4/5 |
| 随机骨架划分 1–5 | 0.791 | 0.767 | −0.024 | 5/5 |

结果 9/10，**采纳**。代价是重训模型没有留出数据，它自己的区间不准（确认集上 90% 覆盖只有 0.71，conformal 0.69–0.70）。所以最终系统的不确定度来自同一套系统的"只用训练集"版本：用验证集校准，区间以重训集成的 μ 为中心。

### 9.4 在确认划分上的结果

| 模型 | random seed 6–10 | 随机骨架划分 1–5 |
|---|---:|---:|
| 6 描述符基线（原训练入口） | 0.566 ± 0.043 | 0.893 ± 0.110 |
| desc7 单模型（faithful） | 0.566 ± 0.035 | 0.855 ± 0.104 |
| desc7 GINE × 5 集成（第二轮的系统） | 0.538 ± 0.029 | 0.808 ± 0.113 |
| GINE × 5 + D-MPNN × 5（只用训练集） | 0.529 ± 0.032 | 0.791 ± 0.112 |
| **最终系统（重训）** | **0.516 ± 0.030** | **0.767 ± 0.118** |

按分子做配对 bootstrap（10,000 次，`scripts/bootstrap_compare.py`）：

| 比较（确认集） | random | 随机骨架 |
|---|---|---|
| 只用训练集的系统 − 基线 | −0.038 [−0.057, −0.019]，5/5 | −0.103 [−0.130, −0.077]，5/5 |
| 最终系统（重训）− 基线* | −0.051 [−0.073, −0.028]，5/5 | −0.127 [−0.158, −0.097]，5/5 |
| GINE × 5 + D-MPNN × 5 − GINE × 5（第三轮增量：跨架构） | −0.009 [−0.016, −0.002]，5/5 | −0.017 [−0.024, −0.011]，5/5 |
| 最终系统 − GINE × 5（第三轮增量：跨架构 + 重训）* | −0.022 [−0.035, −0.009]，5/5 | −0.042 [−0.058, −0.026]，5/5 |
| desc7 单模型 − 基线 | −0.001 [−0.021, +0.021]，3/5 | −0.039 [−0.070, −0.008]，3/5 |

\* 重训是在这些确认集上被采纳的，所以带 * 的行略乐观；"只用训练集的系统"那一行是更干净的估计。desc7 单模型在随机骨架上只有 3/5 个划分更好，平均值主要由 seed 5（−0.174）拉动。

**最终系统的区间**（重训 μ + 只用训练集版的校准 σ）：random 90% 覆盖 0.914，宽度 1.64 logS；随机骨架划分只覆盖 0.85，分布偏移下区间偏窄。

### 9.5 原始测试集（第二次查看，系统已冻结）

| 模型 | random | scaffold |
|---|---:|---:|
| 简历上的数字（存档） | 0.559 ± 0.046 | 0.929 ± 0.024 |
| 6 描述符基线（原训练入口，CPU 重跑） | 0.556 ± 0.042 | 0.919 ± 0.018 |
| desc7 单模型（faithful） | 0.549 ± 0.041 | 0.860 ± 0.014 |
| desc7 GINE × 5 集成（第二轮锁定） | 0.543 ± 0.042 | 0.847 ± 0.008 |
| GINE × 5 + D-MPNN × 5（只用训练集） | 0.541 ± 0.040 | 0.840 ± 0.007 |
| **最终系统（重训）** | **0.531 ± 0.039** | **0.813 ± 0.007** |

按分子做 bootstrap（scaffold 的 5 个 seed 共用同一个测试集，所以按分子重抽样，不把 seed 当独立数据）：

| 比较 | random | scaffold |
|---|---|---|
| 最终系统 − 6 描述符基线 | **−0.025 [−0.050, −0.001]**，5/5 | **−0.106 [−0.178, −0.033]**，5/5 |
| 最终系统 − 第二轮 desc7 集成 | −0.012 [−0.024, 0.000]，4/5 | −0.034 [−0.081, +0.010]，5/5 |
| 第二轮 desc7 集成 − 基线 | −0.014 [−0.037, +0.009]，5/5 | −0.071 [−0.123, −0.020]，5/5 |

**最终系统的不确定度**：90% 区间覆盖 0.891（random）/ 0.906（scaffold），宽度 1.58 / 2.73 logS。确认集上是 0.914 / 0.85（9.4）。

**解读**：
- 最终系统相对基线的提升，两种划分上置信区间都在 0 以下。
- 第三轮相对第二轮的增量，在原始测试集上方向基本一致（random 4/5、scaffold 5/5），但只有 168–170 个分子，区间碰到或跨过 0。
- 在确认集上，同样的增量是显著的：−0.022 [−0.035, −0.009] / −0.042 [−0.058, −0.026]。

**最终系统的定义**：
- **输入**：desc7 特征，即 6 个 Delaney 描述符加 Crippen logP。
- **成员**：5 个 GINE 和 5 个 D-MPNN，读出头和融合方式相同。
- **训练**：每个成员先在训练集上训练、用验证集选 epoch，再在训练集 + 验证集上重训同样的 epoch 数。
- **点预测**：10 个重训成员的平均。
- **90% 区间**：μ ± 1.645·s·σ。σ 来自 10 个只用训练集训练的成员（混合矩），s 在验证集上拟合。

## 10. 局限与披露

1. **原始测试集看过两次**：第二轮锁定后一次（2026-10-18T01:58），第三轮冻结后一次（04:17）。另外：
   - 第二轮之前，16 描述符的测试 RMSE（0.563 / 0.932）被提前看过。
   - 一位设计 agent 探索时，算过 6 描述符集成和梯度提升树混合的测试集结果，所以混合模型只算探索性结果。
   - 第二轮的训练日志会打印测试 RMSE（之后已改为只打印验证集）。
   - 第三轮的 `r3/ensembles/aggregate.csv` 在选择前 19 秒就写入了三个候选的原始测试 RMSE。脚本只打印验证集，选择也和测试数字的方向相反：GINE × 10 在 random 测试上其实更好。
   - 所以"锁定前没看测试集"是流程承诺，没有强制机制。
2. **确认集不是全新分子，而且重训决定是在它上面做的**：
   - 随机骨架测试集和原 scaffold 测试集共享 39–71 个分子，占每个确认测试集的 23–42%。
   - random seed 6–10 的测试分子里，51–62% 出现在原始 random 测试集中，57–66% 出现在 seed 1–5 的验证集中，而验证集参与过选择。
   - 所以确认集结果会略乐观，带重训的行更乐观。
   - 随机骨架划分的 5 个 seed 之间差别很大（基线标准差 0.11）。
3. **第三轮规则的落盘时间**：17 组搜索早于 plan 文件；候选集成和规则是在一次只看验证集的 D-MPNN 试跑之后定的（见第 9 节开头）。
4. **验证集被用了很多次**：早停、选 checkpoint、选配置、方差头早停、混合权重、校准。验证集数字偏乐观，conformal 的覆盖保证也只是近似。
5. **scaffold 统计**：原 scaffold 划分是确定性的，5 个 seed 只反映初始化噪声，最终比较改用按分子的 bootstrap。
6. **数值复现**：
   - CPU 上 GINE 的结果在线程数固定（2）时逐位可复现。
   - D-MPNN 不是逐位可复现的：同样的命令重跑会得到不同的训练轨迹，可能和机器负载有关。所以最终系统的 D-MPNN 成员只能在统计意义上复现。
   - 存档的简历数字很可能是 CUDA 上跑的，和本次 CPU 重跑每个 seed 最多差 0.028。
7. **重训模型没有自己的校准**：区间来自只用训练集训练的同一系统，以重训 μ 为中心。这个组合在确认集和原始测试集上都检验过，分布偏移下偏窄。
8. **只做了 ESOL**：Lipophilicity 和 FreeSolv 完全没做本报告里的实验。
9. **环境问题**：
   - 项目在 iCloud 同步的 `~/Documents` 下，磁盘用到 96%，macOS 把大部分文件移出了本地，读取时会超时或读成空文件。
   - 13 个新脚本里，只有 3 个（`run_interpretability`、`bootstrap_compare`、`make_confirmation_splits`）带按文件大小校验的重试。集成脚本会检查成员的分子和标签一致，能挡住大部分截断。失败的步骤都重跑过。
   - `calibration_runs/` 里还留有 148 个 iCloud 冲突副本（文件名带 " 2"），所有脚本都只读不带后缀的文件。

## 11. 面试怎么讲（中文要点）

**30 秒版本**：
> 主模型是图 + 描述符融合的 GINE。我先把回归头换成高斯 NLL 输出均值和方差：random 上 RMSE 不变，ECE 从 0.07 降到 0.03。scaffold 上单模型过度自信，用 5 个模型的深度集成把 90% 区间覆盖从 0.83 恢复到 0.90。
>
> 提升性能这块，我用验证集按事先定好的规则筛了 30 多组改动。真正有用的是四件事：
> - 把 Crippen logP 加回描述符：原来 6 个描述符正好漏掉了 Delaney 溶解度方程里最重要的一项；
> - 深度集成；
> - 加 D-MPNN 做跨架构集成；
> - 用训练集 + 验证集重训。
>
> 最后在原始测试集上，scaffold 从 0.93 降到 0.81，random 从 0.56 降到 0.53。在另外生成的确认划分上也检验了：只用训练集的版本比基线好 0.04 / 0.10，bootstrap 置信区间完全在 0 以下。

**一定会被追问、要主动讲的点**：
- **你怎么保证没对测试集调参**：
  - 所有选择只看验证集，规则写进文件后才揭开测试集。
  - 第三轮因为原测试集已经看过一次，所以另做了确认划分，重训这种验证集判断不了的决定就在确认集上做。
  - 最后用按分子的 bootstrap 给置信区间。
  - 要主动说的不完美之处：原测试集看过两次；第二轮之前提前看过 16 描述符的测试结果；确认集和原测试集有部分分子重叠；第三轮的候选是在验证集试跑之后定的。讲清楚这些比数字好看更加分。
- **验证集提升不一定迁移**：
  - 16 个描述符在验证集上好了 0.010 / 0.055，测试集上只有 0.002 / 0.020，都不显著。
  - desc8（加芳香原子比例）因为 random 验证变差 0.006 被规则拒掉，测试集上它也不如 desc7。
  - 验证集只有 169 个分子，噪声很大，所以规则要求两种划分都不变差，而且多数 seed 都更好。
- **logP 为什么有用**：
  - Crippen logP 是按原子贡献加和的疏水性，在 Delaney 方程里贡献最大。
  - scaffold 测试集的分子更大（平均 18.6 个重原子，训练集 11.6）；logP 的均值几乎没变（2.52 vs 2.48），但分布更宽。
  - logP 是可加和的物理量，比图模型从局部消息传递学到的东西更容易外推到大分子。
- **集成为什么有用、跨架构为什么还有一点额外收益**：
  - 集成的主要收益来自平均掉初始化噪声（第二轮 GINE × 5 已经拿到大部分）。
  - random 上，D-MPNN 和 GINE 的误差相关更低（0.87 vs 0.91）。
  - scaffold 上，额外收益主要因为 D-MPNN 单模型更准。
  - 确认集上，加 D-MPNN 比 GINE × 5 好 0.009 / 0.017。
- **重训的取舍**：
  - 训练数据多 21%，scaffold 验证集里的分子也更像测试集，确认集上 9/10 个划分更好。
  - 但重训后没有留出数据做校准，所以区间来自只用训练集训练的那套模型，中心换成重训的 μ。
- **ECE 怎么算、conformal 是什么**：
  - ECE：19 个名义覆盖率上，实际覆盖和名义值之差的平均。
  - conformal：在验证集上取 |误差|/σ 的分位数来定区间宽度，有限样本下覆盖率有保证，但前提是验证集没被拿来选模型，所以我们这里只是近似。
  - random 上异方差 σ 让区间窄了 0.18 logS，scaffold 上反而更宽。
- **注意力 vs 积分梯度**：
  - 同口径比较，逐原子遮挡时 |IG| 和预测变化的相关是 0.47–0.50，注意力是 0.24–0.30。
  - 遮掉排第一的原子，IG 选的是随机的 2.2–2.7 倍，注意力是 1.9–2.1 倍。
  - IG 更可靠，注意力也比随机强。读出层的注意力池化说明的是读出时看哪里，不说明正负贡献。
- **和 Crippen 的一致性只是弱的**：
  - 原子级相关约 0.2，分子级约 0.4。
  - 羟基、芳香碳、Cl 等方向一致；F、羰基 O、硝基等不一致。
  - 要说"弱一致"，不要说"验证了化学规律"。
- **σ 能不能识别没见过的分子**：
  - 单模型 σ 基本不能。
  - 集成的模型间分歧在 random 上能（最陌生的四分之一高 1.5–1.8 倍），在 scaffold 测试集内部不能。
  - 最陌生那部分仍偏过度自信（覆盖 0.74–0.86）。
- **失败案例**：
  - 1-十八醇被高估溶解度：模型给 OH 的功劳和苯酚里差不多，可能是没学到长链会稀释它。这是归因给出的线索，不是因果证明。
  - 磷胺（有机磷农药）被严重低估：它和训练集最相似的分子也只有 0.26，σ 只略高于平均，误差却是 3σ，是"自信地错了"的例子。
- **数据细节**：
  - 最终模型的描述符是 6 个 Delaney 列加 Crippen logP。
  - 简历上的 0.56 / 0.93 是 6 个描述符的单模型；0.53 / 0.81 是 10 个模型的集成系统，再加重训。如果简历要用新数字，要写清楚是集成系统。

## 12. 复现与文件

所有命令都在 `ChemPath-GINE/` 下运行，Python 环境是 `~/.venvs/gnn-transformer`。训练和归因脚本（`run_calibration`、`run_interpretability`、`validate_attributions`）统一加 `--threads 2`；其他汇总脚本没有这个参数。

```bash
# 第一轮：校准（mode = mse | nll | beta_nll | faithful）
python scripts/run_calibration.py --config configs/esol_fusion_ablation_random.yaml --mode nll --seeds 1,2,3,4,5 --out calibration_runs/esol_random --threads 2
python scripts/summarize_calibration.py --roots calibration_runs/esol_random,calibration_runs/esol_scaffold --out calibration_runs/summary --fig-dir ../reports/figures/calibration_interpretability_2026-10-17
python scripts/run_interpretability.py --runs-root calibration_runs/esol_random --mode nll --fig-dir ../reports/figures/calibration_interpretability_2026-10-17/esol_random_interpretability --out calibration_runs/interpretability/esol_random

# 第二轮：集成成员（init seed = 1000·seed + m）、集成 + conformal、归因验证、新颖度分析
python scripts/run_calibration.py --config configs/esol_fusion_ablation_random.yaml --mode nll --seeds 1,2,3,4,5 --members 1,2,3,4 --out calibration_runs/esol_random --threads 2
python scripts/build_ensembles.py --roots calibration_runs/esol_random,calibration_runs/esol_scaffold --out calibration_runs/ensembles --fig-dir ../reports/figures/calibration_interpretability_2026-10-17
python scripts/validate_attributions.py
python scripts/sigma_vs_similarity.py --runs-root calibration_runs/esol_scaffold --mode nll --out calibration_runs/similarity/nll_scaffold --tag nll_scaffold

# 第二轮调参：只看验证集（--show-test 只在锁定后用一次）
sh calibration_runs/perf/run_arms.sh "desc7_logp desc16 ..." "1 2 3 4 5" 5
python scripts/summarize_perf.py
python scripts/run_hybrid_blend.py --phase select ...
python scripts/reveal_locked.py --lock calibration_runs/locked/lock.json

# 第三轮：编码器成员、跨架构集成、确认划分、重训、最终评估
python scripts/build_mixed_ensembles.py --split-roots random,scaffold --system "gine5_dmpnn5=...;..." --out calibration_runs/r3/ensembles
python scripts/make_confirmation_splits.py
sh calibration_runs/confirm/evaluate_confirm.sh with_refit
python scripts/run_calibration.py --config calibration_runs/perf/configs/desc7_dmpnn_random.yaml --mode faithful --seeds 1 --members 0 --out calibration_runs/r3/refit/desc7_dmpnn/esol_random --refit-from calibration_runs/perf/desc7_dmpnn/esol_random:faithful --threads 2
python scripts/final_system_eval.py --refit calibration_runs/r3/final/gine5_dmpnn5_refit/esol_random:gine5_dmpnn5_refit --trainonly calibration_runs/r3/final/gine5_dmpnn5/esol_random:gine5_dmpnn5 --out calibration_runs/r3/final_system_random.json
python scripts/bootstrap_compare.py --a ROOT:MODE --b ROOT:MODE --seeds 1,2,3,4,5 --out FILE.json
```

**新增代码**（全部是新文件；原训练入口 `chempath_gine/training/train.py` 和 `data/split.py` 没有改动）：
- `chempath_gine/uncertainty/`：`gaussian.py`（μ, log σ² 头，faithful 模式，可接其他编码器）、`calibration.py`（NLL、ECE、覆盖率、验证集缩放、conformal）、`attribution.py`（嵌入层 IG、注意力池化权重、对称类平均）。
- `chempath_gine/extensions/`：`pooling.py`（sum 池化）、`encoders.py`（D-MPNN、AttentiveFP），以及 `apply_model_extensions` 入口。
- `scripts/`：
  - 训练与集成：`run_calibration.py`、`build_ensembles.py`、`build_mixed_ensembles.py`。
  - 汇总与分析：`summarize_calibration.py`、`run_interpretability.py`、`validate_attributions.py`、`sigma_vs_similarity.py`、`summarize_perf.py`。
  - 锁定、确认与评估：`run_hybrid_blend.py`、`reveal_locked.py`、`make_confirmation_splits.py`、`final_system_eval.py`、`bootstrap_compare.py`。
- `tests/`：`test_uncertainty.py`、`test_extensions_encoders.py`、`test_bootstrap_compare.py`。全套测试通过（78 个）。

**结果文件**：
- 第一轮：`calibration_runs/esol_{random,scaffold}/`、`calibration_runs/summary/`、`calibration_runs/interpretability/`。
- 第二轮：`calibration_runs/ensembles/`、`calibration_runs/similarity/`、`calibration_runs/interpretability/validation/`、`calibration_runs/perf/`、`calibration_runs/locked/{lock.json,reveal.json,reveal.md}`。
- 第三轮：
  - 规则与决定：`calibration_runs/locked/{round3_plan,round3_decision,round3_final_look}.json`。
  - 集成与最终结果：`calibration_runs/r3/{ensembles,refit,final}/`、`calibration_runs/r3/final_system_*.json`、`calibration_runs/r3/bootstrap_*.json`。
  - 确认集：`calibration_runs/confirm/`（划分、运行、评估、bootstrap，以及 `eval_extra/` 里的 GINE × 5 行）。
- 图：`GNN-Transformer/reports/figures/calibration_interpretability_2026-10-17/`。
