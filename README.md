# EIS-PEM ShanHaiJue 山海决

Routed multi-estimator identification and risk-controlled
certification for lithium-ion battery EIS parameter recovery on a
48-parameter DFN-like SEIS forward model.

This is the next version of
[eis-pem-identification](https://github.com/eric-haotian/eis-pem-identification), codename
**ShanHaiJue**. The forward model, the parameter machinery, and the identifiability
diagnostics carry over unchanged; what is new is the layer above them.

learning AI at **www.haotianblog.com**

## What Changed

The previous version screened identifiability once, estimated the free subset, and
reported the rest as fixed reference assumptions. It stated the boundary honestly — a good
impedance fit does not imply reliable physical parameters — but it answered that boundary
with a single estimator and a single selection.

ShanHaiJue answers it with a population and a budget:

- **An ensemble along the prior-strength axis.** Arms differ only in the prior scale λ and
  in initialization. At λ = 1 the prior stabilises the flat directions of the sloppy cost
  valley at the price of bias on the data-determined directions; as λ → 0 that bias
  vanishes but the flat directions destabilise. The regimes therefore fail on *different*
  coordinates (correct-set Jaccard index 0.515), so a per-coordinate oracle over eight arms
  reaches 35.7 of 48 against 17.1–18.5 for any single member.
- **Per-coordinate routing on ground-truth-free evidence.** A gradient-boosted ranker maps
  ten diagnostics — none of which requires knowing the answer — to a claim probability per
  (arm, coordinate), and each coordinate is taken from its highest-scoring arm. Truth labels
  train the ranker offline; deployment consumes none.
- **Certification under a stated false-claim budget.** The claim threshold is the smallest
  one whose false claims per cell respect a preregistered budget at a 95 % upper confidence
  bound, bootstrapped over generating truths rather than over rows. Abstention is the
  default state: a coordinate below the threshold is simply not claimed.

| Quantity | ShanHaiJue | Previous version |
|---|---|---|
| Recovered coordinates per cell (of 48) | **27.8** | 17.1 single-estimator baseline |
| Certified at realized precision 0.960 | **10.38** | no certification layer |
| Certified at realized precision 0.968 | 8.48 | — |
| Evidence used at inference | ten ground-truth-free diagnostics | identifiability screen |
| Cost per cell | 23 bounded fits + 1 Jacobian | 1 fit |

Recovery is validated on three never-before-generated panels; every certification rung was
validated on its own panel, analysed exactly once, never pooled.

## Method Boundary

The previous version's boundary statement still holds, and the certification layer adds two
of its own. Read these before quoting any number above.

- All quantities are **synthetic-panel** estimates within the calibration generator class.
  Transfer to measured cells is not validated: the out-of-distribution guard withheld
  calibrated labels from all 87 real commercial-cell spectra tested.
- "Certified" means certified under a **stated empirical risk budget**, not a finite-sample
  guarantee. The cluster bootstrap is anti-conservative at ~30 clusters (measured one-sided
  coverage 0.88–0.93 against nominal 0.95), so read budgets with a 2–7 percentage-point
  haircut.
- **Identification runs on any grid; certification does not.** The measured conditions go to
  the forward model verbatim, so a single spectrum, a scattered set of temperature/SOC
  pairs, and a full 5 × 5 grid all produce estimates for all 48 coordinates. The calibrated
  probabilities and the claim threshold were fitted at one information content, so on a
  different grid the claims are point estimates until you re-freeze and re-validate.
- **Freeze before you validate.** Reserve the validation seeds at freeze time, generate the
  validation panel afterwards, analyse it once, and never pool panels. Re-tuning on a
  validation panel silently destroys the meaning of every number above.
- The best operating point (10.38 @ 0.960) is a maximum over nine rungs selected by
  sequential dose-finding and carries the corresponding selection optimism. 

## The Two Editions

Identical science, different solver. The pipeline package, the ten diagnostics, the router,
the certifier, and the frozen validated system are byte-identical between them; they differ
only in how the ensemble is fitted.

| | [`cpu/`](cpu/) | [`gpu/`](gpu/) |
|---|---|---|
| Estimator ensemble | scipy trust-region-reflective, multiprocessing across cells | batched JAX/CUDA Levenberg-Marquardt with SVD-damped steps |
| Requires | Python 3.9+ only | a CUDA device, ~8 GB VRAM at the default batch size |
| Cost per cell | ~48 s | ~2 s per fit, ~100× single-fit throughput |
| Desktop interface | yes | yes |
| macOS app | yes, see below | — |

Pick `cpu/` unless you are generating panels. Panel generation (60 cells) is the only step
where the GPU edition changes what is practical: ~45 minutes instead of hours.

## Install and Run

```bash
git clone https://github.com/eric-haotian/eis-pem-shanhaijue.git
cd eis-pem-shanhaijue/cpu
pip install -r requirements.txt
python3 bin/certify.py --system artifacts/validated_system --budget 0.60 --demo
```

The demo simulates one fresh cell, fits the eight-arm ensemble, routes every coordinate,
and applies the frozen certifier, printing the certified claims with their calibrated
probabilities and marking which were correct. It runs the full 25 × 120 grid on one core,
so budget **15–20 minutes** (measured: 16 min on an M-series MacBook Air). To check the
installation instead, `python3 bin/selftest.py` exercises every stage on a deliberately
small grid in about 90 seconds.

For the GPU edition, `cd gpu` instead and add the CUDA build of JAX that matches your
driver:

```bash
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"
python3 -c "import jax; print(jax.default_backend(), jax.devices())"   # expect: gpu
```

Each edition's own README documents the full freeze-and-validate workflow, the desktop
interface, and the deployment API.

## Desktop Interface

Double-click **`启动界面_Start_GUI.command`** (macOS) or **`启动界面_Start_GUI.bat`**
(Windows) in either edition. The launcher finds Python, installs anything missing from
`requirements.txt`, and opens the window. The interface switches between Chinese and
English at any time.

```text
input CSV -> ensemble fit -> routing -> certification -> XLSX (4 sheets)
```

The window reports which backend it is running on, marks every certification cell *out of
calibration* when the input grid differs from the frozen system's, and records in the log
every inference the CSV reader had to make.

## macOS App

The CPU edition packages into a double-clickable `.app`:

```bash
cd cpu
./script/package_macos_app.sh
open "dist/EIS-PEM ShanHaiJue.app"
```

The bundle carries the pipeline, the frozen validated system, the reference CSVs, and an
embedded Python runtime, so it launches without the user installing anything. It also
exposes each `bin/` script as a command-line entry inside `Contents/MacOS/`, and
`Contents/Resources/PipelineManifest.txt` records the exact git state, frozen system, and
package versions that went in.

Like the previous version's `package_macos_app.sh`, this is a **local-machine bundle**: the
embedded runtime is a virtual environment built from the packaging machine's Python. A
cross-machine distribution still needs a frozen runtime plus Developer ID signing and
notarization.

## CSV Contract

The canonical format is five columns, one row per frequency, each condition's spectrum
concatenated head to tail:

```text
Temperature_K,SOC,Frequency_Hz,Re_Z,Im_Z
278.15,0.1,0.001,0.019984408733,-0.005062697
```

`examples/calibration_grid_25x120.csv` is that format on the grid the frozen system was
calibrated on, so it is both the format reference and the acquisition design that carries
the certification guarantee. `examples/format_sample.csv` is a 30-row version.

Other layouts are accepted: only frequency and complex impedance are genuinely required.
Header aliases (including Chinese ones), bracketed units, sniffed delimiters, headerless
numeric files, `|Z|`+phase input, and °C-or-K and 0–1-or-0–100 auto-detection are all
handled, and every inference is reported in the log and the workbook. See each edition's
README for the full table.

## Repository Layout

```text
cpu/                       CPU edition — the reference implementation
gpu/                       GPU edition — same science, batched JAX/CUDA solver
  shanhaijue/              panel, arms, features, router, certifier, pipeline
    vendor/eispem/         the 48-parameter physics layer, 35 modules, one flat namespace
  gui/                     desktop interface
  bin/                     panel generation, routing, freezing, validation, certification
  artifacts/               the frozen validated system behind the numbers above
  examples/                reference CSVs
cpu/script/                macOS app packaging
```

The two trees differ in exactly six files — `README.md`, `requirements.txt`,
`shanhaijue/__init__.py`, `shanhaijue/arms.py`, `bin/run_panel.py`, `bin/selftest.py` —
plus `cpu/script/`, which has no GPU counterpart. Everything else, the vendored physics
layer and the frozen artifact included, is byte-identical.

The previous version, its benchmark comparisons, and the research
history are at
[eric-haotian/eis-pem-identification](https://github.com/eric-haotian/eis-pem-identification).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

#以下为中文

# EIS-PEM ShanHaiJue 山海决

面向锂离子电池 EIS 参数恢复的**路由式多估计器辨识**与**风险受控认证**方法，基于一个包含 48 个参数的类 DFN SEIS 正向模型。

这是 [eis-pem-identification](https://github.com/eric-haotian/eis-pem-identification) 的下一代版本，代号为 **ShanHaiJue（山海决）**。正向模型、参数体系和可辨识性诊断均保持不变；真正新增的是构建于这些基础之上的上层方法体系。

AI 学习与交流：**www.haotianblog.com**

## 主要变化

上一版本首先执行一次可辨识性筛选，然后估计可自由辨识的参数子集，并将其余参数报告为固定参考假设。它如实指出了一个关键边界：

> 良好的阻抗拟合并不意味着物理参数能够被可靠恢复。

但上一版本只使用单一估计器和单一参数选择方案来处理这一问题。

ShanHaiJue 则通过一个估计器群体和一个风险预算来应对这一边界：

- **沿先验强度轴构建估计器集成。**  
  各估计器分支仅在先验尺度 λ 和初始化方式上有所不同。当 λ = 1 时，先验能够稳定病态代价谷中的平坦方向，但会对由数据充分决定的方向引入偏差；随着 λ → 0，这种偏差逐渐消失，但平坦方向也会重新变得不稳定。因此，不同估计器会在不同参数坐标上失败。各估计器正确坐标集合之间的 Jaccard 指数为 0.515；从 8 个估计器中逐坐标选取最优结果的 oracle，可恢复 48 个坐标中的 35.7 个，而任一单独估计器只能恢复 17.1–18.5 个。

- **依据无需真实值的证据进行逐坐标路由。**  
  一个梯度提升排序器将 10 项诊断指标映射为每个“估计器分支—参数坐标”组合的声明概率。这些诊断指标均不需要知道真实答案。每个参数坐标最终采用评分最高的估计器分支。真实值标签仅用于离线训练排序器；部署阶段不需要真实值。

- **在明确的错误声明预算下实施认证。**  
  声明阈值被定义为满足以下条件的最小阈值：在 95% 上置信界下，每个电芯的错误声明数不超过预注册预算。该置信界通过对生成真值进行聚类 Bootstrap 获得，而不是对单独数据行进行 Bootstrap。系统默认选择弃权：当某个坐标的概率低于阈值时，系统不对其作出声明。

| 指标 | ShanHaiJue | 上一版本 |
|---|---:|---:|
| 每个电芯恢复的参数坐标数（共 48 个） | **27.8** | 单估计器基线为 17.1 |
| 在实际精确率 0.960 下通过认证的坐标数 | **10.38** | 无认证层 |
| 在实际精确率 0.968 下通过认证的坐标数 | 8.48 | — |
| 推理阶段使用的证据 | 10 项无需真实值的诊断指标 | 可辨识性筛选 |
| 每个电芯的计算成本 | 23 次有界拟合 + 1 次 Jacobian 计算 | 1 次拟合 |

恢复性能在 3 个此前从未生成过的全新面板上进行了验证；每一级认证阶梯均在其独立面板上验证，每个面板只分析一次，不同面板之间从不合并。

## 方法边界

上一版本所声明的方法边界依然成立，而认证层又增加了两个新的边界。在引用上述任何数字之前，请先阅读以下内容。

- 所有指标均为**校准生成器类别内部的合成面板估计结果**。  
  该方法向真实测量电芯的迁移能力尚未得到验证：在所测试的 87 条真实商用电芯阻抗谱上，分布外检测机制均拒绝输出经过校准的标签。

- “经过认证”是指在一个**明确规定的经验风险预算**下通过认证，而不是获得有限样本意义下的理论保证。  
  当聚类数量约为 30 时，聚类 Bootstrap 存在反保守性。其实际单侧覆盖率为 0.88–0.93，低于名义上的 0.95。因此，在解读风险预算时，应保守地折减约 2–7 个百分点。

- **参数辨识可以运行在任意采样网格上，但认证不可以。**  
  测得的实验条件会原样传入正向模型。因此，无论输入是单条阻抗谱、零散的温度/SOC 条件组合，还是完整的 5 × 5 网格，系统都可以为全部 48 个参数坐标给出估计值。但是，校准概率和声明阈值是在特定信息量条件下拟合得到的。当输入网格发生变化时，在重新冻结并重新验证之前，相应声明只能作为点估计结果解读。

- **必须先冻结，再验证。**  
  应在冻结阶段预先保留验证随机种子，随后生成验证面板，只分析一次，并且不得合并不同验证面板。若在验证面板上重新调参，就会在不易察觉的情况下破坏上述所有数值的统计含义。

- 最佳工作点“10.38 个认证坐标、实际精确率 0.960”是从 9 个认证阶梯中，通过序贯剂量探索选出的最大值，因此包含相应的选择乐观偏差。

## 两个版本

两个版本的科学方法完全相同，仅使用不同的求解器。流水线软件包、10 项诊断指标、路由器、认证器以及冻结后的已验证系统，在两个版本之间均逐字节一致；差异仅在于估计器集成的拟合方式。

|  | [`cpu/`](cpu/) | [`gpu/`](gpu/) |
|---|---|---|
| 估计器集成 | SciPy 信赖域反射算法，并在不同电芯之间进行多进程并行 | 基于 JAX/CUDA 的批量 Levenberg–Marquardt，采用 SVD 阻尼步 |
| 系统要求 | 仅需 Python 3.9+ | CUDA 设备；默认批量大小下约需 8 GB 显存 |
| 每个电芯的计算成本 | 约 48 秒 | 每次拟合约 2 秒；相对于单次拟合约有 100 倍吞吐量 |
| 桌面界面 | 支持 | 支持 |
| macOS 应用 | 支持，见下文 | — |

除非需要生成大规模面板，否则应优先选择 `cpu/`。

生成包含 60 个电芯的面板，是 GPU 版本唯一会显著改变计算可行性的步骤：GPU 版本约需 45 分钟，而 CPU 版本通常需要数小时。

## 安装与运行

```bash
git clone https://github.com/eric-haotian/eis-pem-shanhaijue.git
cd eis-pem-shanhaijue/cpu
pip install -r requirements.txt
python3 bin/certify.py --system artifacts/validated_system --budget 0.60 --demo
```

该演示程序会模拟一个新的电芯，拟合包含 8 个分支的估计器集成，对每个参数坐标进行路由，并应用冻结后的认证器。程序会输出经过认证的参数声明及其校准概率，同时标明这些声明是否正确。

演示使用完整的 25 × 120 条件—频率网格，并在单个 CPU 核心上运行，因此建议预留 **15–20 分钟**。在 M 系列 MacBook Air 上的实测时间为 16 分钟。

仅需要检查安装是否正确时，可运行：

```bash
python3 bin/selftest.py
```

该程序会在一个经过刻意缩小的网格上执行整个流程，耗时约 90 秒。

使用 GPU 版本时，进入 `gpu/` 目录，并安装与你的 CUDA 环境匹配的 JAX 版本：

```bash
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"
python3 -c "import jax; print(jax.default_backend(), jax.devices())"   # 预期输出：gpu
```

每个版本各自的 README 均介绍了完整的冻结—验证工作流、桌面界面以及部署 API。

## 桌面界面

在任一版本中，双击以下文件即可启动：

- macOS：**`启动界面_Start_GUI.command`**
- Windows：**`启动界面_Start_GUI.bat`**

启动器会自动寻找 Python，安装 `requirements.txt` 中缺失的依赖，并打开程序窗口。界面可随时在中文和英文之间切换。

```text
输入 CSV → 估计器集成拟合 → 参数路由 → 风险认证 → XLSX（4 个工作表）
```

界面会显示当前使用的计算后端。

当输入网格与冻结系统的校准网格不一致时，界面会将所有认证单元格标记为“超出校准范围”。CSV 读取器对数据格式作出的每一项推断，也都会记录在日志和输出工作簿中。

## macOS 应用

CPU 版本可以打包为可直接双击运行的 `.app`：

```bash
cd cpu
./script/package_macos_app.sh
open "dist/EIS-PEM ShanHaiJue.app"
```

应用包中包含：

- 完整流水线；
- 冻结后的已验证系统；
- 参考 CSV 文件；
- 内嵌 Python 运行时。

因此，用户无需自行安装任何依赖即可启动程序。

应用还会将各个 `bin/` 脚本作为命令行入口暴露在：

```text
Contents/MacOS/
```

以下文件记录了被打包进应用的确切 Git 状态、冻结系统及软件包版本：

```text
Contents/Resources/PipelineManifest.txt
```

与上一版本的 `package_macos_app.sh` 一样，这仍然是一个**面向本机构建的应用包**：内嵌运行时本质上是使用打包机器上的 Python 创建的虚拟环境。

若要制作可在其他 Mac 上可靠分发的版本，还需要：

- 冻结的独立运行时；
- Developer ID 签名；
- Apple 公证。

## CSV 输入规定

标准格式包含 5 列，每一行对应一个频率点，同一实验条件下的阻抗谱按顺序首尾连接：

```text
Temperature_K,SOC,Frequency_Hz,Re_Z,Im_Z
278.15,0.1,0.001,0.019984408733,-0.005062697
```

`examples/calibration_grid_25x120.csv` 使用上述格式，并对应冻结系统进行校准时所采用的实验网格。因此，它既是格式参考，也是承载认证保证的采集设计。

`examples/format_sample.csv` 是一个包含 30 行数据的简化示例。

系统也接受其他数据布局。真正必需的只有：

- 频率；
- 复阻抗。

系统还支持：

- 表头别名，包括中文表头；
- 带括号的单位；
- 自动识别分隔符；
- 无表头数值文件；
- `|Z|` 与相位格式；
- 摄氏温度或开尔文温度自动识别；
- 0–1 或 0–100 SOC 自动识别。

所有自动推断都会记录在日志和输出工作簿中。完整格式说明见各版本自己的 README。

## 仓库结构

```text
cpu/                       CPU 版本——参考实现
gpu/                       GPU 版本——相同科学方法，使用批量 JAX/CUDA 求解器
  shanhaijue/              面板、估计器分支、特征、路由器、认证器、流水线
    vendor/eispem/         48 参数物理层，包含 35 个模块，采用统一扁平命名空间
  gui/                     桌面界面
  bin/                     面板生成、路由、冻结、验证和认证
  artifacts/               支撑上述结果的冻结已验证系统
  examples/                参考 CSV 文件
cpu/script/                macOS 应用打包脚本
```

两个代码树仅有 6 个文件不同：

```text
README.md
requirements.txt
shanhaijue/__init__.py
shanhaijue/arms.py
bin/run_panel.py
bin/selftest.py
```

此外，`cpu/script/` 仅存在于 CPU 版本，在 GPU 版本中没有对应目录。

除此以外，两个版本的所有内容均逐字节一致，包括所内嵌的物理模型层和冻结后的系统工件。


上一版本、相关基准比较和研究发展历史可见：

[eric-haotian/eis-pem-identification](https://github.com/eric-haotian/eis-pem-identification)

## 许可证

采用 Apache License 2.0，详见 [LICENSE](LICENSE)。
