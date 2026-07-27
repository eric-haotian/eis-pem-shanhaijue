# learning AI at www.haotianblog.com
"""Bilingual strings for the desktop interface.

Every user-visible string lives here, keyed by a short identifier. `T(key)` returns the
string in the currently selected language; `set_language()` switches and the window
re-labels itself in place.
"""

LANGUAGES = ("zh", "en")
_current = "zh"

STRINGS = {
    # -- window / sections
    "title": ("EIS-PEM 山海决 · 参数辨识与风险受控认证",
              "EIS-PEM ShanHaiJue · Parameter Identification and Risk-Controlled Certification"),
    "sec_io": ("文件", "Files"),
    "sec_params": ("参数", "Parameters"),
    "sec_status": ("状态", "Status"),
    "sec_log": ("日志", "Log"),
    "lang_button": ("English", "中文"),

    # -- file row
    "in_csv": ("输入 CSV", "Input CSV"),
    "out_xlsx": ("输出 XLSX", "Output XLSX"),
    "system": ("冻结系统", "Frozen system"),
    "browse": ("浏览…", "Browse…"),

    # -- parameters
    "arms": ("集成臂数", "Ensemble arms"),
    "arms_hint": ("8 臂为已验证的最强配置", "8 arms is the strongest validated configuration"),
    "budget": ("错误声明预算", "False-claim budget"),
    "budget_hint": ("越大声明越多、精度越低", "Larger budget: more claims, lower precision"),
    "max_nfev": ("每次拟合迭代上限", "Iterations per fit"),
    "starts": ("每臂起点数", "Starts per arm"),
    "procs": ("并行进程", "Worker processes"),
    "chunk": ("GPU 批大小", "GPU batch size"),
    "split": ("谱线切分", "Spectrum splitting"),
    "split_auto": ("自动（频率回跳处分割）", "Automatic (split where frequency restarts)"),
    "split_col": ("按列", "By column"),
    "group": ("电池分组列", "Cell grouping column"),
    "group_none": ("（无：全部谱线属同一电池）", "(none: all spectra belong to one cell)"),
    "negate": ("虚部取负（列为 −Z″ 时勾选）", "Negate imaginary part (tick if the column is -Z\")"),
    "def_temp": ("缺省温度 °C", "Default temperature, deg C"),
    "def_soc": ("缺省 SOC", "Default SOC"),

    # ==============================================================================
    # parameter panel:每个可调项 = 标题 + 一行趋势提示 + 悬停完整说明
    # ==============================================================================
    "sec_input": ("输入解析", "Input parsing"),
    "sec_ensemble": ("估计器集成", "Estimator ensemble"),
    "sec_solver": ("求解器", "Solver"),
    "sec_claim": ("声明与认证", "Claim and certification"),
    "sec_compute": ("算力", "Compute"),
    "sec_refreeze": ("重新冻结专用（不影响本次运行）",
                     "Re-freeze only (inert for this run)"),

    # -- input parsing
    "p_negate": ("虚部取负", "Negate imaginary part"),
    "p_negate_hint": ("列名已含 −Z″ 时会自动处理，此处仅用于列名看不出符号的情况",
                      "auto-handled when the header says -Z\"; use this only when the sign is not in the name"),
    "p_negate_tip": (
        "把 Z″ 整列乘以 −1。EIS 数据常以 −Z″ 上报（Nyquist 图第一象限），而模型用的是 Z″ 本身。\n"
        "列名写明 −Z″/-Z_im 时程序已自动还原，日志会写明；只有列名看不出符号、而拟合明显把曲线"
        "翻到错误半平面时，才需要手动勾选。\n"
        "勾错的后果：容性弧变成感性弧，所有拟合都会跑偏，不增加算力开销。",
        "Multiplies the whole Z\" column by -1. EIS data is often reported as -Z\" so the Nyquist "
        "plot sits in the first quadrant, while the model uses Z\" itself.\n"
        "When the header says -Z\"/-Z_im this is already undone automatically and the log says so; "
        "tick this only when the sign is not visible in the header and the fit is clearly landing "
        "in the wrong half-plane.\n"
        "Getting it wrong turns the capacitive arc into an inductive one and every fit goes astray. "
        "No compute cost."),
    "p_def_temp": ("缺省温度 °C", "Fallback temperature, deg C"),
    "p_def_temp_hint": ("仅当 CSV 没有温度列时才使用", "used only when the CSV has no temperature column"),
    "p_def_temp_tip": (
        "CSV 里有温度列时，这个框不生效，会自动置灰——温度一律以文件为准，°C 与 K 自动判别"
        "（数值小于 200 视为 °C）。\n"
        "只有文件完全没有温度信息时，才用这里的值当作全部谱线的测量温度。\n"
        "填错的影响：温度进入 Arrhenius 项，直接改变动力学与扩散参数的标定，估计值会系统性偏移。",
        "Inactive, and greyed out, whenever the CSV has a temperature column: the file always wins, "
        "and deg C versus K is detected automatically (below 200 is read as deg C).\n"
        "This value is used only when the file carries no temperature at all.\n"
        "It matters: temperature enters the Arrhenius terms, so a wrong value systematically shifts "
        "the kinetic and diffusion estimates."),
    "p_def_soc": ("缺省 SOC", "Fallback SOC"),
    "p_def_soc_hint": ("仅当 CSV 没有 SOC 列时才使用", "used only when the CSV has no SOC column"),
    "p_def_soc_tip": (
        "与缺省温度同理：CSV 有 SOC 列时此框置灰不生效，0–1 与 0–100 两种写法自动判别"
        "（最大值大于 1.5 视为百分数）。\n"
        "SOC 决定两极的化学计量工作点，进而决定开路电位斜率与界面阻抗，填错会使化学计量族与"
        "动力学族的估计整体偏移。",
        "Same rule as the fallback temperature: greyed out when the CSV supplies SOC, and 0-1 versus "
        "0-100 is detected automatically (a maximum above 1.5 is read as percent).\n"
        "SOC sets the stoichiometric operating point of both electrodes, hence the open-circuit "
        "slope and the interfacial impedance; a wrong value shifts the stoichiometry and kinetics "
        "families as a group."),
    "p_group_column": ("电池分组列", "Cell grouping column"),
    "p_group_column_hint": ("留空＝自动；填列名可强制按该列分电池",
                            "blank = automatic; a column name forces grouping by that column"),
    "p_group_column_tip": (
        "多个电池的谱线放在同一个文件里时，用哪一列区分它们。\n"
        "留空时程序自动识别 cell / cell_id / sample / 电池 等常见列名；若你的列名特殊（例如 "
        "pack_no），在这里写上即可。\n"
        "分组决定“一个电池”的边界：同一电池的各工况被拼成一条堆叠观测共同拟合 48 个参数，"
        "分错会把不同电池的谱线当成同一只电池的不同工况，估计必然错误。",
        "Which column separates cells when several are in one file.\n"
        "Left blank, common names (cell, cell_id, sample) are detected automatically; give the "
        "column name here if yours is unusual, e.g. pack_no.\n"
        "Grouping defines what 'one cell' means: a cell's conditions are stacked into a single "
        "observation and fitted with one shared 48-parameter vector, so grouping wrongly makes "
        "spectra from different cells be treated as conditions of one, and the estimates cannot "
        "be right."),

    # -- ensemble
    "p_arms": ("集成臂数", "Ensemble arms"),
    "p_arms_hint": ("越多恢复越多、耗时线性增加；8 臂为已验证的最强配置",
                    "more arms recover more, cost grows linearly; 8 arms is the strongest validated configuration"),
    "p_arms_tip": (
        "沿先验强度 λ 排布的估计器数量。8 臂＝λ∈{3, 1, 0.6, 0.3, 0.1, 0.03} 六条加宽起点臂与"
        "纯数据臂；6 臂取 λ∈{3, 1, 0.3, 0.1}；3 臂只保留 λ=1 加两条。\n"
        "机制：λ 大时先验稳住了病态代价谷的平坦方向，代价是在数据已确定的方向上引入偏差；"
        "λ→0 时偏差消失但平坦方向失稳。两种机制在不同坐标上失败（正确集 Jaccard 0.515），"
        "路由器逐坐标挑选最可能对的那一臂，臂越多可挑的越多。\n"
        "臂越多恢复越多，且已验证：验证面板上 8 臂典型恢复约 27 个坐标（随 3→6→8 递增）。"
        "但这是校准网格上的数字，真实数据的实际恢复数取决于谱线的数量与质量，可能高于或低于此值。\n"
        "耗时与臂数近似成正比；8 臂是论文中最强且已验证的配置。",
        "How many estimators are placed along the prior-strength axis lambda. Eight arms = the six "
        "lambda values {3, 1, 0.6, 0.3, 0.1, 0.03} plus a wide-multistart arm and a data-only arm; "
        "six arms uses lambda in {3, 1, 0.3, 0.1}; three keeps only lambda = 1 plus the other two.\n"
        "Mechanism: at large lambda the prior stabilises the flat directions of the sloppy cost "
        "valley at the price of bias on the data-determined directions; as lambda goes to zero that "
        "bias vanishes but the flat directions destabilise. The two regimes fail on different "
        "coordinates (correct-set Jaccard 0.515), and the router picks per coordinate, so more arms "
        "means more to pick from.\n"
        "More arms recover more, and it is validated: on the validation panel eight arms typically "
        "recover about 27 coordinates, rising across 3, 6, 8. That figure is on the calibration grid; "
        "on real data the actual count depends on how many spectra you have and their quality, and "
        "may be higher or lower.\n"
        "Run time is roughly proportional to the arm count. Eight is the strongest validated "
        "configuration in the paper."),
    "p_starts": ("每臂起点数", "Starts per arm"),
    "p_starts_hint": ("越多越不易陷局部极小、耗时线性增加；改动后认证概率需重新标定",
                      "more starts escape local minima more often, cost grows linearly; changing it "
                      "means the calibrated probabilities no longer strictly apply"),
    "p_starts_tip": (
        "每条 λ 臂拟合多少个不同起点，最终取搜索空间内的逐坐标中位数。\n"
        "增大：更不容易被局部极小困住，中位数更稳，估计的重复性提高；耗时与起点数成正比。\n"
        "减小：明显变快，但单次拟合落进坏盆地的风险直接传给该臂的估计。\n"
        "注意：跨起点的离散度本身是路由器使用的诊断量之一，改变起点数会移动这个特征的分布，"
        "冻结的认证概率不再严格适用（辨识仍然照常给出）。",
        "How many different starting points each lambda arm fits, with the arm's estimate taken as "
        "the coordinate-wise median in search space.\n"
        "Higher: local minima are escaped more often and the median is steadier, so estimates "
        "repeat better; cost is proportional to the count.\n"
        "Lower: noticeably faster, but one bad basin now propagates straight into that arm.\n"
        "Note: the dispersion across starts is itself one of the diagnostics the router reads, so "
        "changing this moves that feature's distribution and the frozen calibrated probabilities no "
        "longer strictly apply. Identification still runs normally."),
    "p_jitter": ("起点扰动幅度", "Start dispersion"),
    "p_jitter_hint": ("大：搜索更广、更慢更散；小：更快更稳但更依赖先验中心",
                      "larger: wider search, slower and noisier; smaller: faster and steadier but "
                      "anchored to the prior centre"),
    "p_jitter_tip": (
        "λ 臂的起点围绕先验中心的高斯扰动标准差（搜索坐标，即取对数后的尺度）。\n"
        "增大：起点撒得更开，更可能找到远离先验的真解，但也更容易落进坏盆地，跨起点离散度上升。\n"
        "减小：起点集中在先验中心附近，收敛快而稳，但如果真值离先验较远就找不回来。\n"
        "不改变单次拟合的耗时，只改变结果分布。属于会移动路由特征分布的量。",
        "Standard deviation of the Gaussian scatter of each lambda arm's starting points around the "
        "prior centre, in search coordinates (i.e. on the log scale).\n"
        "Higher: starts spread further, so a truth far from the prior is more reachable, at the cost "
        "of landing in bad basins more often and a higher across-start dispersion.\n"
        "Lower: starts cluster near the prior centre and converge fast and steadily, but a truth far "
        "from the prior stays out of reach.\n"
        "Does not change the cost of one fit, only the distribution of results. It does move the "
        "router's feature distribution."),
    "p_wide_jitter": ("宽起点扰动幅度", "Wide-start dispersion"),
    "p_wide_jitter_hint": ("专供多起点臂；大于起点扰动才有意义",
                           "for the multistart arm only; only meaningful if larger than the ordinary dispersion"),
    "p_wide_jitter_tip": (
        "多起点臂专用的扰动幅度，默认 0.35，明显大于普通 λ 臂的 0.15。\n"
        "这条臂的职责就是去先验管不到的远处试探，因此它的价值来自于比其他臂撒得更开；"
        "把它调到和普通扰动一样大，这条臂就退化成又一条普通臂，集成的多样性下降。\n"
        "增大：覆盖更广的盆地，代价是这条臂本身更不稳；减小：这条臂的独特作用被削弱。",
        "The scatter used by the multistart arm alone, 0.35 by default against 0.15 for the ordinary "
        "lambda arms.\n"
        "This arm exists to probe where the prior cannot reach, so its value comes from spreading "
        "wider than the others; set it equal to the ordinary dispersion and the arm degenerates into "
        "one more ordinary arm, reducing the ensemble's diversity.\n"
        "Higher: more basins covered, at the cost of this arm being less stable. Lower: its distinct "
        "role is weakened."),
    "p_n_wide": ("宽起点数量", "Wide starts"),
    "p_n_wide_hint": ("多起点臂额外撒的起点数；越多越稳越慢",
                      "extra starts for the multistart arm; more is steadier and slower"),
    "p_n_wide_tip": (
        "多起点臂在常规起点之外额外撒的宽起点数量。\n"
        "增大：这条臂找到全局盆地的概率上升，耗时按起点数线性增加。\n"
        "置 0：多起点臂退化为普通起点，集成里就没有专门做全局探索的成员了。",
        "How many extra wide starting points the multistart arm draws beyond its ordinary ones.\n"
        "Higher: this arm finds the global basin more often, with cost linear in the count.\n"
        "Set to zero and the multistart arm degenerates to ordinary starts, leaving the ensemble "
        "with no member dedicated to global exploration."),
    "p_prior_weight": ("先验权重系数", "Prior weight factor"),
    "p_prior_weight_hint": ("整体缩放全部 λ；大：更贴先验更稳但有偏；小：更贴数据但病态方向失稳",
                            "scales every lambda: higher stays near the prior (steady, biased), "
                            "lower follows the data (less biased, unstable in flat directions)"),
    "p_prior_weight_tip": (
        "对全部估计器臂的先验权重做统一缩放（乘在基准权重 0.005/σ 上），相当于把整条 λ 阶梯"
        "平移。\n"
        "增大：所有臂都更靠近物理先验，病态方向被压住、结果更稳，但数据已确定的方向上引入的"
        "偏差更大，可辨识参数的估计会被先验拉走。\n"
        "减小：更忠实于数据，偏差减小，但代价谷的平坦方向失稳，弱可辨识坐标的估计会剧烈跳动。\n"
        "这是集成机制本身的定义量：路由器和认证器都是在基准值 1.0 上标定的，改动后认证保证"
        "不再适用，建议仅在诊断先验影响时临时使用。",
        "Scales the prior weight of every arm together (multiplying the base weight 0.005/sigma), "
        "which shifts the whole lambda ladder.\n"
        "Higher: every arm sits closer to the physics prior, the ill-conditioned directions are held "
        "down and results steady, but the bias on the data-determined directions grows and "
        "well-identified parameters get pulled toward the prior.\n"
        "Lower: more faithful to the data and less biased, but the flat directions of the cost "
        "valley destabilise and weakly identified coordinates swing wildly.\n"
        "This is a defining quantity of the ensemble: the router and certifier were both calibrated "
        "at 1.0, so the certification guarantee does not survive a change. Use it to diagnose the "
        "prior's influence, not for production runs."),

    # -- solver
    "p_max_nfev": ("每次拟合迭代上限", "Iterations per fit"),
    "p_max_nfev_hint": ("越大拟合越收敛、耗时线性增加；太小会在未收敛处停下",
                        "more iterations converge further at a linear cost; too few stop short of convergence"),
    "p_max_nfev_tip": (
        "单次拟合允许的最大迭代次数。CPU 版是 scipy 信赖域反射法的 max_nfev（收敛判据先满足"
        "就会提前停止，所以调大不一定真的更慢）；GPU 版是批量 LM 的固定步数（不提前退出，"
        "耗时严格与步数成正比）。\n"
        "增大：残差进一步下降，弱可辨识方向上的估计继续移动；到达收敛后再增大没有收益。\n"
        "减小：明显变快，但拟合可能停在未收敛处，此时残差类诊断量偏大，路由与认证会因此更"
        "保守（弃权更多）。",
        "The iteration cap for a single fit. On CPU this is scipy's max_nfev for the "
        "trust-region-reflective solver, which stops early once its convergence criteria are met, so "
        "raising it is not necessarily slower. On GPU it is the fixed step count of the batched "
        "Levenberg-Marquardt scan, which never exits early, so cost is strictly proportional.\n"
        "Higher: the residual falls further and weakly identified directions keep moving; past "
        "convergence there is no further gain.\n"
        "Lower: noticeably faster, but fits may stop short of convergence, which inflates the "
        "residual-based diagnostics and makes routing and certification more conservative, i.e. more "
        "abstentions."),
    "p_noise": ("仿真噪声水平", "Simulation noise level"),
    "p_noise_hint": ("只用于生成合成面板；不影响对你的 CSV 的拟合",
                     "only used when generating synthetic panels; does not affect fitting your CSV"),
    "p_noise_tip": (
        "生成合成面板时加在谱线上的比例高斯噪声标准差（默认 0.005，即 0.5%）。\n"
        "本次运行不使用它：你的 CSV 自带测量噪声，程序不会再加噪。它只在重新冻结流程里"
        "（bin/run_panel.py）决定训练与验证面板的信噪比。\n"
        "在重新冻结时增大：任务变难，可恢复坐标减少，认证阈值被迫抬高；减小：任务变易，"
        "得到的系统在真实噪声下会过于乐观。",
        "Standard deviation of the proportional Gaussian noise added to simulated spectra, 0.005 "
        "(0.5 %) by default.\n"
        "It is not used by this run: your CSV carries its own measurement noise and nothing is added "
        "to it. It only sets the signal-to-noise ratio of the training and validation panels in the "
        "re-freeze workflow (bin/run_panel.py).\n"
        "Raising it there makes the task harder, so fewer coordinates are recoverable and thresholds "
        "rise; lowering it makes the task easier and yields a system that is optimistic against real "
        "noise."),

    # -- claim and certification
    "p_budget": ("错误声明预算 B", "False-claim budget B"),
    "p_budget_hint": ("每电池允许的错误声明数上界：越大声明越多、精度越低",
                      "cap on false claims per cell: larger means more claims and lower precision"),
    "p_budget_tip": (
        "认证阈值由预算反解：t* = min{ t : UCB95[每电池错误声明数(t)] ≤ B }，上界是对生成真值"
        "做的聚类自助百分位。\n"
        "增大 B：允许更多错误，阈值下降，声明数上升而精度下降。已验证的取舍——B=0.60 时"
        "10.38 个声明、精度 0.960；再放宽到 0.65 就跌破 0.95 的合规边界。\n"
        "减小 B：更保守，声明更少但更可信。\n"
        "只能选择冻结时已标定过的档位：其他数值没有对应的冻结阈值。注意该上界在约 30 个聚类"
        "时实测单侧覆盖率 0.88–0.93（名义 0.95），读数应留 2–7 个百分点的余量。",
        "The threshold is solved from the budget: t* = min{ t : UCB95[false claims per cell at t] "
        "<= B }, where the bound is a percentile bootstrap over generating truths as the cluster "
        "unit.\n"
        "Raising B tolerates more errors, so the threshold falls, claims rise and precision drops. "
        "The validated trade-off: at B = 0.60, 10.38 claims at precision 0.960; loosening to 0.65 "
        "crosses below the 0.95 compliance boundary.\n"
        "Lowering B is more conservative: fewer claims, more trustworthy.\n"
        "Only budgets calibrated at freeze time are selectable, since no other value has a frozen "
        "threshold. Note the bound is anti-conservative at about 30 clusters, with measured "
        "one-sided coverage 0.88-0.93 against a nominal 0.95, so read it with a 2-7 point haircut."),
    "p_basis": ("声明坐标基", "Claim basis"),
    "p_basis_hint": ("phys＝谱线真正约束的物理组合（默认）；raw＝原始 48 个模型参数",
                     "phys = the physical combinations the spectra actually constrain (default); "
                     "raw = the 48 raw model parameters"),
    "p_basis_tip": (
        "在哪一组坐标上做声明。\n"
        "phys（默认，也是冻结系统标定所用）：把谱线真正约束的乘积与商作为声明对象，例如交换"
        "电流密度 i₀、扩散时间常数 τ_d = r_s²/D_s、SEI 电阻与电容。这些组合是可辨识的，而构成"
        "它们的单个参数往往不是。\n"
        "raw：直接声明 48 个原始模型参数。看起来更直接，但在病态方向上把不可辨识的量单独拿"
        "出来声明，精度会明显下降。\n"
        "改为 raw 后，路由器与认证器的标定不再对应，认证保证不适用。",
        "Which set of coordinates claims are made in.\n"
        "phys (the default, and what the frozen system was calibrated on): claims are made on the "
        "products and quotients the spectra actually constrain, e.g. the exchange current density "
        "i0, the diffusion time constant tau_d = r_s^2/D_s, and the SEI resistance and capacitance. "
        "These combinations are identifiable where their constituent parameters often are not.\n"
        "raw: claims the 48 raw model parameters directly. More literal, but it isolates quantities "
        "that are not identifiable along the ill-conditioned directions, and precision falls "
        "markedly.\n"
        "Switching to raw leaves the router and certifier calibration no longer matched, so the "
        "certification guarantee does not apply."),
    "p_tolerance": ("正确性容差", "Correctness tolerance"),
    "p_tolerance_hint": ("定义“算对”的相对误差门限；只在重新冻结时起作用",
                         "the relative error that counts as correct; only used when re-freezing"),
    "p_tolerance_tip": (
        "判定一个坐标是否恢复正确的相对误差门限，默认 0.10（10%）。\n"
        "本次运行不使用它：真实数据没有真值，无从判定对错。它只在训练与冻结时定义标签 y。\n"
        "在重新冻结时放宽（如 0.20）：更多坐标被记为正确，恢复数与认证数都会变大——但这是"
        "把尺子改松了，不是方法变强了，跨容差的数字不可直接比较。收紧则相反。",
        "The relative-error threshold at which a coordinate counts as recovered, 0.10 (10 %) by "
        "default.\n"
        "It is not used by this run: real data has no truth, so nothing can be scored. It defines "
        "the label y during training and freezing only.\n"
        "Loosening it at re-freeze time (say 0.20) marks more coordinates correct and inflates both "
        "recovery and certification counts, but that is a looser ruler rather than a stronger "
        "method, and numbers at different tolerances are not comparable. Tightening does the "
        "reverse."),

    # -- compute
    "p_procs": ("并行进程数", "Worker processes"),
    "p_procs_hint": ("多个电池时按电池并行；单个电池时无效",
                     "parallelises across cells; no effect on a single-cell file"),
    "p_procs_tip": (
        "同时拟合几个电池。各电池之间完全独立，结果与串行完全一致，只是更快。\n"
        "文件里只有一个电池时该设置不起作用——单个电池的拟合本身是串行的。\n"
        "每个进程要独立加载一遍物理模型，启动开销约十几秒；因此只有单个电池的拟合本身耗时"
        "远超这个开销（正常配置下每个电池几分钟）时才划算，小算例反而更慢。\n"
        "建议取物理核心数减一到二；设得比核心数多不会更快，反而因争抢内存带宽而变慢。"
        "内存占用按进程数增长。",
        "How many cells are fitted at once. Cells are fully independent, so the result is identical "
        "to running them one after another, only faster.\n"
        "It has no effect on a single-cell file, since one cell's fit is itself sequential.\n"
        "Each process loads the physics model independently, costing on the order of ten seconds of "
        "start-up, so this pays off only when a single cell's fit takes far longer than that (minutes "
        "per cell at normal settings). On small runs it is slower, not faster.\n"
        "Physical cores minus one or two is a good setting; more processes than cores does not help "
        "and tends to hurt through memory-bandwidth contention. Memory grows with the count."),
    "p_chunk": ("GPU 批大小", "GPU batch size"),
    "p_chunk_hint": ("一次编译批里并行多少次拟合：大＝快但吃显存",
                     "fits per compiled batch: larger is faster but uses more VRAM"),
    "p_chunk_tip": (
        "一个编译批里同时求解多少次拟合。默认 10，是按 8 GB 显存调的。\n"
        "增大：GPU 利用率上升、总时间下降，但显存占用近似线性增长，超出可用显存会直接 OOM。\n"
        "减小：显存压力下降，但每批的固定开销占比上升，整体变慢。\n"
        "不影响数值结果，只影响吞吐——同样的输入在不同批大小下给出相同的估计。",
        "How many fits are solved inside one compiled batch. The default of 10 is tuned for 8 GB of "
        "VRAM.\n"
        "Higher: better GPU utilisation and shorter total time, but VRAM grows roughly linearly and "
        "exceeding it is an out-of-memory failure.\n"
        "Lower: less memory pressure, but the fixed per-batch overhead weighs more and throughput "
        "falls.\n"
        "It does not affect the numbers, only the throughput: the same input gives the same estimates "
        "at any batch size."),

    # -- re-freeze only
    "p_n_folds": ("交叉拟合折数", "Cross-fitting folds"),
    "p_n_folds_hint": ("按真值分组的折数；越多标定越不乐观、训练越慢",
                       "truth-grouped folds: more folds means less optimistic calibration and slower training"),
    "p_n_folds_tip": (
        "训练时按生成真值分组做几折交叉拟合。路由器的取胜偏差（arg-max 的胜者诅咒）必须由"
        "折外得分吸收，这个数决定折外得分怎么算。默认 5。\n"
        "增大：每折训练数据更多、折外估计偏差更小，但训练次数线性增加。\n"
        "减小：训练快，但折外得分更乐观，认证阈值会被标定得偏松。\n"
        "本次运行不使用：冻结系统里的阈值已经定好了。",
        "How many truth-grouped folds are used for cross-fitting during training. The arg-max "
        "winner's-curse optimism must be absorbed by out-of-fold scores, and this sets how those are "
        "computed. Default 5.\n"
        "Higher: more training data per fold and less biased out-of-fold estimates, at a linear "
        "increase in training runs.\n"
        "Lower: faster, but the out-of-fold scores are more optimistic and the thresholds end up "
        "calibrated too loosely.\n"
        "Not used by this run: a frozen system's thresholds are already fixed."),
    "p_n_boot": ("自助重采样次数", "Bootstrap resamples"),
    "p_n_boot_hint": ("越多上界越稳、冻结越慢；不改变期望值",
                      "more resamples steady the bound and slow freezing; the expected value is unchanged"),
    "p_n_boot_tip": (
        "计算错误声明数 95% 上界时的聚类自助重采样次数，默认 2000。\n"
        "增大：上界的蒙特卡洛噪声下降，阈值更可复现；冻结耗时线性增加。\n"
        "减小：冻结更快，但阈值本身带上了抽样抖动，重复冻结会得到不同的阈值。\n"
        "注意它治不了真正的问题：在约 30 个聚类时该上界本身就偏乐观（实测覆盖 0.88–0.93），"
        "增加重采样次数只会让这个偏乐观的数字更稳定，不会让它更正确。\n"
        "本次运行不使用。",
        "The number of cluster bootstrap resamples behind the 95 % upper bound on false claims, "
        "2000 by default.\n"
        "Higher: less Monte-Carlo noise in the bound and more reproducible thresholds, at a linear "
        "cost during freezing.\n"
        "Lower: faster freezing, but the threshold itself carries resampling jitter and repeated "
        "freezes disagree.\n"
        "It does not fix the real limitation: at about 30 clusters the bound is anti-conservative to "
        "begin with (measured coverage 0.88-0.93), and more resamples only stabilise that optimistic "
        "number rather than correct it.\n"
        "Not used by this run."),
    "p_seed": ("随机种子", "Random seed"),
    "p_seed_hint": ("固定训练与自助的随机性；不改变方法，只改变具体实现",
                    "fixes training and bootstrap randomness; changes the realisation, not the method"),
    "p_seed_tip": (
        "训练、交叉拟合与自助过程的随机数种子。\n"
        "换一个种子会得到略有不同的路由器与阈值，差异反映方法的随机波动，不代表哪个更好——"
        "挑一个看起来最好的种子来报告结果就是选择性偏倚。\n"
        "冻结-验证流程要求验证种子在冻结时就预留好，之后不得更改。\n"
        "本次运行不使用。",
        "The random seed for training, cross-fitting, and the bootstrap.\n"
        "A different seed gives a slightly different router and thresholds; the spread reflects the "
        "method's randomness and does not make one seed better. Picking the seed that looks best is "
        "selection bias.\n"
        "The freeze-validate discipline requires validation seeds to be reserved at freeze time and "
        "never changed afterwards.\n"
        "Not used by this run."),

    # -- panel-level messages
    "log_knobs_changed": (
        "注意：{names} 已偏离标定值。辨识照常进行，但冻结的认证概率与阈值是在默认值上标定的，"
        "改动后其精度保证不再严格适用。",
        "Note: {names} differ from the calibrated defaults. Identification proceeds normally, but "
        "the frozen probabilities and thresholds were calibrated at those defaults, so their "
        "precision guarantee no longer strictly applies."),
    "log_cell_done_par": ("电池 {cell}：完成，认证 {n} 个参数（{done}/{total}）",
                          "Cell {cell}: done, {n} parameters certified ({done}/{total})"),
    "log_parallel": ("按电池并行：{n} 个进程处理 {cells} 个电池",
                     "Parallel over cells: {n} processes for {cells} cells"),
    "reset": ("恢复默认值", "Restore defaults"),

    # -- buttons
    "start": ("开始", "Start"),
    "stop": ("停止", "Stop"),
    "open_out": ("打开输出位置", "Open output folder"),
    "clear_log": ("清空日志", "Clear log"),

    # -- status
    "idle": ("就绪", "Idle"),
    "reading": ("正在读取 CSV…", "Reading CSV…"),
    "fitting": ("拟合中", "Fitting"),
    "routing": ("路由与认证中…", "Routing and certifying…"),
    "writing": ("写入 XLSX…", "Writing XLSX…"),
    "done": ("完成", "Done"),
    "stopped": ("已停止", "Stopped"),
    "failed": ("失败", "Failed"),
    "elapsed": ("已用", "Elapsed"),
    "remaining": ("预计剩余", "Remaining"),
    "cell_progress": ("电池", "Cell"),

    # -- log messages
    "log_loaded": ("已载入 {rows} 行 → 识别出 {spectra} 条谱线 / {cells} 个电池",
                   "Loaded {rows} rows -> {spectra} spectra / {cells} cells"),
    "log_grid": ("采集网格：{cond} 个工况 × {freq} 个频点 = {pts} 点/电池",
                 "Acquisition grid: {cond} conditions x {freq} frequencies = {pts} points per cell"),
    "log_grid_match": ("网格与冻结系统的校准网格一致——认证阈值适用。",
                       "Grid matches the frozen system's calibration grid; certification thresholds apply."),

    # -- axis-by-axis grid comparison (more actionable than a single in/out verdict)
    "grid_axis_head": ("与冻结系统校准网格的逐轴比对：",
                       "Axis-by-axis comparison against the frozen system's calibration grid:"),
    "grid_axis_temp_ok": ("  温度 一致：{v} °C", "  temperature matches: {v} deg C"),
    "grid_axis_temp_no": ("  温度 不同：本文件 {v} °C ／ 校准 {r} °C",
                          "  temperature DIFFERS: file {v} deg C / calibrated {r} deg C"),
    "grid_axis_soc_ok": ("  SOC 一致：{v}", "  SOC matches: {v}"),
    "grid_axis_soc_no": ("  SOC 不同：本文件 {v} ／ 校准 {r}",
                         "  SOC DIFFERS: file {v} / calibrated {r}"),
    "grid_axis_freq_ok": ("  频点 一致：{n} 个", "  frequencies match: {n} points"),
    "grid_axis_freq_no": ("  频点 不同：本文件 {n} 个 ／ 校准 {r} 个",
                          "  frequencies DIFFER: file {n} / calibrated {r}"),
    "grid_same_size": (
        "  点数相同（{pts}）但布点不同：信息预算未变，采集设计已变。认证阈值是在另一套布点上标定的，"
        "此处按校准外处理；若这套设计要长期使用，值得在其上重新冻结-验证一次。",
        "  Same point count ({pts}) but a different placement: the information budget is unchanged, "
        "the acquisition design is not. The thresholds were calibrated on the other placement, so "
        "this run is treated as out of calibration; if this design is here to stay, it is worth one "
        "freeze-validate pass on it."),
    "log_grid_mismatch": (
        "本数据的采集网格与冻结系统的校准网格不同。辨识照常进行并给出全部 48 个坐标的估计值，"
        "这一步不依赖校准网格；受影响的只有认证：概率与阈值是在另一个信息量的网格上标定的，"
        "≥0.95 精度保证不随之转移。输出中的认证列标注为“校准外”，请按点估计看待。"
        "若要在本网格上获得有效的认证保证，用 bin/freeze_system.py 在该网格重新冻结并验证。",
        "The acquisition grid of this data differs from the frozen system's calibration grid. "
        "Identification proceeds normally and returns estimates for all 48 coordinates; that step "
        "does not depend on the calibration grid. Only certification is affected: the probabilities "
        "and thresholds were calibrated at a different information content, so the >=0.95 precision "
        "guarantee does not transfer. Certification columns are marked out-of-calibration and should "
        "be read as point estimates. To obtain a valid guarantee on this grid, re-freeze and "
        "re-validate on it with bin/freeze_system.py."),
    "warn_headerless": ("表头为纯数字，按无表头文件读取（{n} 列）。",
                        "The first row is numeric; read as a headerless file ({n} columns)."),
    "warn_neg_imag": ("检测到 −Z″ 列，已自动取负还原为 Z″。",
                      "A -Z\" column was detected and negated back to Z\"."),
    "warn_polar": ("按极坐标（|Z|、相位）读取并转换为复阻抗。",
                   "Read in polar form (|Z|, phase) and converted to complex impedance."),
    "warn_default_temp": ("文件未提供温度，按 {t:.1f} °C 处理。",
                          "No temperature column; assuming {t:.1f} deg C."),
    "warn_default_soc": ("文件未提供 SOC，按 {s:.2f} 处理。",
                         "No SOC column; assuming SOC {s:.2f}."),
    "warn_resampled": ("电池 {cell} 各工况频点不一致，已在公共频段上重采样（{before} → {after} 点）。",
                       "Cell {cell}: conditions were measured at different frequencies; resampled "
                       "onto their common range ({before} -> {after} points)."),
    "banner_out_cal": (
        "⚠ 校准外网格：估计值有效，认证精度保证不适用（详见日志）。",
        "Out-of-calibration grid: estimates are valid, the certification precision "
        "guarantee is not (see log)."),
    "warn_thin": ("本电池仅 {pts} 个复数点约束 48 个参数，可辨识信息有限，多数坐标预期弃权。",
                  "Only {pts} complex points constrain 48 parameters here; the information content "
                  "is limited and most coordinates are expected to abstain."),
    "log_cell_start": ("电池 {cell}：开始拟合 {arms} 个估计器臂", "Cell {cell}: fitting {arms} estimator arms"),
    "log_cell_done": ("电池 {cell}：完成，用时 {sec:.0f} 秒，认证 {n} 个参数",
                      "Cell {cell}: done in {sec:.0f} s, {n} parameters certified"),
    "log_written": ("已写入 {path}", "Written to {path}"),
    "log_stop_req": ("收到停止请求，正在结束当前电池…", "Stop requested; finishing the current cell…"),
    "log_error": ("错误：{msg}", "Error: {msg}"),

    # -- errors
    "err_no_input": ("请先选择输入 CSV。", "Please choose an input CSV first."),
    "err_no_system": ("请先选择冻结系统（artifacts/validated_system）。",
                      "Please choose a frozen system (artifacts/validated_system)."),
    "err_columns": (
        "无法从 CSV 中读出频率与复阻抗。频率列可命名为 frequency/freq/f/频率；阻抗可给直角坐标"
        "（Zre/Z'/实部 与 Zim/Z''/−Z''/虚部）或极坐标（|Z| 与 phase）。温度与 SOC 可省略，"
        "省略时按面板上的缺省值处理。无表头的纯数字文件也可读入（3 列＝f, Z', Z''）。当前表头：{cols}",
        "Frequency and complex impedance could not be read from the CSV. The frequency column may "
        "be named frequency/freq/f; impedance may be rectangular (Zre/Z'/real with Zim/Z''/-Z''/imag) "
        "or polar (|Z| with phase). Temperature and SOC may be omitted, in which case the panel "
        "defaults are used. A headerless numeric file is also accepted (3 columns = f, Z', Z''). "
        "Header found: {cols}"),
    "err_empty": ("CSV 中没有可用的数据行。", "No usable data rows in the CSV."),

    # -- xlsx sheet names and headers
    "sheet_claims": ("认证声明", "Certified claims"),
    "sheet_all": ("全部坐标", "All coordinates"),
    "sheet_summary": ("运行摘要", "Run summary"),
    "sheet_log": ("日志", "Log"),
    "col_cell": ("电池", "Cell"),
    "col_param": ("参数", "Parameter"),
    "col_family": ("物理族", "Family"),
    "col_value": ("估计值", "Estimate"),
    "col_prob": ("认证概率", "Certified probability"),
    "col_thr": ("阈值", "Threshold"),
    "col_arm": ("选中估计器", "Selected estimator"),
    "col_certified": ("是否认证", "Certified"),
    "col_status": ("状态", "Status"),
    "val_yes": ("是", "yes"),
    "val_no": ("否（弃权）", "no (abstained)"),
    "val_in_cal": ("校准内", "in calibration"),
    "val_out_cal": ("校准外", "OUT OF CALIBRATION"),
    "sum_grid": ("采集网格（工况 × 频点）", "Acquisition grid (conditions x frequencies)"),
    "sum_key": ("项目", "Item"),
    "sum_val": ("值", "Value"),

    # -- tabs and the in-window results table
    "tab_params": ("参数", "Parameters"),
    "tab_results": ("辨识结果", "Results"),
    "tab_log": ("日志", "Log"),
    "res_none": ("尚无结果。选择 CSV 后点“开始”。",
                 "No results yet. Choose a CSV and press Start."),
    "res_summary": ("{cells} 个电池 · 每个 {coords} 个坐标 · 认证 {claimed} 个（预算 {budget}）",
                    "{cells} cells x {coords} coordinates, {claimed} certified (budget {budget})"),
    "res_filter_all": ("全部坐标", "All coordinates"),
    "res_filter_cert": ("仅认证声明", "Certified only"),
    "res_filter_abst": ("仅弃权坐标", "Abstained only"),
    "res_sort_hint": ("点击表头排序", "Click a column header to sort"),
    "res_copy": ("复制选中行", "Copy selection"),
    "res_copied": ("已复制 {n} 行到剪贴板", "Copied {n} rows to the clipboard"),

    # -- CSV auto-detection feedback on the file row
    "probe_ok": ("已识别列：{cols}", "Columns detected: {cols}"),
    "probe_from_csv": ("由 CSV 提供，此处不生效", "supplied by the CSV; this control is inactive"),
    "probe_fallback": ("CSV 无此列，将使用此值", "the CSV has no such column; this value is used"),
    "probe_bad": ("无法解析该文件的列", "The columns of this file could not be resolved"),

    # -- canonical input format
    "fmt_head": ("标准输入格式", "Canonical input format"),
    "fmt_body": (
        "Temperature_K, SOC, Frequency_Hz, Re_Z, Im_Z —— 每个工况的谱线首尾相接排列。"
        "examples/calibration_grid_25x120.csv 就是该格式，且正好落在冻结系统的校准网格上。",
        "Temperature_K, SOC, Frequency_Hz, Re_Z, Im_Z, with each condition's spectrum "
        "concatenated head to tail. examples/calibration_grid_25x120.csv is that format, on the "
        "frozen system's calibration grid."),
}


def set_language(lang: str):
    global _current
    if lang in LANGUAGES:
        _current = lang


def language() -> str:
    return _current


def T(key: str, **fmt) -> str:
    zh, en = STRINGS[key]
    s = zh if _current == "zh" else en
    return s.format(**fmt) if fmt else s
