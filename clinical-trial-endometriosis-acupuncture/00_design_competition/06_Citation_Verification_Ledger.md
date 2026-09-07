# 文献核验台账（Citation Verification Ledger）V1.0

**用途**：本项目所有设计决策所引用文献的核验状态记录。**在伦理递交与正式 Protocol 定稿之前，团队必须在可访问 PubMed / ClinicalTrials.gov / ChiCTR 的环境中逐条复核标记为 UNVERIFIED 的条目**，并将本表更新为 V1.1。

## 0. 本轮检索的真实条件（必须如实向审稿人/伦理委员会说明的方法学限制）

本轮设计审计在一个受限网络环境中完成。出口策略屏蔽了以下全部站点（CONNECT 403）：PubMed、PMC、eutils、Europe PMC、EBI REST、OpenAlex、Crossref、Semantic Scholar、doi.org、ClinicalTrials.gov（含 cdn 与 api/v2）、WHO ICTRP、ChiCTR、Cochrane Library、EQUATOR、ESHRE、NICE，以及 JAMA/Lancet/BMJ/Elsevier/Wiley/OUP/Springer/Frontiers/MDPI/Dove 等出版商站点与全部注册库镜像。

因此：

- 唯一可用渠道为搜索引擎返回的**摘要片段（snippet）**，其额度（200 次）已全部用尽；
- 极少数条目通过 GitHub 上缓存的 PubMed/PMC 记录与 RIS 导出文件得到**逐字核对**；
- **没有任何一条文献是通过打开原始摘要页面核实的**（除下表标注 VERIFIED 的少数缓存记录外）。

### 核验等级定义
| 等级 | 含义 |
|---|---|
| **VERIFIED** | 已读到逐字的 PubMed 记录、PMC 全文或数据库导出摘要（本环境中经 GitHub 缓存获得）|
| **VERIFIED-S** | 事实/数字出现在指向原始来源（PubMed/CT.gov/期刊摘要）的搜索片段中，未打开页面 |
| **UNVERIFIED-Q** | 由另一份 VERIFIED 文献或第三方知识库转引 |
| **UNVERIFIED-S** | 仅片段提及题名/存在性，关键数字未获得 |
| **UNVERIFIED-R** | 仅来自方法学负责人的知识记忆，必须复核 |
| **NOT RETRIEVED** | 明确尝试但未取得；**不得引用** |

**设计安全性声明**：本项目的任何设计决策都不依赖单一 UNVERIFIED 数字。样本量以情景分析给出（`05_Sample_Size_and_Recruitment_Scenarios.md`），并在启动前由本院数据校准；主要文献仅用于确定研究方向与终点选择，其结论方向在多个独立来源中一致。

---

## 1. 核心决策依赖的文献（按对设计的影响排序）

| # | 文献/注册号 | 用于支持的设计决策 | 核验等级 | 复核要点 |
|---|---|---|---|---|
| 1 | Gomez-Llerena A 等, J Minim Invasive Gynecol 2026;33(3):258-265, PMID 40721059 | CSI/CS 为**预后**因素：CS 者术后疼痛改善更小 RR 0.79 (0.73–0.86)、持续疼痛 RR 2.27 (1.40–3.68)；CS 患病率 28.3–80.2% | **VERIFIED**（逐字 PubMed 记录）| 注意其摘要结论误写为 "catastrophization"，引用时须谨慎 |
| 2 | Gentles A, Yong PJ 等, J Clin Med 2024;13(24):7521, PMID 39768444 | "目前没有检测 CS/nociplastic pain 的标准方法"；CSI ≥40 为内异症人群已验证切点；PPT 是最常用 QST | **VERIFIED**（全文）| — |
| 3 | Orr NL 等, JAMA Netw Open 2023;6(2):e230780, PMID 36848090 | 基线 CSI 预测术后 CPP（校正基线疼痛后），N=239 前瞻队列 | UNVERIFIED-S / UNVERIFIED-Q | **必须复核 β 值与 95% CI**；本文件中未引用任何具体效应量 |
| 4 | Orr NL 等, Pain 2022;163(2):e234-e245, PMID 34030173 | CSI ≥40 在内异症中的切点与患病率（约 42%）；激素治疗自报无应答 18% vs 6% | UNVERIFIED-S / UNVERIFIED-Q | 复核 N 与结论原文；分层因素设定依赖该切点 |
| 5 | Li PS 等, Fertil Steril 2023;119(5):815-823, PMID 36716811（NCT03125304）| 现有最高质量的内异症针刺假针对照 RCT：106 例、4 中心、12 周（每周 3 次）、痛经改善但第 24 周消失、**非经期盆腔痛与性交痛无差异** | VERIFIED-S（设计与结论方向）；效应量 NOT RETRIEVED | **必须复核效应量、SD、失访率**——SD 直接影响样本量假设 |
| 6 | Woolf CJ, Pain 2011;152(3 Suppl):S2-15, PMID 20961685 | central sensitization 定义；"疼痛超敏本身不足以诊断 CS" → 措辞纪律 | **VERIFIED**（全文）| — |
| 7 | Kosek E 等, Pain 2021;162(11):2629-2634, PMID 33974577 | nociplastic pain 临床分级标准 → CSI 不等于 nociplastic pain 诊断 | UNVERIFIED-S（标准内容经 VERIFIED 综述转引）| 复核条目原文 |
| 8 | Schug SA 等, Pain 2019;160(1):45-52, PMID 30586070 | ICD-11 chronic postsurgical pain 定义（术后新发或加重、≥3 个月）→ **术语改为 PPP-ES** | UNVERIFIED-S（文献条目已核；"术前已有疼痛延续不算 CPSP" 的排除条款为 UNVERIFIED-R）| **必须复核排除条款原文**——这是术语决定的依据 |
| 9 | Duffy JMN 等, BJOG 2020, PMID 32227676 | 内异症核心结局集：overall pain、most troublesome symptom、QoL、AE、satisfaction | VERIFIED-S | 复核核心结局逐条列表 |
| 10 | Systematic Reviews (BMC) 2024, DOI 10.1186/s13643-024-02692-0, PMC11624600 | 针灸盲法系统综述：亚洲人群、**穿刺性假针**、研究中反复询问分组 → 去盲风险升高 | VERIFIED-S（三个风险因素）；合并 BI 数值 NOT RETRIEVED | 复核合并 Bang BI 与纳入试验数——影响假针选择 |
| 11 | MacPherson H 等, PLoS One 2014;9:e93739 | 穿刺性假针使真/假差异更小 | VERIFIED-S（定性）；SMD NOT RETRIEVED | 复核 |
| 12 | Linde K 等, BMC Med 2010;8:75 | 假针相对无治疗 SMD −0.45 (−0.57 to −0.34) → 假针非惰性 | VERIFIED-S | — |
| 13 | Vickers AJ 等, J Pain 2018;19:455-474 | 针刺 vs 假针 ≈0.2 SD；vs 无针刺 ≈0.5 SD → Δ 假设与"设计边界"声明 | VERIFIED-S | 复核，并复核效应修饰因素分析 |
| 14 | Bang H 等, Control Clin Trials 2004;25:143-156 | Bang 盲法指数与 ±0.2 判读 | VERIFIED-S（范围与阈值）| 复核 PMID 15020033 |
| 15 | Park J 等, Acupunct Med 2002;20:168-174, PMID 12512790 | Park 非穿刺假针验证（58 例卒中患者无人认为是假针）| VERIFIED-S | — |
| 16 | Enblom A 等 2011, PMID 21747890 | Park 型假针的盲法效果不受既往针灸经验影响 | VERIFIED-S | 与"中国患者既往针灸暴露率高"这一顾虑直接相关，**优先复核** |
| 17 | Witt CM 等, Forsch Komplementmed 2009, PMID 19420954 | 针刺安全性背景率：8.6% 任一 AE、6.1% 出血/血肿、1.7% 疼痛、0.7% 植物神经症状、229,230 例中 2 例气胸 | VERIFIED-S（逐字数字）| 用于研究者手册与知情同意 |
| 18 | MacPherson H 2001 (BMJ, PMID 11532841) / White A 2001 (PMID 11532840) | 34,407 与 32,000 次治疗无严重 AE | VERIFIED-S | — |
| 19 | Carr DJ, Acupunct Med 2015;33(5):413-419, PMID 26362792 | "禁忌穴"针刺 15 项试验 823 名孕妇无客观危害证据 | VERIFIED-S | 用于妊娠相关安全条款 |
| 20 | Becker CM 等（ESHRE 2022）, Hum Reprod Open 2022, PMID 35350465 | 术后长期激素治疗建议；对非药物治疗（含针灸）"无法作出推荐…需要设计充分的试验" | VERIFIED-S（引文与非药物段落）；术后激素段落逐字 UNVERIFIED | **必须复核逐字推荐与推荐强度**——用于背景治疗设计与立题论证 |
| 21 | NICE NG73（2017，2024 部分更新）| 术后可考虑激素治疗以延长手术获益 | VERIFIED-S（推荐语）；条目编号 UNVERIFIED | — |
| 22 | 《子宫内膜异位症诊治指南（第三版）》中华妇产科杂志 2021;56(12) | 术后长期管理；地诺孕素为长期管理首选 | VERIFIED-S（经二次文献）| **复核原文与页码** |
| 23 | 《子宫内膜异位症相关疼痛诊治指南（2024 实践版）》 | 药物为疼痛首选，含中药 | VERIFIED-S（片段）；期刊卷期 UNVERIFIED | 复核 |
| 24 | 《子宫内膜异位症中西医结合诊疗指南（2024 年版）》中华中医药学会 | 中西医结合诊疗定位 | VERIFIED-S；针灸推荐内容 UNVERIFIED | **复核是否含针灸推荐**——对立题与院内评审有价值 |
| 25 | 《地诺孕素临床应用中国专家共识》中华妇产科杂志 2024;59(7):505-512 | 地诺孕素用法与闭经/出血特征 | VERIFIED-S（引文）；闭经率 UNVERIFIED | **复核闭经率**——决定痛经能否作为全样本终点 |
| 26 | 国卫科教发〔2024〕32号（2024-10-01 施行）| IIT 管理：干预性研究须机构立项与外部专家科学性审查；正式启动前 30 日登记备案 | VERIFIED-S（要点）；条款号 UNVERIFIED | **复核条款号与本院执行细则** |
| 27 | 《涉及人的生命科学和医学研究伦理审查办法》（2023）| 干预性针刺研究不可免除伦理审查；跟踪审查 ≤12 个月 | VERIFIED-S | 复核条款号 |
| 28 | NCT01259180 | 韩国 2010 年注册的内异症/腺肌病 CPP 针刺三臂研究（真针 / Park 假针 / 观察；12 次 6 周；30 例；状态 Unknown，未见结果）| VERIFIED-S（设计与状态）；"术后人群"**未获证实** | **不得称其为术后研究**，除非复核确认 |
| 29 | NCT07305025 | 仅题名 "Pelvic Pain Electro-Acupuncture" 获证实；"腹腔镜诊断前电针 pilot" 的描述**未获证实** | UNVERIFIED-S | **复核后再引用** |
| 30 | NCT04553562 / NCT05223517（广安门医院，女性非周期性 CPP 针刺 RCT）| 主要外部竞争威胁 | VERIFIED-S（存在与人群）；结果未发表 | 每 6 个月复查 |
| 31 | J Pain Res 2024 NMA, DOI 10.2147/JPR.S488343（42 RCT / 3,635 例）；Arch Gynecol Obstet 2025 NMA, PMID 40019501（23 RCT / 1,545 例）；Medicine 2025 SR/MA（9 RCT / 535 例，VAS MD −1.67）| 领域拥挤度判断 | VERIFIED-S | PMID 40859475 与 Medicine 2025 的对应关系**需复核** |
| 32 | Cochrane CD007864.pub2（Zhu X 2011）| 仅纳入 1 项研究；呼吁高质量双盲 RCT | VERIFIED-S | 复核是否已更新 |
| 33 | Baeumler PI 等, Front Neurosci 2019, PMID 31354400；Front Neurol 2024, PMC11378649；J Pain 2024, PMID 38986891；J Integr Med 2025, PMID 39915157 | temporal summation 预测针刺应答（结果不一致）；低 PPT 者对**假针**应答更好 | UNVERIFIED-S | **优先复核**——直接影响 HTE 分析的解释框架与预期 |
| 34 | Mechsner S 等, Pain Med 2023;24(7):809-817, PMID 36882181 | 内异症 CPP 的 tDCS RCT 以 **PPT 为主要终点**（N=36）→ QST 子研究的方法学先例 | **VERIFIED** | — |
| 35 | Horne AW 等（GaPP2）, Lancet 2020 | 加巴喷丁治疗无明显病灶的 CPP 阴性（N=306）→ "中枢机制假说 → 治疗" 链条不自动成立 | UNVERIFIED-S | 用于红队论证 |
| 36 | Cooper KG 等（PRE-EMPT）, BMJ 2024, PMID 38749550 | 术后 LARC 与 COC 三年疼痛改善均约 40% → 术后仍有大量残余疼痛 | UNVERIFIED-S | — |
| 37 | SPIRIT 2025（Chan AW 等, BMJ 2025;389:e081477, PMID 40294521）；CONSORT 2025（Hopewell S 等, BMJ 2025;389:e081123, PMID 40245901）| 方案与报告规范（34 条 / 30 条 + 流程图）| VERIFIED-S（引文与条目数）；逐条内容 NOT RETRIEVED | **必须下载官方清单逐条对照**，本项目的 SPIRIT 清单文件已标注该待办 |
| 38 | STRICTA 2010（MacPherson H 等, PLoS Med 2010;7:e1000261, PMID 20543992）| 针刺干预报告 6 条 17 子条 | VERIFIED-S | 无 2026 年更新版；FAIR 指南编制中（UNVERIFIED-S）|
| 39 | CONSORT-Outcomes 2022 / SPIRIT-Outcomes 2022（Butcher NJ 等, JAMA 2022）；SPIRIT-PRO 2018（Calvert M 等, JAMA 2018;319:483-494）；CONSORT-Harms 2022（Junqueira DR 等, BMJ 2023）| 结局、PRO、危害报告扩展 | UNVERIFIED-R/S | 复核 PMID |
| 40 | ICH E9(R1) 附录（2019 Step 4）| estimand 五要素与伴发事件策略 | UNVERIFIED（标准文件）| 下载原文 |
| 41 | 上海市 IIT 管理办法（2024）与市级登记系统 | 是否存在独立于国家系统的市级备案要求 | **NOT FOUND** | **必须向本院科研处确认** |
| 42 | ChiCTR 2025–2026 注册要求与费用 | 预注册时限与字段 | VERIFIED-S（预注册须先于入组）；费用与时限 UNVERIFIED | 向 ChiCTR 或本院确认 |
| 43 | 龙华医院妇科/针灸科/临床研究中心的具体资源（手术量、专病门诊人次、IIT 流程、伦理委员会名称、一体化管理平台）| 可行性与时间表 | **NOT FOUND**（仅证实 陆氏针灸 国家级非遗 2011、针灸科 1956 年建科、陆瘦燕首任主任；田立霞 副主任医师 妇科）| 由《01_Feasibility_Audit》采集 |
| 44 | 中文版 CSI（CSI-25 / CSI-9）在中国慢性疼痛人群的验证 | 分层因素工具 | VERIFIED-S（CSI-9 中文版验证存在）；CSI-25 中文版验证 UNVERIFIED | **复核并确认使用授权** |
| 45 | EHP-30 中文版、PCS 中文版、BPI-SF 中文版、AES 中文版的验证与使用授权 | 结局工具 | UNVERIFIED-R | **逐一复核并取得使用许可** |

---

## 2. 复核责任分工与时限（伦理递交前完成）

| 组别 | 条目 | 负责人 | 时限 |
|---|---|---|---|
| A. 临床/证据 | #1–#9, #20–#25, #35, #36 | PI + 妇科研究生 | 4 周 |
| B. 针刺方法学 | #10–#16, #33, #34, #38 | 针灸科 Co-PI | 4 周 |
| C. 安全性 | #17–#19 | 协调员 | 2 周 |
| D. 法规与注册 | #26, #27, #41, #42 | 科研处/临床研究中心 | 4 周 |
| E. 报告规范与量表 | #37, #39, #40, #44, #45 | 方法学/统计 | 4 周 |
| F. 竞争监测 | #28–#32 | PI | 每 6 个月复查一次 |

## 3. 复核可能触发的设计变更（预先约定）
| 若复核发现 | 则 |
|---|---|
| Li 2023 报告的 NRS SD 明显 >2.5 | 提高样本量至 240 上限，或重新评估主要终点窗口平均的方差 |
| CSI-25 中文版无有效验证 | 改用已验证的中文 CSI-9 并同步调整分层切点（须重新界定），或将 CSI 降为连续协变量而不作分层 |
| ICD-11 CPSP 排除条款与本文件理解不符 | 重新审视术语裁定（但 PPP-ES 作为描述性术语仍成立）|
| 广安门试验已发表且人群含术后内异症患者 | 在讨论与立项书中重新定位本研究的增量，必要时强化 CSI-HTE 与三臂分解的独特性 |
| 地诺孕素闭经率高于预期 | 痛经相关次要终点的分析人群随之缩小，须在 SAP 中预设 |
| 本院无法提供 CSI 分层所需的即时评分 | 改为随机前一日完成 CSI 并由 IWRS 录入 |
