# 样本量、把握度与招募情景分析（Sample Size & Recruitment Scenarios）V1.0

**用途**：为《02_Design_Competition_Memo.md》第 6 节与《04_Design_Freeze_Sheet.md》提供可复现的计算依据。所有计算脚本附于文末（Python 3.11，statsmodels 0.15，scipy 1.17），可由临床研究中心统计师复算。

## 1. 统一假设
| 参数 | 取值 | 依据/说明 |
|---|---|---|
| 主要终点 | 第 9–12 周每周"过去 7 天平均盆腔痛"NRS（0–10）的均值 | 4 次均值降低单次评分噪声 |
| SD | 2.2（敏感性 2.0 / 2.5）| 内异症疼痛试验中单次 NRS/VAS(0–10) 的 SD 常见 2.3–2.6；4 周均值更小；取 2.2 为基准 |
| 基线–结局相关 ρ | 0.5（敏感性 0）| ANCOVA 调整基线 NRS；慢性痛试验常见 0.4–0.6 |
| α | 0.05 双侧 | 层级检验，不拆分 α |
| 失访 | 20%（冻结）；15% 为敏感性 | 24 周内；ePRO + 电话补救 |
| Δ(EA − UC) | 1.3（敏感性 1.2 / 1.5）| 针刺 vs 无针刺 SMD≈0.5–0.6（Vickers 2018 IPD 量级）× SD 2.2 ≈ 1.1–1.3 |
| Δ(EA − sham) | 1.0（敏感性 0.8 / 0.9 / 1.2）| 设定为"最小有临床意义的特异性效应"，而非文献均值（SMD≈0.2 ≈ 0.45 NRS，单中心不可检出）|

## 2. 两臂连续终点：每臂样本量（失访前）
| Δ | SD | ρ | power | n/臂 | 总 N | 总 N（+15% 失访）|
|---|---|---|---|---|---|---|
| 0.8 | 2.2 | 0.5 | 0.80 | ≈90 | ≈180 | ≈212 |
| 1.0 | 2.0 | 0.5 | 0.80 | 49 | 98 | 116 |
| 1.0 | 2.3 | 0.5 | 0.80 | 64 | 128 | 151 |
| 1.0 | 2.6 | 0.5 | 0.80 | 81 | 162 | 191 |
| 1.0 | 2.3 | 0.0 | 0.80 | 85 | 170 | 200 |
| 1.2 | 2.3 | 0.5 | 0.80 | 45 | 90 | 106 |
| 1.3 | 2.2 | 0.5 | 0.80 | ≈42 | ≈84 | ≈100 |
| 1.5 | 2.3 | 0.5 | 0.80 | 29 | 58 | 69 |
| 0.6 | 2.3 | 0.5 | 0.80 | 174 | 348 | 410 |

**解读**：特异性效应若真为文献均值（≈0.45–0.6 NRS），需要 400–700 例，单中心不可行；因此任何假针对照设计在单中心都只能宣称"能检出 ≥0.9–1.0 NRS 的特异性效应"。这不是缺陷，而是必须在方案与论文中如实写明的设计边界。

## 3. Champion（三臂 2:2:1）把握度
SD 2.2、ρ 0.5、失访 15%。

| 入组 N（EA/假/UC）| 可评价 | EA−UC Δ1.2 | Δ1.3 | Δ1.5 | EA−sham Δ0.8 | Δ0.9 | Δ1.0 | Δ1.2 |
|---|---|---|---|---|---|---|---|---|
| 180（72/72/36）| 61/61/30 | 0.80 | 0.86 | 0.94 | 0.63 | 0.73 | 0.82 | 0.93 |
| 200（80/80/40）| 68/68/34 | 0.84 | 0.90 | 0.96 | 0.68 | 0.78 | 0.86 | 0.95 |
| 210（84/84/42，失访 15%）| 71/71/35 | 0.86 | 0.91 | 0.97 | 0.70 | 0.80 | 0.87 | 0.96 |
| **210（84/84/42，失访 20%）——冻结值** | 67/67/34 | 0.83 | **0.90** | 0.96 | 0.67 | 0.77 | **0.86** | 0.95 |
| 240（96/96/48）| 81/81/40 | 0.90 | 0.94 | 0.98 | 0.76 | 0.85 | 0.91 | 0.98 |

SD 敏感性（N=210，失访 20%）：SD 2.0 → EA−UC(Δ1.3) ≈0.94、EA−sham(Δ1.0) ≈0.92；SD 2.5 → ≈0.81、≈0.76。
最小可检出差异（80% power，N=210，失访 20%）：EA−UC ≈1.14；EA−sham ≈0.93；sham−UC ≈1.14。

### 3.1 分配比选择（N=192 情景对比）
| 分配 | 可评价 | EA−UC Δ1.5 / Δ1.2 | EA−sham Δ1.0 / Δ0.8 |
|---|---|---|---|
| 1:1:1 | 54/54/54 | 0.97 / 0.87 | 0.73 / 0.54 |
| **2:2:1** | 65/65/32 | 0.93 / 0.79 | 0.81 / 0.62 |
| 3:3:2 | 61/61/40 | 0.96 / 0.83 | 0.79 / 0.59 |

2:2:1 在两个关键对比之间最均衡；UC 臂缩小的代价（EA−UC 把握度 0.97→0.93）远小于 EA−sham 的收益（0.73→0.81）。

### 3.2 盲态样本量再估计（SSR）
入组 100 例时由独立统计师用盲态合并 SD 重估；若合并 SD >2.4，则上调 N，上限 240；不允许基于效应量的调整。

## 4. 二分类应答终点（次要）
| p 对照 | p 治疗 | n/臂 | 总 N（+15%）|
|---|---|---|---|
| 0.25 | 0.50 | 58 | 137 |
| 0.30 | 0.55 | 60 | 142 |
| 0.30 | 0.50 | 93 | 219 |
| 0.35 | 0.55 | 96 | 226 |
在 N=200 下，≥30% 应答率的 EA−UC 对比对 25 个百分点的差异有约 80% 把握度；作为次要终点足够。

## 5. 预防设计（D5）为何被淘汰
| p(UC) | p(EA) | n/臂 | 总 N（+15%）|
|---|---|---|---|
| 0.25 | 0.15 | 248 | 584 |
| 0.25 | 0.125 | 150 | 353 |
| 0.30 | 0.15 | 119 | 280 |
| 0.20 | 0.10 | 195 | 459 |
需要 280–580 例手术患者且全部同意随机，且干预在术后早期（患者往返负担最大时）实施 → 单中心 3 年不可完成。

## 6. CSI 效应修饰（treatment × CSI 连续交互）把握度（模拟，3000 次）
交互定义：CSI 相差 30 分（约 2 个 SD）时治疗效应相差 D NRS。
| 对比 | 可评价 | D=1.0 | D=1.5 | D=2.0 |
|---|---|---|---|---|
| EA vs sham | 77/77 | 0.29 | 0.56 | 0.81 |
| EA vs sham | 65/65 | 0.26 | 0.51 | 0.75 |
| EA vs UC | 77/38 | 0.20 | 0.39 | 0.62 |
| 两臂 | 115/115 | 0.42 | 0.74 | 0.93 |
| 两臂 | 200/200 | 0.62 | 0.93 | 0.99 |

**结论**：本试验对 HTE 的把握度不足以做确证性检验；HTE 分析预注册为假设生成性，报告交互项估计值与 95% CI；其估计值将直接用于未来多中心（N≈400–800）试验的样本量设计。

## 7. 富集设计（D2）的筛查代价
CSI≥40 患病率 0.35 / 0.45 / 0.55 → 每随机 1 例需筛查合格者 2.86 / 2.22 / 1.82 例；在 8 例/月的合格流量下，随机速度降至 2.8–4.4 例/月，N=160 需 36–57 个月。

## 8. 招募情景（泊松到达；月数：中位 / 第 90 百分位）
| 合格随机例/月 | N=120 | N=150 | N=180 | N=192 | N=240 | N=300 |
|---|---|---|---|---|---|---|
| 3 | 40/45 | 50/56 | 60/66 | 64/71 | 80/87 | 101/108 |
| 4 | 30/34 | 38/42 | 45/50 | 48/53 | 60/65 | 75/81 |
| 5 | 25/27 | 30/34 | 36/40 | 39/43 | 49/52 | 60/65 |
| 6 | 20/23 | 26/28 | 30/33 | 32/35 | 40/44 | 50/54 |
| 8 | 15/17 | 19/21 | 23/25 | 24/27 | 30/33 | 38/41 |
| 10 | 12/14 | 15/17 | 18/20 | 20/21 | 24/26 | 30/33 |
| 12 | 10/12 | 13/14 | 15/17 | 16/18 | 20/22 | 26/27 |

### 8.1 招募流量的构成估计（待本院数据校准）
| 来源 | 估算链 | 悲观 | 基准 | 乐观 |
|---|---|---|---|---|
| 本院手术后 3 个月复诊者 | 手术量/年 × 持续痛比例 × 同意率 | 120×0.20×0.5 = 12/年 | 200×0.25×0.6 = 30/年 | 300×0.30×0.6 = 54/年 |
| 慢性盆腔痛/内异症专病门诊（外院术后）| 月就诊有手术史且 NRS≥4 者 × 合格率 × 同意率 | 20×0.4×0.4×12 = 38/年 | 40×0.4×0.5×12 = 96/年 | 60×0.45×0.5×12 = 162/年 |
| 合计 | | ≈50/年（4/月）| ≈126/年（≈10/月）| ≈216/年（18/月）|

**警示**：2023 Fertil Steril 多中心试验 4 个中心 3.5 年仅随机 106 例（566 例筛查），约 8 例/中心/年——其纳入条件（20–40 岁、按月经周期定时、每周 3 次共 36 次）比本方案严苛得多，但这仍是单中心 N=200 必须正视的下限情景。**第 9 个月招募审计 + 启动前病例量核查**是本设计不可省略的部分。

## 9. 时间表（基准情景 8 例/月）
| 阶段 | 月份 |
|---|---|
| 方案定稿、IIT 备案、伦理、ChiCTR 注册、ePRO 搭建 | 0–4 |
| 假针可信度 pilot（n≈24）| 3–5 |
| 首例随机（FPI）| 5–6 |
| 招募期 | 6–33（N=210；8 例/月时第 90 百分位约 29 个月）|
| 末例 24 周随访（LPLV-24w）| ≈39 |
| 主要分析与投稿 | 40–43 |
| 末例 52 周 ePRO | ≈46 |

## 附录：计算脚本
```python
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
rng = np.random.default_rng(20260906)

def n_per_arm_cont(delta, sd, power=0.8, alpha=0.05, ratio=1.0, rho=0.0):
    """ANCOVA-adjusted n per arm (arm1); ratio = n2/n1; rho = baseline-outcome corr."""
    sd_adj = sd*np.sqrt(1-rho**2)
    es = delta/sd_adj
    n1 = TTestIndPower().solve_power(effect_size=es, power=power, alpha=alpha, ratio=ratio, alternative='two-sided')
    return n1

def power_cont(delta, sd, n1, n2, alpha=0.05, rho=0.0):
    sd_adj = sd*np.sqrt(1-rho**2)
    es = delta/sd_adj
    return TTestIndPower().power(effect_size=es, nobs1=n1, alpha=alpha, ratio=n2/n1, alternative='two-sided')

print("="*80)
print("TABLE 1. Two-arm, continuous NRS endpoint. n per arm (before dropout), alpha=0.05 two-sided")
print("delta  SD   rho  power  n/arm  total  total+15%drop")
for delta in [0.6,0.8,1.0,1.2,1.5]:
    for sd in [2.0,2.3,2.6]:
        for rho in [0.0,0.5]:
            for pw in [0.8,0.9]:
                n1=n_per_arm_cont(delta,sd,pw,rho=rho)
                tot=2*np.ceil(n1)
                print(f"{delta:4.1f} {sd:4.1f} {rho:4.1f} {pw:5.2f} {np.ceil(n1):6.0f} {tot:6.0f} {np.ceil(tot/0.85):8.0f}")

print("="*80)
print("TABLE 2. Three-arm designs: power per pairwise contrast (no alpha split; hierarchical testing).")
print("Assume SD=2.3, rho=0.5 (ANCOVA). Evaluable N after 15% dropout shown.")
for total in [150,180,192,210,240]:
    for alloc,name in [((1,1,1),'1:1:1'),((2,2,1),'2:2:1'),((3,3,2),'3:3:2')]:
        w=np.array(alloc)/sum(alloc)
        n=np.floor(total*w*0.85)
        nA,nS,nU=n
        pAU15=power_cont(1.5,2.3,nA,nU,rho=0.5); pAU12=power_cont(1.2,2.3,nA,nU,rho=0.5)
        pAS10=power_cont(1.0,2.3,nA,nS,rho=0.5); pAS08=power_cont(0.8,2.3,nA,nS,rho=0.5); pAS07=power_cont(0.7,2.3,nA,nS,rho=0.5)
        print(f"N={total:3d} {name}: evaluable A/S/U={nA:.0f}/{nS:.0f}/{nU:.0f} | A-vs-UC d=1.5:{pAU15:.2f} d=1.2:{pAU12:.2f} | A-vs-Sham d=1.0:{pAS10:.2f} d=0.8:{pAS08:.2f} d=0.7:{pAS07:.2f}")

print("="*80)
print("TABLE 3. Binary responder endpoint (>=30% reduction). n per arm, power 0.8")
for p0 in [0.25,0.30,0.35]:
    for p1 in [0.45,0.50,0.55,0.60]:
        es=proportion_effectsize(p1,p0)
        n1=NormalIndPower().solve_power(effect_size=es,power=0.8,alpha=0.05,ratio=1)
        print(f"p_ctrl={p0:.2f} p_trt={p1:.2f} n/arm={np.ceil(n1):.0f} total+15%={np.ceil(2*np.ceil(n1)/0.85):.0f}")

print("="*80)
print("TABLE 4. Prevention design (perioperative EA vs UC; persistent pain at 6 mo binary)")
for p0,p1 in [(0.25,0.15),(0.25,0.125),(0.30,0.18),(0.30,0.15),(0.20,0.10)]:
    es=proportion_effectsize(p1,p0)
    n1=NormalIndPower().solve_power(effect_size=es,power=0.8,alpha=0.05,ratio=1)
    print(f"p_UC={p0:.3f} p_EA={p1:.3f} n/arm={np.ceil(n1):.0f} total+15%={np.ceil(2*np.ceil(n1)/0.85):.0f}")

print("="*80)
print("TABLE 5. Effect-modification (treatment x CSI) power by simulation. Continuous CSI (SD=14).")
print("Interaction defined as difference in treatment effect between CSI=25 and CSI=55 (30 pts).")
def sim_interaction(n_trt,n_ctl,int_diff30,sd_resid=2.0,nsim=3000,alpha=0.05,main=1.0):
    beta_int=int_diff30/30.0
    rej=0
    for _ in range(nsim):
        n=n_trt+n_ctl
        trt=np.r_[np.ones(n_trt),np.zeros(n_ctl)]
        csi=rng.normal(40,14,n); csic=csi-40
        y = -main*trt - beta_int*trt*csic + 0.03*csic + rng.normal(0,sd_resid,n)
        X=np.c_[np.ones(n),trt,csic,trt*csic]
        beta,res,rk,sv=np.linalg.lstsq(X,y,rcond=None)
        resid=y-X@beta; s2=resid@resid/(n-4)
        cov=s2*np.linalg.inv(X.T@X); se=np.sqrt(cov[3,3])
        t=beta[3]/se
        if abs(t)>stats.t.ppf(1-alpha/2,n-4): rej+=1
    return rej/nsim
for (nt,nc,label) in [(77,77,'2-arm 77/77 (A vs Sham)'),(77,38,'A vs UC 77/38'),(65,65,'2-arm 65/65'),(115,115,'2-arm 115/115'),(200,200,'2-arm 200/200')]:
    for d30 in [1.0,1.5,2.0]:
        print(f"{label}: interaction diff over 30 CSI pts = {d30:.1f} NRS -> power {sim_interaction(nt,nc,d30):.2f}")

print("="*80)
print("TABLE 6. Subgroup-enriched: CSI>=40 prevalence 0.45 -> screening multiplier")
for prev in [0.35,0.45,0.55]:
    print(f"prevalence {prev:.2f}: need {1/prev:.2f}x screened-eligible per randomized")

print("="*80)
print("TABLE 7. Recruitment simulation: months to reach N (90th percentile), Poisson arrivals")
def months_to_N(rate,N,nsim=4000):
    out=[]
    for _ in range(nsim):
        m=0;c=0
        while c<N:
            m+=1; c+=rng.poisson(rate)
        out.append(m)
    return np.percentile(out,[50,90])
for rate in [3,4,5,6,8,10,12]:
    row=[]
    for N in [120,150,180,192,240,300]:
        p50,p90=months_to_N(rate,N)
        row.append(f"N{N}:{p50:.0f}/{p90:.0f}")
    print(f"rate {rate:2d}/mo -> "+"  ".join(row))
```

```python
import numpy as np
from statsmodels.stats.power import TTestIndPower
def pw(delta,sd,n1,n2,rho=0.5,alpha=0.05):
    es=delta/(sd*np.sqrt(1-rho**2))
    return TTestIndPower().power(effect_size=es,nobs1=n1,alpha=alpha,ratio=n2/n1,alternative='two-sided')
print("CHAMPION 3-arm 2:2:1; dropout 15% -> evaluable")
for N in [180,200,210,220,240]:
    a=int(np.floor(N*0.4*0.85)); u=int(np.floor(N*0.2*0.85))
    print(f"N={N} enrolled -> EA/Sham/UC = {int(N*0.4)}/{int(N*0.4)}/{int(N*0.2)}; evaluable {a}/{a}/{u}")
    for sd in [2.0,2.2,2.5]:
        s=f"   SD={sd}: EA-vs-UC d=1.2:{pw(1.2,sd,a,u):.2f} d=1.3:{pw(1.3,sd,a,u):.2f} d=1.5:{pw(1.5,sd,a,u):.2f} | EA-vs-Sham d=0.8:{pw(0.8,sd,a,a):.2f} d=0.9:{pw(0.9,sd,a,a):.2f} d=1.0:{pw(1.0,sd,a,a):.2f} d=1.2:{pw(1.2,sd,a,a):.2f}"
        print(s)
print()
print("RUNNER-UP 2-arm EA vs UC, dropout 15%")
for N in [100,110,120,140]:
    a=int(np.floor(N*0.5*0.85))
    print(f"N={N} -> evaluable {a}/{a}: SD2.2 d=1.0:{pw(1.0,2.2,a,a):.2f} d=1.2:{pw(1.2,2.2,a,a):.2f} d=1.3:{pw(1.3,2.2,a,a):.2f} d=1.5:{pw(1.5,2.2,a,a):.2f} | SD2.5 d=1.3:{pw(1.3,2.5,a,a):.2f}")
print()
print("VARIANT 2-arm EA vs Sham, dropout 15%")
for N in [150,160,180,200]:
    a=int(np.floor(N*0.5*0.85))
    print(f"N={N} -> evaluable {a}/{a}: SD2.2 d=0.8:{pw(0.8,2.2,a,a):.2f} d=1.0:{pw(1.0,2.2,a,a):.2f} d=1.2:{pw(1.2,2.2,a,a):.2f} | SD2.5 d=1.0:{pw(1.0,2.5,a,a):.2f}")
print()
print("Minimal detectable difference (80% power) for champion N=200, SD=2.2, rho=0.5")
from scipy.optimize import brentq
a=68;u=34
print(" EA vs UC  MDD:", round(brentq(lambda d: pw(d,2.2,a,u)-0.8,0.1,4),2))
print(" EA vs Sham MDD:", round(brentq(lambda d: pw(d,2.2,a,a)-0.8,0.1,4),2))
print(" Sham vs UC MDD:", round(brentq(lambda d: pw(d,2.2,a,u)-0.8,0.1,4),2))
```
