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
