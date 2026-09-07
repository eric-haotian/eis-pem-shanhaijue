import numpy as np, itertools
crit = [
 # (name, weight, domain)
 ("科学新颖性",8,"S"),("临床重要性",9,"S"),("因果识别强度",9,"S"),("阴性结果解释价值",9,"S"),
 ("招募速度",10,"F"),("患者依从性",5,"F"),("单中心可完成性",8,"F"),("成本",4,"F"),("针灸科工作负荷",4,"F"),("随访负担",4,"F"),
 ("伦理复杂度(低=好)",4,"E"),("统计把握度",6,"E"),
 ("论文档次上限",6,"C"),("院内课题竞争力",4,"C"),("PI代表作价值",5,"C"),("升级市级/国家级潜力",5,"C"),
]
designs = {
 "D1 SHAM-2 (EA vs 假EA, 两臂)":      [3,4,4,3, 4,3,4,3,3,4, 4,3, 4,4,4,4],
 "D2 ENRICH (CSI≥40富集, 两臂假针)":  [5,3,4,2, 2,3,2,2,3,4, 3,3, 4,4,3,4],
 "D3 TRIAD (EA vs 假EA vs 常规, 2:2:1)":[4,5,5,5, 3,3,3,3,3,4, 3,4, 5,5,5,5],
 "D4 COHORT-NEST (手术队列+嵌套RCT)": [4,4,4,4, 2,3,3,2,3,2, 3,2, 4,4,4,5],
 "D5 PREVENT (围术期预防, EA vs 常规)":[3,4,3,2, 2,2,1,1,1,2, 3,2, 3,3,3,3],
 "D6 PRAGMA (务实两臂 EA+SOC vs SOC)":[2,4,2,2, 5,4,5,4,4,4, 4,4, 2,3,2,2],
}
w=np.array([c[1] for c in crit]); W=w.sum()
print("weights sum", W)
print("| 设计 | 加权总分(满分%d) | 百分制 | 科学(35) | 可行(35) | 伦理/统计(10) | 职业/战略(20) |"%(5*W))
print("|---|---|---|---|---|---|---|")
dom={"S":[], "F":[], "E":[], "C":[]}
for i,c in enumerate(crit): dom[c[2]].append(i)
res={}
for k,v in designs.items():
    v=np.array(v); tot=(v*w).sum(); res[k]=tot
    parts=[ (v[idx]*w[idx]).sum()/(5*w[idx].sum())*100 for d,idx in dom.items()]
    print(f"| {k} | {tot} | {tot/(5*W)*100:.0f} | {parts[0]:.0f}% | {parts[1]:.0f}% | {parts[2]:.0f}% | {parts[3]:.0f}% |")
print()
print("Sensitivity: alternative weightings")
alts = {
 "均等权重": np.ones(len(crit)),
 "可行性优先(F×2)": np.array([c[1]*(2 if c[2]=="F" else 1) for c in crit]),
 "科学优先(S×2)": np.array([c[1]*(2 if c[2]=="S" else 1) for c in crit]),
 "职业优先(C×2)": np.array([c[1]*(2 if c[2]=="C" else 1) for c in crit]),
}
for name,ww in alts.items():
    ranking=sorted(designs.items(), key=lambda kv: -(np.array(kv[1])*ww).sum())
    print(f"{name}: "+" > ".join([f"{k.split()[0]}({(np.array(v)*ww).sum()/(5*ww.sum())*100:.0f})" for k,v in ranking]))
print()
print("Full score sheet (rows=criteria, cols=designs)")
print("| 指标 | 权重 | "+" | ".join(k.split()[0] for k in designs)+" |")
print("|---|---|"+"---|"*len(designs))
for i,c in enumerate(crit):
    print(f"| {c[0]} | {c[1]} | "+" | ".join(str(v[i]) for v in designs.values())+" |")
