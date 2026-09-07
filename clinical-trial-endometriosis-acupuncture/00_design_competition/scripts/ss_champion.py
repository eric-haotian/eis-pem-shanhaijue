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
