import json, ast, numpy as np
from data import *
P='../paper/'
# ---------- numbers ----------
est={r['eb']:r for r in b['est']}
def pct(x): return f"{100*x:.1f}"
gA={m:gap('A',m,0.05) for m in ['oracle','blind','soft','hard','m2m4','dd','egc']}
gB={m:gap('B',m,0.02) for m in ['oracle','blind','soft','hard','m2m4','dd','egc']}
print('gapA',gA); print('gapB',gB)
def gd(g,m): return None if g[m] is None or g['oracle'] is None else g[m]-g['oracle']
gAd={m:gd(gA,m) for m in gA}; gBd={m:gd(gB,m) for m in gB}
print(gAd,gBd)
def fmt(x): return '--' if x is None else f"{x:.2f}"
# ---------- Table I ----------
e1=json.load(open(R+'exp1.json'))['validation']; e7={(r['eb'],r['a']):r for r in json.load(open(R+'exp7.json'))}
rows=[]; mer=0; ver=0
for v in e1:
    if v['eb']!=3.0: continue
    ind=e7[(v['eb'],v['a'])]['var_indep']; mer=max(mer,abs(v['mean_mc']/v['mean_th']-1)); ver=max(ver,abs(v['var_mc']/v['var_th']-1))
    a='$\\infty$' if v['a']=='inf' else f"{float(v['a']):.0f}"
    rows.append(f"{v['eb']:.0f} & {a} & {v['mean_th']:.4f} & {v['mean_mc']:.4f} & {v['var_th']*1e3:.3f} & {v['var_mc']*1e3:.3f} & {ind*1e3:.3f} \\\\ \\hline")
print('max mean err',mer,'max var err',ver)
t1="\\begin{table}[t]\n\\caption{Mean and variance of $S_a$. Variances are in units of $10^{-3}$. The last column treats the checks as independent.}\n\\label{tab:val}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{|c|c|c|c|c|c|c|}\\hline\n$E_b/N_0$ (dB) & $a$ & mean (5) & mean sim. & var. (6) & var. sim. & var. indep. \\\\ \\hline\n"+"\n".join(rows)+"\n\\end{tabular}\n\\end{table}\n"
# ---------- Table II ----------
def cell(scen,cc,m):
    k,n=get(scen,cc,m); f,h=wil(k,n); return f"{100*f:.2f}$\\pm${100*h:.2f}"
names=[('oracle','Oracle weights',None),('blind','Blind maximum (this letter)',None),('soft','Per-frame soft estimates (this letter)','soft'),('hard','Hard syndrome \\cite{lechner}','hard'),('m2m4','M2M4 \\cite{pauz}','m2m4'),('dd','Decode then ML','dd'),('egc','Equal weights',None)]
rows=[]
for m,name,ek in names:
    r=[name]
    for eb in [0,3]: r.append('--' if ek is None else pct(est[eb][ek]['rmse']))
    for cc in [1.5,2.5]: r.append(cell('A',cc,m))
    r.append('0' if m=='oracle' else fmt(gAd[m]))
    for cc in [2.0,3.0]: r.append(cell('B',cc,m))
    r.append('0' if m=='oracle' else fmt(gBd[m]))
    rows.append(' & '.join(r)+' \\\\ \\hline')
t2="\\begin{table*}[t]\n\\caption{Single-frame estimation error and frame error rate (FER) of two combined frames. FER values are in percent with 95\\% half-widths. In scenario A the weak frame is 6 dB below the strong frame and the stated SNR is that of the strong frame. In scenario B both frames have independent SNR values uniform within $\\pm4$ dB of the stated mean. The gap columns give the SNR loss against oracle weights at FER 5\\% (A) and 2\\% (B) by linear interpolation of $\\log_{10}$FER.}\n\\label{tab:main}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}\\hline\n& \\multicolumn{2}{c|}{RMSE of $\\hat\\rho$ (\\%)} & \\multicolumn{3}{c|}{Scenario A, strong frame at} & \\multicolumn{3}{c|}{Scenario B, mean SNR}\\\\ \\hline\nWeight rule & 0 dB & 3 dB & 1.5 dB & 2.5 dB & gap (dB) & 2.0 dB & 3.0 dB & gap (dB)\\\\ \\hline\n"+"\n".join(rows)+"\n\\end{tabular}\n\\end{table*}\n"
open(P+'table1.tex','w').write(t1); open(P+'table2.tex','w').write(t2)
json.dump(dict(gA=gAd,gB=gBd,mer=mer,ver=ver),open(R+'summary.json','w'),indent=1)
