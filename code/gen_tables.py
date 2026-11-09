import json, ast, numpy as np
R='../results/'; P='../paper/'
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c,h
# Table I
e1=json.load(open(R+'exp1.json'))['validation']; e7={(r['eb'],r['a']):r for r in json.load(open(R+'exp7.json'))}
rows=[]
for v in e1:
    k=(v['eb'],v['a']); ind=e7[k]['var_indep']
    a='$\\infty$' if v['a']=='inf' else f"{float(v['a']):.0f}"
    rows.append(f"{v['eb']:.0f} & {a} & {v['mean_th']:.4f} & {v['mean_mc']:.4f} & {v['var_th']*1e3:.3f} & {v['var_mc']*1e3:.3f} & {ind*1e3:.3f} \\\\ \\hline")
t1="\\begin{table}[t]\n\\caption{Mean and variance of $S_a$ at two SNR values. Variances are in units of $10^{-3}$. The last column treats the checks as independent.}\n\\label{tab:val}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{|c|c|c|c|c|c|c|}\\hline\n$E_b/N_0$ (dB) & $a$ & mean (5) & mean sim. & var. (6) & var. sim. & var. indep. \\\\ \\hline\n"+"\n".join(rows)+"\n\\end{tabular}\n\\end{table}\n"
open(P+'table1.tex','w').write(t1)
# Table II
e3=json.load(open(R+'exp3b.json')); nf=e3['nf']
estd={r['eb']:r for r in e3['est']}
try: e3c=json.load(open(R+'exp3c.json')); B=e3c['B']; nfB=e3c['nf']
except Exception: B=[]; nfB=1
A={p['c']:p for p in e3['A']}; Bd={p['c']:p for p in B}
meth=[('oracle','Oracle weights',None),('soft','Soft parity (this letter)','soft'),('hard','Hard syndrome \\cite{lechner}','hard'),('m2m4','M2M4 \\cite{pauz}','m2m4'),('dd','Decode then ML','dd'),('egc','Equal weights',None)]
Acols=[2.0,3.0]; Bcols=[2.5,3.5]
def cell(d,c,k,n):
    if c not in d: return '--'
    f,h=wil(d[c]['err'][k],n); return f"{100*f:.2f}$\\pm${100*h:.2f}"
rows=[]
for k,name,ek in meth:
    r=[name]
    for eb in [0,3]: r.append('--' if ek is None else f"{100*estd[eb][ek]['rmse']:.1f}")
    for c in Acols: r.append(cell(A,c,k,nf))
    for c in Bcols: r.append(cell(Bd,c,k,nfB))
    rows.append(' & '.join(r)+' \\\\ \\hline')
t2="\\begin{table*}[t]\n\\caption{Single-frame estimation error and frame error rate of two combined frames. Error rates are in percent with 95\\% half-widths. In scenario A the weak frame is 6 dB below the strong frame. In scenario B both frames have independent SNR values uniform within $\\pm4$ dB of the stated mean.}\n\\label{tab:main}\n\\centering\\footnotesize\n\\begin{tabular}{|l|c|c|c|c|c|c|}\\hline\n& \\multicolumn{2}{c|}{RMSE of $\\hat\\rho$ (\\%)} & \\multicolumn{2}{c|}{FER (\\%), scenario A} & \\multicolumn{2}{c|}{FER (\\%), scenario B}\\\\ \\hline\n"+"Weight rule & 0 dB & 3 dB & "+" & ".join(f"{c:.1f} dB" for c in Acols)+" & "+" & ".join(f"{c:.1f} dB" for c in Bcols)+" \\\\ \\hline\n"+"\n".join(rows)+"\n\\end{tabular}\n\\end{table*}\n"
open(P+'table2.tex','w').write(t2); print(t2)
