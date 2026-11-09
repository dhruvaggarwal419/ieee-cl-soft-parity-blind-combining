import json, os, sys
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
SD = ['-5', '0', '5', '10', '15', '20']
L = lambda f: json.load(open(os.path.join(D, f)))
P = print


def hdr(t):
    P('\n' + '=' * 78); P(t); P('=' * 78)


# ============================== A ==========================================
A = L('r2_A_guard.json')
hdr('(A1) gap-report bias (noise units) vs GUARD WIDTH, 64-ant array, N=64, 3 paths')
P('%-6s %s' % ('guard', ' '.join('%8s' % ('SNR' + s) for s in SD)))
for g, v in A['A1_guard_width'].items():
    P('%-6s %s' % ('g=' + g, ' '.join('%8.2f' % v['gap'][s]['bias'][0] for s in SD)))
P('--- std of (report - truth), same cells ---')
for g, v in A['A1_guard_width'].items():
    P('%-6s %s' % ('g=' + g, ' '.join('%8.2f' % v['gap'][s]['std'][0] for s in SD)))
P('--- reference: other reports at g=3 ---')
v = A['A1_guard_width']['3']
for m in ['raw', 'debias', 'confirm', 'gap_noguard']:
    P('%-12s %s' % (m, ' '.join('%8.2f' % v[m][s]['bias'][0] for s in SD)))
P('beam-selection error rate: ' + ' '.join('%8.3f' % v['_aux'][s]['beamerr'][0] for s in SD))

for key, title, lab in [('A2_separation', '(A2) gap bias vs ANGULAR SEPARATION of nearest side path (beamwidths), pside=0.2, g=3', 'sep'),
                        ('A3_spread', '(A3) gap bias vs ANGULAR SPREAD of dominant cluster (beamwidths, 8 subpaths), g=3', 'spread'),
                        ('A4_sll', '(A4) gap bias vs CODEBOOK SIDELOBE LEVEL (Chebyshev taper, dB; 0=untapered -13.3dB), g=3', 'SLL')]:
    hdr(title)
    P('%-10s %s' % (lab, ' '.join('%8s' % ('SNR' + s) for s in SD)))
    for k, v in A[key].items():
        P('%-10s %s' % (k, ' '.join('%8.2f' % v['gap'][s]['bias'][0] for s in SD)))
    P('--- std ---')
    for k, v in A[key].items():
        P('%-10s %s' % (k, ' '.join('%8.2f' % v['gap'][s]['std'][0] for s in SD)))

hdr('(A5) explicit bias correction  corrected = gap + rho*shat,  shat = max(debias,0)')
P('rho_main uses only (g,N,M,SLL);  rho_full also uses side-path power/count')
P('%-8s %10s %10s' % ('cfg', 'rho_main', 'rho_full'))
for n, v in A['A5_rho'].items():
    P('%-8s %10.5f %10.5f' % (n, v['rho_main'], v['rho_full']))
P('\nrho_main vs guard width (base codebook, M=64,N=64):')
P('  ' + json.dumps({k: round(x, 5) for k, x in A['A5_rho']['base']['rho_by_guard'].items()}))
P('\nresidual bias after correction (noise units)')
P('%-8s %-14s %s' % ('cfg', 'variant', ' '.join('%8s' % ('SNR' + s) for s in SD)))
for n, v in A['A5_correction'].items():
    for tag, lab in [('gap_bias', 'uncorrected'), ('corr_main_bias', '+rho_main'), ('corr_full_bias', '+rho_full')]:
        P('%-8s %-14s %s' % (n, lab, ' '.join('%8.2f' % v[s][tag][0] for s in SD)))

# fitted power-law correction: bias ~ -c0 * (rho_full*shat)^p, fitted on base only
try:
    xs, ys = [], []
    for s in SD:
        d = A['A5_correction']['base'][s]
        ell = A['A5_rho']['base']['rho_full'] * d['Eshat'][0]
        b = -d['gap_bias'][0]
        if ell > 1e-6 and b > 0.05:
            xs.append(np.log(ell)); ys.append(np.log(b))
    pfit, c0 = np.polyfit(xs, ys, 1)
    c0 = float(np.exp(c0)); pfit = float(pfit)
    P('\nfitted on BASE config only:  bias_hat = -%.3f * (rho_full*shat)^%.3f' % (c0, pfit))
    P('residual bias after this fitted correction, transferred unchanged to other configs:')
    P('%-8s %s' % ('cfg', ' '.join('%8s' % ('SNR' + s) for s in SD)))
    for n, v in A['A5_correction'].items():
        row = []
        for s in SD:
            ell = A['A5_rho'][n]['rho_full'] * v[s]['Eshat'][0]
            row.append(v[s]['gap_bias'][0] + c0 * ell ** pfit)
        P('%-8s %s' % (n, ' '.join('%8.2f' % r for r in row)))
except Exception as e:
    P('fit failed: %r' % (e,))

# ============================== B ==========================================
B = L('r2_B_outage.json')
hdr('(B1) paper closed form  P=e^{-1/(1+r)}(1+r)/(1+r+m r)  vs empirical, E1 model, gap report, m=9')
for N in [16, 64, 256, 1024]:
    k = 'm9_N%d' % N
    if k not in B['B1_closed_form']:
        continue
    P('\n--- N=%d ---' % N)
    P('%6s %9s %9s %9s %9s %9s %9s' % ('SNRdB', 'closed', 'emp_TOT', 'emp|corr', 'err_TOT', 'P(wrong)', 'wrongshr'))
    for s, v in B['B1_closed_form'][k].items():
        P('%6s %9.4f %9.4f %9.4f %+9.4f %9.4f %9.3f' % (
            s, v['closed_form'], v['emp_total'][0], v['emp_out_given_correct'][0],
            v['err_vs_total'], v['p_wrong'][0], v['wrong_share_of_outage']))
P('\nmax |closed - empirical TOTAL| and max |closed - empirical|correct beam| over the SNR grid:')
P('%-12s %14s %14s %14s' % ('case', 'max err TOT', 'max err |corr', 'crossover SNR'))
for k, rows in B['B1_closed_form'].items():
    et = max(abs(v['err_vs_total']) for v in rows.values())
    ec = max(abs(v['err_vs_correct']) for v in rows.values())
    cross = None
    ss = sorted(rows, key=lambda x: float(x))
    for s in ss:
        if rows[s]['wrong_share_of_outage'] < 0.5:
            cross = s
            break
    P('%-12s %14.4f %14.4f %14s' % (k, et, ec, 'wrong<50%% above %s dB' % cross))

hdr('(B2) array model: outage decomposition by correct/wrong beam, per fading model')
for name, row in B['B2_array'].items():
    P('\n--- %s ---  (knobs tuned per case)' % name)
    P('%-12s %6s %6s %s' % ('method', 'gp', 'worstO', ' '.join('%7s' % ('S' + s) for s in SD)))
    for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
        d = row.get(m)
        if d is None:
            P('%-12s infeasible' % m); continue
        P('%-12s %6.3f %6.3f %s' % (m, d['gp'][0], d['worst_out'][0],
                                    ' '.join('%7.3f' % d['per_snr'][s]['total'] [0]for s in SD)))
    d = row.get('gap')
    if d:
        P('  gap   P(wrong beam)      ' + ' '.join('%7.3f' % d['per_snr'][s]['p_wrong'] for s in SD))
        P('  gap   P(out|correct)     ' + ' '.join('%7.3f' % d['per_snr'][s]['out_given_correct'] for s in SD))
        P('  gap   P(out|wrong)       ' + ' '.join('%7.3f' % d['per_snr'][s]['out_given_wrong'] for s in SD))
        P('  gap   wrong-beam share   ' + ' '.join('%7.3f' % d['per_snr'][s]['wrong_share'] for s in SD))
    P('  beam-sel error rate  ' + ' '.join('%7.3f' % row['beamerr'][s][0] for s in SD))
    P('  gap bias             ' + ' '.join('%7.2f' % row['bias']['gap'][s][0] for s in SD))
    P('  debias bias          ' + ' '.join('%7.2f' % row['bias']['debias'][s][0] for s in SD))

# ============================== C ==========================================
C = L('r2_C_tradeoff.json')
hdr('(C) full trade-off, one block per array config (knobs tuned per config, target worst outage 0.10)')
for name, row in C.items():
    if name == 'meta':
        continue
    P('\n--- %s ---' % name)
    P('  beam-sel error rate ' + ' '.join('%7.3f' % row['beamerr'][s][0] for s in SD))
    if 'beamloss_dB' in row:
        P('  beamforming loss dB ' + ' '.join('%7.2f' % row['beamloss_dB'][s][0] for s in SD))
    P('%-12s %7s %8s %8s %8s %8s %8s %8s' %
      ('method', 'gp', 'worstOut', 'falseTx', 'missedTx', 'P(tx)', 'O|tx', 'n_cal'))
    for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
        d = row['methods'].get(m)
        if d is None:
            P('%-12s infeasible' % m); continue
        P('%-12s %7.3f %8.3f %8.4f %8.4f %8.3f %8.3f %8s' %
          (m, d['gp'][0], d['worst_out'][0], d['false_tx'][0], d['missed_tx'][0],
           d['p_tx'][0], d['out_cond_tx'][0], row['calibration'][m]['n_cal_needed']))

hdr('(C) calibration cost detail: fraction of 20 independent calibration draws whose knobs '
    'meet worst-case outage<=0.11 on a large held-out set')
for name, row in C.items():
    if name == 'meta':
        continue
    ncs = [k for k in row['calibration']['gap'] if k != 'n_cal_needed']
    P('\n--- %s ---  n_cal per SNR point: %s' % (name, ncs))
    for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
        P('%-12s %s' % (m, ' '.join('%6.2f' % row['calibration'][m][n]['frac_meeting_target'] for n in ncs)))

# ============================== D ==========================================
Dd = L('r2_D_noise.json')
hdr('(D) unknown noise power. knobs tuned with a perfect sigma^2, then deployed with error.')
for cname, sec in Dd.items():
    if cname == 'meta':
        continue
    P('\n--- %s : worst-case outage (target 0.10), knobs TRANSFERRED from clean ---' % cname)
    cases = [k for k in sec if k not in ('knobs_clean',)]
    P('%-12s %s' % ('method', ' '.join('%9s' % c for c in cases)))
    for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
        P('%-12s %s' % (m, ' '.join(
            ('%9.3f' % sec[c]['methods'][m]['transfer']['worst_out'][0])
            if sec[c]['methods'][m]['transfer'] else '%9s' % '-' for c in cases)))
    P('--- goodput, same cells ---')
    for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
        P('%-12s %s' % (m, ' '.join(
            ('%9.3f' % sec[c]['methods'][m]['transfer']['gp'][0])
            if sec[c]['methods'][m]['transfer'] else '%9s' % '-' for c in cases)))
    P('--- bias at 20 dB SNR ---')
    for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
        P('%-12s %s' % (m, ' '.join('%9.2f' % sec[c]['methods'][m]['bias']['20'][0] for c in cases)))
    P('--- bias at 0 dB SNR ---')
    for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
        P('%-12s %s' % (m, ' '.join('%9.2f' % sec[c]['methods'][m]['bias']['0'][0] for c in cases)))
    P('--- worst-case outage after RETUNING under the same noise error ---')
    for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
        P('%-12s %s' % (m, ' '.join(
            ('%9.3f' % sec[c]['methods'][m]['retuned']['worst_out'][0])
            if sec[c]['methods'][m]['retuned'] else '%9s' % '-' for c in cases)))

# ============================== E ==========================================
E = L('r2_E_mobility.json')
hdr('(E) mobility / channel aging: AR(1) correlation rho between training and data')
for sname, sec in E.items():
    if sname == 'meta':
        continue
    P('\n--- %s ---' % sname)
    rh = [k for k in sec if k.startswith('rho')]
    for tag, lab in [('transfer_from_static', 'knobs tuned at rho=1 (no retune)'),
                     ('retuned', 'knobs retuned at each rho')]:
        P('  worst-case outage, %s' % lab)
        P('  %-12s %s' % ('method', ' '.join('%9s' % r for r in rh)))
        for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
            P('  %-12s %s' % (m, ' '.join(
                ('%9.3f' % sec[r]['methods'][m][tag]['worst_out'][0])
                if sec[r]['methods'][m][tag] else '%9s' % '-' for r in rh)))
        P('  goodput, %s' % lab)
        for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
            P('  %-12s %s' % (m, ' '.join(
                ('%9.3f' % sec[r]['methods'][m][tag]['gp'][0])
                if sec[r]['methods'][m][tag] else '%9s' % '-' for r in rh)))

# ============================== F ==========================================
F = L('r2_F_olla.json')
hdr('(F) OLLA (ACK/NACK outer loop on the EXISTING raw report) vs the five static reports')
P('config: 64-ant array, N=64, 3 paths, target outage 0.10; OLLA knobs %s' % json.dumps(F['olla_knobs']))
P('\n%-22s %7s %9s %s' % ('scheme', 'gp', 'worstOut', ' '.join('%7s' % ('S' + s) for s in SD)))
for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
    d = F['static']['methods'].get(m)
    if d is None:
        P('%-22s infeasible' % ('static ' + m)); continue
    P('%-22s %7.3f %9.3f %s' % ('static ' + m, d['gp'][0], d['worst_out'][0],
                                ' '.join('%7.3f' % o for o in d['per_snr_out'])))
for m, d in F['olla'].items():
    if d is None:
        P('%-22s infeasible' % ('OLLA on ' + m)); continue
    P('%-22s %7.3f %9.3f %s' % ('OLLA on ' + m, d['gp'][0], d['worst_out'][0],
                                ' '.join('%7.3f' % o for o in d['per_snr_out'])))
P('\nper-SNR goodput')
for m in ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']:
    d = F['static']['methods'].get(m)
    if d:
        P('%-22s %s' % ('static ' + m, ' '.join('%7.3f' % g for g in d['per_snr_gp'])))
for m, d in F['olla'].items():
    if d:
        P('%-22s %s' % ('OLLA on ' + m, ' '.join('%7.3f' % g for g in d['per_snr_gp'])))
P('\nOLLA converged offset (dB) per SNR, and transient outage')
for m, d in F['olla'].items():
    if not d:
        continue
    P('%-22s offset %s' % ('OLLA on ' + m, ' '.join('%7.2f' % o for o in d['converged_offset_dB'])))
    P('%-22s out slots 1-50   %s' % ('', ' '.join('%7.3f' % o for o in d['transient_out_first50'])))
    P('%-22s out slots 50-200 %s' % ('', ' '.join('%7.3f' % o for o in d['transient_out_50_200'])))
P('\ndone')
