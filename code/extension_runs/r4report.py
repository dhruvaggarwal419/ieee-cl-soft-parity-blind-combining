import json, numpy as np, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def J(f):
    try: return json.load(open(f))
    except Exception: return None
D=J('r4_D_baselines.json'); F=J('r4_F_selection.json'); G=J('r4_G_norm.json'); B2=J('r4_B2_mu4.json')
if D:
    for cname,cv in D['codes'].items():
        print('=== D: %s (n=%d, M=%d, rate %.3f) ==='%(cname,cv['n'],cv['M'],cv['rate']))
        ks=['soft5','soft2p','softm','softinf','lp','fang5','fangm','m2m4','dd','genie']
        print('%7s'%'snr_dB',' '.join('%8s'%k for k in ks),'%9s %9s %9s'%('crbDA','crbNDA','crbSYN'))
        for lab,fld in [('RMSE dB','rmse_db'),('bias dB','bias_db')]:
            print(' -- '+lab)
            for r in cv['rows']:
                print('%7.1f'%r['snr_db'],' '.join('%8.3f'%r[k][fld] for k in ks),
                      '%9.3f %9.3f %9.3f'%(r['crb_da_db'],r['crb_nda_db'],r['crb_syndrome_db']) if fld=='rmse_db' else '')
        print(' -- RMSE in rho')
        for r in cv['rows']:
            print('%7.1f'%r['snr_db'],' '.join('%8.4f'%r[k]['rmse_rho'] for k in ks))
        print(' -- cost us/frame, decoder iterations')
        r=cv['rows'][len(cv['rows'])//2]
        for k in ks: print('    %-9s %9.2f us  %3d iters'%(k,r[k]['us_per_frame'],r[k]['dec_iters']))
        print('    Fang table build (one-off): %.1f s'%cv['table_build']['fang5_table_s'])
        print(' -- LP vs a->inf soft inversion: max|diff| in rho / corr')
        for r in cv['rows']: print('    snr %5.1f  %.3e  %.6f'%(r['snr_db'],r['lp_vs_softinf_max_abs_diff_rho'],r['lp_vs_softinf_corr']))
        print()
if F:
    print('=== F: blind ratio search, total combined Eb/N0 fixed at %.1f dB ==='%F['sweep'][0]['eb_total_db'])
    keys=['oracle','egc','best1','G5','G11','G31','G61','G121','G241','G481','G31_parab','G31_smooth','G31_shrink0.50','G31_shrink0.70','G31_shrink0.85']
    print('%7s'%'gap_dB',' '.join('%9s'%k for k in keys))
    for r in F['sweep']: print('%7.1f'%r['gap_db'],' '.join('%9.5f'%r['fer'].get(k,float('nan')) for k in keys))
    print('-- log(r_hat/r_oracle) bias / std')
    gk=[k for k in keys if k.startswith('G')]
    for r in F['sweep']:
        print('%7.1f'%r['gap_db'],' '.join('%+.2f/%.2f'%(r['logratio'][k]['bias'],r['logratio'][k]['std']) for k in gk))
    print('-- grid-31 neighbour correlation of S')
    for r in F['sweep']: print('   gap %5.1f  corr(k,k+1)=%.4f  corr(k,k+2)=%.4f'%(r['gap_db'],r['grid31_neighbour_corr'],r['grid31_lag2_corr']))
    print()
if G:
    print('=== G: normalisation ===')
    ks=['oracle','rms','det','genie','none','rms_scaled0.8','rms_scaled1.25','egc']
    print('%7s'%'gap_dB',' '.join('%14s'%k for k in ks))
    for r in G['rows']: print('%7.1f'%r['gap_db'],' '.join('%14.5f'%r['fer'].get(k,float('nan')) for k in ks))
    print('-- fraction of frames whose chosen grid index equals the sample-rms choice')
    for r in G['rows']: print('%7.1f'%r['gap_db'],' '.join('%14.3f'%r['agree_with_rms'].get(k,float('nan')) for k in ks if k in r['agree_with_rms']))
    print('-- mean |index shift| vs rms')
    for r in G['rows']: print('%7.1f'%r['gap_db'],' '.join('%14.2f'%r['mean_abs_index_shift_vs_rms'].get(k,float('nan')) for k in ks if k in r['mean_abs_index_shift_vs_rms']))
    print('-- log(r_hat/r_oracle) bias/std')
    for r in G['rows']: print('%7.1f'%r['gap_db'],' '.join('%14s'%('%+.2f/%.2f'%(r['logratio'][k]['bias'],r['logratio'][k]['std'])) for k in ks if k in r['logratio']))
    print()
if B2:
    print('=== B2: exact mu4 cost and validation ===')
    print('%-22s %5s %5s %7s %7s %10s %10s %9s %9s %12s %12s'%('code','M','dcmax','depdeg','depmax','#triples','#quads','t_enum','t_eval','kurt_exact','kurt_MC'))
    for r in B2['cost']:
        print('%-22s %5d %5d %7.1f %7d %10s %10s %9s %9s %12s %12.4f'%(r['code'],r['M'],r['dc_max'],r['dep_deg_mean'],r['dep_deg_max'],
            r.get('n_triples','-'),r.get('n_quads','-'),
            ('%.2f'%(r.get('t_triples_s',0)+r.get('t_quads_s',0))) if r['feasible'] else '-',
            ('%.1f'%r['t_eval_s']) if r['feasible'] else '-',
            ('%.4f'%r['kurt_exact_rho0p6']) if r['kurt_exact_rho0p6'] else 'infeasible', r['kurt_mc_rho0p6']))
    print('MC kurtosis stderr = %.4f (N=%d)'%(B2['cost'][0]['kurt_mc_stderr'],B2['cost'][0]['N_mc']))
    print()
    print('-- confidence sets on the (256,126) code, a=5')
    print('%7s %8s %8s | %s'%('eb_dB','sd','kurt',' '.join('%22s'%('k=%s: cheb/m4/gauss'%k) for k in ['2','3','4'])))
    for r in B2['conf_sets']:
        print('%7.1f %8.5f %8.4f | %s'%(r['eb_db'],r['sd'],r['kurt'],
            ' '.join('%22s'%('%.4f/%.4f/%.5f'%(r['cheb_k%s'%k],r['m4_k%s'%k],r['gauss_k%s'%k])) for k in ['2','3','4'])))
    print('-- MC validation')
    for r in B2['mu4_validation']:
        print('   eb %.0f N=%d  mean %.6f/%.6f  sd %.6f/%.6f  kurt %.4f/%.4f (+-%.4f)  emp fail k=2,3,4: %.4f %.5f %.6f'%(
            r['eb_db'],r['N'],r['mean_exact'],r['mean_emp'],r['sd_exact'],r['sd_emp'],r['kurt_exact'],r['kurt_emp'],r['kurt_emp_stderr'],
            r['emp_fail_k2'],r['emp_fail_k3'],r['emp_fail_k4']))
