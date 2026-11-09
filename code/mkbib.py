import requests,re,json
refs=[('lechner','10.1109/lcomm.2013.091113.131646'),('saeedi07','10.1109/tcomm.2006.887488'),('saeedi09','10.1109/tcomm.2009.0901.060118'),
('ru01','10.1109/18.910577'),('hag','10.1109/18.485714'),('chen02','10.1109/26.990903'),('xu14','10.1109/tbc.2014.2364532'),
('pauz','10.1109/26.871393'),('bellili','10.1109/tsp.2014.2328328'),('boyer','10.1109/tcomm.2009.02.050081'),('cui','10.1109/glocom.2012.6503699'),
('he17','10.1109/vtcspring.2017.8108433'),('xia','10.1109/lcomm.2013.051313.130462'),('larsson','10.1109/wcnc.2010.5506707'),
('chase','10.1109/tcom.1985.1096314'),('brennan','10.1109/jrproc.1959.287136'),('nach','10.1109/jstsp.2017.2788405'),('dai','10.1109/jsac.2021.3078488'),
('cam20','10.1109/tcomm.2020.3002915'),('kwak','10.1109/jsac.2025.3531553'),('rovella','10.1109/globecom54140.2023.10436980'),
('jeong','10.1109/tgcn.2026.3659107'),('rk18','10.1109/mcom.2018.1700839'),('shl','10.1109/jproc.2023.3247480'),('gappmair','10.1109/tcomm.2008.060275')]
bib=[];tab=['| key | DOI | title (Crossref) | authors | venue | year | vol/pages |','|---|---|---|---|---|---|---|']
for k,d in refs:
    m=requests.get('https://api.crossref.org/works/'+d,headers={'User-Agent':'refcheck (mailto:dhruv_a2@cs.iitr.ac.in)'}).json()['message']
    au=m.get('author',[]); names=' and '.join(f"{a.get('family','')}, {a.get('given','')}" for a in au)
    ct=(m.get('container-title') or [''])[0]; yr=m['issued']['date-parts'][0][0]
    ty='article' if m['type']=='journal-article' else 'inproceedings'
    e=f"@{ty}{{{k},\n  author={{{names}}},\n  title={{{m['title'][0]}}},\n  {'journal' if ty=='article' else 'booktitle'}={{{ct}}},\n  year={{{yr}}},\n"
    if m.get('volume'): e+=f"  volume={{{m['volume']}}},\n"
    if m.get('issue'): e+=f"  number={{{m['issue']}}},\n"
    if m.get('page'): e+=f"  pages={{{m['page'].replace('-','--')}}},\n"
    e+=f"  doi={{{d}}}\n}}\n"; bib.append(e)
    tab.append(f"| {k} | {d} | {m['title'][0]} | {', '.join(a.get('family','') for a in au[:5])} | {ct} | {yr} | {m.get('volume','')}/{m.get('page','')} |")
open('paper/refs.bib','w').write('\n'.join(bib)); open('refs_table.md','w').write('\n'.join(tab))
print('\n'.join(tab))
