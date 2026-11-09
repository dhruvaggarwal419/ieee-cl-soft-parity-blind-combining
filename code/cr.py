import sys,requests,json
qs=open(sys.argv[1]).read().strip().split('\n')
for q in qs:
    r=requests.get('https://api.crossref.org/works',params={'query.bibliographic':q,'rows':2,'mailto':'dhruv_a2@cs.iitr.ac.in'},timeout=30).json()['message']['items']
    print('Q:',q)
    for it in r[:2]:
        au=', '.join(a.get('family','') for a in it.get('author',[])[:4])
        print('  ',it.get('DOI'),'|',(it.get('title') or [''])[0][:90],'|',au,'|',(it.get('container-title') or [''])[0][:50],'|',it.get('issued',{}).get('date-parts',[[None]])[0][0])
