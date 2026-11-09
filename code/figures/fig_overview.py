import json,numpy as np,torch,sys
sys.path.insert(0,'.')
from lib import *
from diagram_kit import *
from matplotlib.ticker import NullFormatter
torch.manual_seed(3); np.random.seed(3)
H=np.load('H.npy'); n=256; M=H.shape[0]; R=(n-gf2_rank(H))/n
Hidx=torch.tensor(np.stack([np.nonzero(r)[0] for r in H]))
rho_t=lambda eb:1/(1+1/(2*R*10**(eb/10)))
rho=float(rho_t(3.0)); A=5.0
y=np.sqrt(rho)+np.sqrt(1-rho)*torch.randn(20000,n)
S=syndrome_stat(y,Hidx,A).double().numpy(); mu,var=stat_mean_var(H,rho,A); sd=np.sqrt(var)
g=np.linspace(0.05,0.97,60); mug=[stat_mean_var(H,r,A)[0] for r in g]
r1,r2=float(rho_t(3.0)),float(rho_t(-3.0))
y1=np.sqrt(r1)+np.sqrt(1-r1)*torch.randn(1,n); y2=np.sqrt(r2)+np.sqrt(1-r2)*torch.randn(1,n)
RG=np.geomspace(0.1,10,61); Sr=[]
for r in RG:
    z=y1+r*y2; z=z/z.pow(2).mean(1,keepdim=True).sqrt(); Sr.append(float(syndrome_stat(z,Hidx,A)))
w=lambda r:2*np.sqrt(r)/(1-r); ro=w(r2)/w(r1)
F=json.load(open('r4_F_selection.json'))['sweep']; gaps=[r['gap_db'] for r in F]
D=Diagram(7.16,3.3,100,46)
Y0,YH=10.5,32
D.group(0.8,Y0,15,YH,'Received frames',c='in',fs=6.4)
D.group(17.5,Y0,27.5,YH,'Soft parity statistic',c='stat',fs=6.4)
D.group(46.5,Y0,23,YH,'Exact moments (proved)',c='theory',fs=6.4)
D.group(71.5,Y0,27.7,YH,'Blind combining',c='dec',fs=6.4)
for yy,lab,yc in [(y1,'Frame 1: strong',27.2),(y2,'Frame 2: weak',13.0)]:
    ax=D.inset(2.4,yc,11.8,7.8); ax.bar(range(60),np.sign(yy[0,:60].numpy()),color='#6C8EBF',width=0.9); ax.plot(range(60),yy[0,:60].numpy(),color='#B85450',lw=0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab,fontsize=5.2,pad=1.5)
D.text(8.3,24.4,'blue: signs, red: values\nSNR unknown',fs=4.4)
# statistic group
D.box(19,30,7.6,6.2,'$t_i=\\tanh$\n$(a\\,y_i/2)$','stat',5.6)
ax=D.inset(28.2,29.6,7.2,7.2); xs=np.linspace(-3,3,80); ax.plot(xs,np.tanh(A*xs/2),color='#2E7D32',lw=1); ax.set_xticks([]); ax.set_yticks([]); ax.set_title('soft sign',fontsize=5,pad=1)
D.text(39.8,33.2,'$a\\to\\infty$:\nhard sign',fs=5.0,color='#B85450')
ax=D.inset(18.8,13.5,11.2,12.5); ax.set_xlim(0,1); ax.set_ylim(-0.12,1); ax.axis('off')
vx=np.linspace(0.05,0.95,9); cx=[0.2,0.5,0.8]
for ci,c in enumerate(cx):
    nb=[0,1,2,3,4,5] if ci==0 else ([3,4,5,6,7,8] if ci==1 else [0,2,4,6,7,8])
    for v in nb: ax.plot([c,vx[v]],[0.82,0.18],color='#999',lw=0.5)
ax.scatter(vx,[0.18]*9,s=14,c='#6C8EBF',ec='k',lw=0.4,zorder=3); ax.scatter(cx,[0.82]*3,s=24,marker='s',c='#82B366',ec='k',lw=0.4,zorder=3)
ax.text(0.5,-0.08,'values  and  parity checks',fontsize=4.8,ha='center')
D.box(32,14,11.5,11.5,'product of the $t_i$\nover the bits of\neach check, then\nmean over $M$\nchecks gives $S_a$','stat',5.5)
D.arrow(14.3,33.1,18.9,33.1); D.arrow(14.3,19.0,18.7,19.0)
D.arrow(26.7,33.1,28.1,33.1); D.arrow(22.8,29.9,22.8,26.1); D.arrow(30.1,19.7,31.9,19.7)
# moments group
D.box(48,31.0,20,6.0,'mean $\\mu_a(\\rho)$,  exact variance,\nexact fourth moment','theory',5.4)
ax=D.inset(48.2,16.2,10.8,11.8); ax.hist(S,bins=45,density=True,color='#9DB9D8',ec='none'); xx=np.linspace(S.min(),S.max(),200); ax.plot(xx,np.exp(-0.5*((xx-mu)/sd)**2)/(sd*np.sqrt(2*np.pi)),color='#B85450',lw=0.9)
ax.axvline(mu-2*sd,color='#555',lw=0.5,ls='--'); ax.axvline(mu+2*sd,color='#555',lw=0.5,ls='--'); ax.set_yticks([]); ax.set_title('$S_a$ over frames',fontsize=5,pad=1)
ax=D.inset(61.2,16.2,7.4,11.8); ax.plot(g,mug,color='#D79B00',lw=1); ax.axhline(mu,color='#555',lw=0.5,ls='--'); ax.set_xticks([0,1]); ax.set_yticks([]); ax.set_title('$\\mu_a(\\rho)$ rises',fontsize=5,pad=1)
D.text(58.2,13.0,'exact for any parity-check matrix\nchecks without shared bits add 0',fs=4.7)
D.arrow(43.6,22,47.9,22); D.arrow(58.2,28.4,58.2,30.9)
# combining group
D.box(73,31.2,10.3,5.8,'$\\mathbf{z}_r=\\mathbf{y}_1+r\\,\\mathbf{y}_2$\n$r$ on a grid','dec',5.4)
ax=D.inset(85.6,28.6,11.5,8.8); ax.plot(RG,Sr,color='#6C8EBF',lw=1); k=int(np.argmax(Sr)); ax.plot(RG[k],Sr[k],'o',color='#B85450',ms=2.8); ax.axvline(ro,color='#2E7D32',lw=0.6,ls=':'); ax.set_xscale('log'); ax.set_yticks([]); ax.set_title('$S_a(r)$: peak gives $\\hat r$',fontsize=5,pad=1)
D.box(73,20.2,10.3,6.2,'$\\mathbf{L}=\\mathbf{y}_1+\\hat r\\,\\mathbf{y}_2$\nmin-sum decoder','dec',5.2)
ax=D.inset(87.0,13.0,10.2,11.5)
ax.semilogy(gaps,[r['fer']['egc'] for r in F],'--',color='#999',lw=0.9,label='equal'); ax.semilogy(gaps,[r['fer']['G31'] for r in F],color='#B85450',lw=1.1,label='blind'); ax.semilogy(gaps,[r['fer']['oracle'] for r in F],':',color='#2E7D32',lw=1.0,label='oracle')
ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_yticks([0.05,0.5]); ax.set_yticklabels(['.05','.5']); ax.set_ylim(0.025,1.2)
ax.set_title('FER vs SNR gap (dB)',fontsize=5,pad=1); ax.legend(fontsize=4.0,frameon=False,loc='lower right',handlelength=1.2,borderpad=0.1,labelspacing=0.2,bbox_to_anchor=(1.02,0.08)); ax.tick_params(which='both',labelsize=4)
D.text(78.2,15.5,'blind beats equal\nweights above 2.3 dB',fs=5.0,color='#B85450',bold=True)
D.arrow(78.2,31.1,78.2,26.5); D.arrow(83.4,34.1,85.5,34.1); D.arrow(83.4,23.0,86.9,23.0)
D.arrow(69.6,34.0,72.9,34.0)
D.line([8.3,8.3],[10.5,8.2],dashed=True,color='#444',lw=0.8); D.line([8.3,77.6],[8.2,8.2],dashed=True,color='#444',lw=0.8); D.arrow(77.6,8.2,77.6,10.4,dashed=True,color='#444',lw=0.8)
D.text(43,9.2,'the two frames also enter the weighted sum $\\mathbf{z}_r$',fs=4.9,color='#444')
D.legend([('in','Received data'),('stat','Statistic'),('theory','Exact moments (proved)'),('dec','Combining')],y=5.0,x0=3,dx=24)
D.text(50,1.5,'The hard syndrome estimator is the limit $a\\to\\infty$ of this statistic. Finite $a$ keeps improving at high SNR, where the hard one fails.',fs=5.4,style='italic')
D.save('../fig_overview.pdf','/tmp/l4v2_ov.png')
