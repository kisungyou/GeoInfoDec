"""Studies A/B with unchanged paper protocol and a five-replicate smoke mode.
GID_MODE=paper selects full replication; the default smoke mode checks execution.
See run_config.py and each generated config JSON for the selected run.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import csv,json,time,sys
import numpy as np
from scipy.special import i0,logsumexp
from scipy.stats import chi2,ncx2
from gid_pipeline import *

from run_config import ROOT,OUT,MODE,PLOTS,repetitions
SEED=2026091601


def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {k:serial(v) for k,v in x.items()}
    if isinstance(x,list): return [serial(y) for y in x]
    return x


def dump(path,x):
    path.write_text(json.dumps(serial(x),indent=2,allow_nan=True)+'\n')


def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as h:
        wr=csv.DictWriter(h,fieldnames=keys); wr.writeheader();wr.writerows(rows)


def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    c=(p+z*z/(2*n))/den; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h


def cos4_sample(n,a,rng):
    arr=[]; total=0
    while total<n:
        x=rng.uniform(0,2*np.pi,2*(n-total)+20)
        take=x[rng.random(len(x))<(1+a*np.cos(4*x))/(1+abs(a))]
        arr.append(take); total+=len(take)
    return np.concatenate(arr)[:n]


def circle_tilt_sample(n,beta,rng,kappa=2):
    arr=[]; total=0
    while total<n:
        x=rng.vonmises(0,kappa,2*(n-total)+20)
        take=x[rng.random(len(x))<np.exp(beta*np.cos(2*x)-abs(beta))]
        arr.append(take);total+=len(take)
    return np.concatenate(arr)[:n]


def record_fit(a):
    f=a['full']; l=a['reduced']; d=a['diagnostics']
    return dict(valid=a['valid'],status=a.get('status',f['status'] if f['status']!='ok' else l['status']),
       gap=a['gap'],I1=a['I1'],se=a['se'],score=a['score'],wald=a['wald'],
       p_gap=a.get('p_gap'),p_score=a.get('p_score'),p_wald=a.get('p_wald'),
       naive_p=a.get('naive_p'),kish_p=a.get('kish_p'),ess=a['ess'],a2=a['a2'],
       max_weight=d['max_weight'],moment_error=max(f['moment_error'],l['moment_error']),
       condition=max(f['condition'],l['condition']),iterations=f['iterations']+l['iterations'],
       gap_kl_error=d['gap_kl_error'],gap_score_error=d['gap_score_error'],
       lr_identity_error=d['lr_identity_error'],score_discrepancy=d['score_discrepancy'],
       eig_min=float(a['spectrum'].min()),eig_max=float(a['spectrum'].max()),
       min_sampling_eigenvalue=d['covariance_min_eigenvalue'],
       dual_entropy_discrepancy=max(abs(f['dual_entropy_discrepancy']),abs(l['dual_entropy_discrepancy'])),
       fit_seconds=f['elapsed']+l['elapsed'],
       refinement_status=a.get('refinement_status','not_checked'),
       refinement_tolerance=a.get('refinement_tolerance',1e-3),
       refined_full_status=a.get('refined_full_status','not_checked'),
       refined_reduced_status=a.get('refined_reduced_status','not_checked'))


def study_a():
    out=OUT/'study_a';out.mkdir(exist_ok=True)
    cells=[]
    for n in (200,800):
        for law in ('uniform_direct','vm2_direct','cos4_direct','uniform_importance','vm2_importance'):
            cells.append(dict(domain='circle',law=law,n=n,repeats=repetitions(1000)))
    for law in ('uniform_direct','vm8_direct','uniform_importance'):
        cells.append(dict(domain='sphere',law=law,n=400,repeats=repetitions(400)))
    config=dict(mode=MODE,seed=SEED,alpha=.05,cells=cells,
       feature_definitions='gid_pipeline.py; raw circle cos/sin in degree order; S2 linear plus Frobenius-orthonormal traceless quadratics',
       moment_null='I2=0',circle_target='uniform; vM(mu=0,kappa=2); (1+0.8cos(4theta)) relative to dtheta/(2pi)',
       circle_proposals='uniform target: vM(0,1.2); vM(0,2) target: vM(0.5,1.2)',
       circle_ratios='uniform: I0(1.2)*exp(-1.2cos(theta)); vm2: I0(1.2)/I0(2)*exp(2cos(theta)-1.2cos(theta-0.5))',
       sphere_proposal='vMF(north,1.2); uniform ratio sinh(1.2)/(1.2)*exp(-1.2z)',
       grids='circle 512 -> 1024; S2 Gauss-Legendre z times uniform azimuth 24x48 -> 48x96',
       optimizer='unpenalized damped Newton; infinity moment tolerance 2e-10; max80; no ridge or shrinkage',
       reference='q2:256 midpoint angles, q5:(1+exceedances)/(2^14+1), scrambled Sobol Gaussian squares seed19417; 2^16 check on refinement draws',
       refinement='up to first 25 available replicates per cell; rotate up to first 10; retain every failure',
       refinement_gate=dict(tolerance=1e-3,scale='n',criterion='n*abs(coarse gap-refined gap) <= tolerance; both refined fits successful and finite',history='Added during final implementation audit, not used to tune statistical results; an operational diagnostic, not a uniform numerical error certificate'),
       reported_statistic='2*a2*(optimized full empirical dual - optimized reduced empirical dual)',
       covariance='empirical-centered design-specific sandwich; iid scale n, importance scale n',
       saved_draws='per-cell NPZ contains observations and exact realized unnormalized weights; per-replicate seed is deterministic from cell index and replicate')
    dump(out/'config.json',config)
    _,cw,CF=circle_grid();_,cw2,CF2=circle_grid(1024)
    sx,sw=sphere_grid(); SF=sphere_features(sx);sx2,sw2=sphere_grid(48,96);SF2=sphere_features(sx2)
    allrows=[];summary=[];start=time.perf_counter()
    for ci,c in enumerate(cells):
        dom,law,n,B=c['domain'],c['law'],c['n'],c['repeats']
        cell=f'{dom}_{law}_n{n}'
        rawx=[];raww=[];rows=[]
        for rep in range(B):
            seed=SEED+ci*100000+rep; rng=np.random.default_rng(seed)
            w=np.ones(n)
            if dom=='circle':
                if law=='uniform_direct': x=rng.uniform(0,2*np.pi,n)
                elif law=='vm2_direct': x=rng.vonmises(0,2,n)
                elif law=='cos4_direct':x=cos4_sample(n,.8,rng)
                elif law=='uniform_importance':
                    x=rng.vonmises(0,1.2,n);w=i0(1.2)*np.exp(-1.2*np.cos(x))
                else:
                    x=rng.vonmises(.5,1.2,n);w=i0(1.2)/i0(2)*np.exp(2*np.cos(x)-1.2*np.cos(x-.5))
                F=circle_features(x);FG,qw,FG2,qw2,split=CF,cw,CF2,cw2,2
            else:
                k=8 if law=='vm8_direct' else 1.2 if law=='uniform_importance' else 0
                x=sample_vmf(n,k,rng)
                if law=='uniform_importance':w=(np.sinh(1.2)/1.2)*np.exp(-1.2*x[:,2])
                F=sphere_features(x);FG,qw,FG2,qw2,split=SF,sw,SF2,sw2,3
            design='importance' if 'importance' in law else 'iid'
            a=audit_refinement(analyze(F,w,FG,qw,split=split,design=design))
            row=dict(cell=cell,replicate=rep,seed=seed,**record_fit(a))
            if rep<25:
                ar=analyze(F,w,FG2,qw2,split=split,design=design,reference_power=16,tol=2e-12)
                a=audit_refinement(a,ar,scale=n);row.update(record_fit(a))
                row['scaled_refinement']=a['diagnostics']['scaled_refinement']
                row['reference_refinement']=abs(a['p_gap']-reference_sf(2*n*a['gap'],a['spectrum'],16))
                if rep<10:
                    if dom=='circle':Fr=circle_features(x+.381)
                    else:
                        U,_=np.linalg.qr(rng.normal(size=(3,3)));Fr=sphere_features(x@U)
                    rot=analyze(Fr,w,FG2,qw2,split=split,design=design,infer=False,tol=2e-12)
                    row['scaled_rotation']=n*abs(ar['gap']-rot['gap'])
            if law=='cos4_direct':row['analytic_p']=reference_sf(2*n*a['gap'],[1.4,.6]) if a['valid'] else float('nan')
            rows.append(row);rawx.append(x);raww.append(w)
        np.savez_compressed(out/f'{cell}_draws.npz',observations=np.array(rawx),raw_weights=np.array(raww))
        csvwrite(out/f'{cell}_replicates.csv',rows)
        for method in ('p_gap','p_score','p_wald','naive_p','kish_p','analytic_p'):
            pp=np.array([r.get(method,np.nan) for r in rows],float);valid=np.isfinite(pp)
            if valid.sum()==0:continue
            k=int(np.sum(pp<.05));lo,hi=wilson(k,B)
            summary.append(dict(cell=cell,domain=dom,law=law,n=n,repeats=B,method=method,
                rejections=k,rejection_rate=k/B,mc_se=float(np.sqrt(k/B*(1-k/B)/B)),
                wilson_low=lo,wilson_high=hi,failures=int(B-valid.sum()),
                cdf_01=float(np.sum(pp<.01)/B),cdf_10=float(np.sum(pp<.1)/B),
                cdf_50=float(np.sum(pp<.5)/B),cdf_90=float(np.sum(pp<.9)/B)))
        allrows+=rows
        print(cell,'seconds',round(time.perf_counter()-start,1),'failures',sum(not r['valid'] for r in rows),flush=True)
        csvwrite(out/'rejection_summary.csv',summary)
    dump(out/'numerical_manifest.json',dict(mode=MODE,total_replicates=len(allrows),failed=[r for r in allrows if not r['valid']],
       refinement_status_counts={s:sum(r['refinement_status']==s for r in allrows) for s in ('not_checked','ok','unavailable_refined_fit','excessive_scaled_difference')},
       max_moment_error=max(r['moment_error'] for r in allrows),max_condition=max(r['condition'] for r in allrows),
       max_abs_kl_identity=max(abs(r['gap_kl_error']) for r in allrows),
       max_scaled_refinement=max(r.get('scaled_refinement',0) for r in allrows),
       max_scaled_rotation=max(r.get('scaled_rotation',0) for r in allrows),
       max_reference_refinement=max(r.get('reference_refinement',0) for r in allrows),
       fit_seconds=sum(r['fit_seconds'] for r in allrows),total_seconds=time.perf_counter()-start,
       disclaimer='Observed refinement and rotation differences are diagnostics, not certified bounds on continuous-integration error.'))
    discrepancy=[]
    for c in sorted(set(r['cell'] for r in allrows)):
        rr=[r for r in allrows if r['cell']==c];v=np.array([r['score_discrepancy'] for r in rr])
        discrepancy.append(dict(cell=c,mean=float(v.mean()),median=float(np.median(v)),
            q05=float(np.quantile(v,.05)),q95=float(np.quantile(v,.95)),max_abs=float(abs(v).max()),
            median_ess=float(np.median([r['ess'] for r in rr])),max_condition=max(r['condition'] for r in rr)))
    csvwrite(out/'surrogate_discrepancy.csv',discrepancy)
    if PLOTS: plot_a(out,summary,allrows)


def plot_a(out,summary,rows):
    from figure_style import plt,save_figure,COLORS,TEXT_WIDTH_IN
    fig,axes=plt.subplots(1,3,figsize=(TEXT_WIDTH_IN,2.65),sharex=True,sharey=True)
    fig.subplots_adjust(left=.095,right=.970,bottom=.29,top=.845,wspace=.20)
    for ax,cell,title in zip(axes,['circle_uniform_importance_n800','circle_cos4_direct_n800','sphere_vm8_direct_n400'],['Informative weights','Fourth-harmonic null','Spherical vMF null']):
        rr=[r for r in rows if r['cell']==cell]
        for key,label,color,style in [('p_gap','Fitted gap',COLORS['blue'],'-'),('p_wald','Residual Wald',COLORS['orange'],'--'),('naive_p','Naive chi-square',COLORS['purple'],'-.')]:
            v=np.sort([float(r[key]) for r in rr]);ax.plot(v,np.arange(1,len(v)+1)/len(v),label=label,color=color,ls=style)
        ax.plot([0,1],[0,1],color=COLORS['gray'],ls=':',lw=.9,zorder=0)
        ax.set(xlabel='p-value',title=title,xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1])
    axes[0].set_ylabel('Empirical CDF')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.53,.015))
    meta=save_figure(fig,out/'calibration_cdfs');plt.close(fig)
    return meta


def population_circle(beta,kappa=2,M=16384):
    t,w,F=circle_grid(M);lam=np.array([kappa,0,beta,0]);p=np.exp(F@lam-logsumexp(F@lam));m=p@F
    lo=fit_dual(F[:,:2],w,m[:2],tol=2e-12);fu=fit_dual(F,w,m,init=lam,tol=2e-12)
    _,_,J,_=partition(F,w,np.r_[lo['lambda_'],0,0]);S=J[2:,2:]-J[2:,:2]@np.linalg.solve(J[:2,:2],J[:2,2:])
    return dict(gap=fu['dual']-lo['dual'],I1=lo['dual'],moment=m,S=S,lambda_=lam)


def study_b():
    out=OUT/'study_b';out.mkdir(exist_ok=True)
    B=repetitions(600)
    config=dict(mode=MODE,seed=SEED+9000000,repeats=B,n=[200,800],fixed_beta=[.3,.6],local_b=[0,1,2,3,4],
       target='circle exp(2cos(theta)+beta cos(2theta))/Z; exact rejection samples from vM(0,2)',
       local_sequence='beta=b/sqrt(n); chi2_2 noncentrality b²S_11 at vM(0,2)',
       intervals='gap +/- 1.95996*sqrt(gamma.T Sigma gamma/n), empirical-centered iid covariance; no clipping',
       population='independent 16384-point circle quadrature, verified at32768; sphere48x96 verified96x192',
       refinement_gate=dict(tolerance=1e-3,scale='n',checked_rows='up to first 10 available replicates per cell',criterion='n*abs(coarse gap-refined gap) <= tolerance; both refined fits successful and finite',history='Added during final implementation audit, not used to tune statistical results; an operational diagnostic, not a uniform numerical error certificate'),
       geometry='population vMF(k8), equally weighted antipodal vMF(k8), exp(-12 z²) girdle, uniform; exact cos4 a=.8 hierarchy')
    dump(out/'config.json',config)
    t,cw,CF=circle_grid();t2,cw2,CF2=circle_grid(1024)
    base=population_circle(0);rows=[];summaries=[];pc=[];start=time.perf_counter()
    specs=[('fixed',b,n) for n in (200,800) for b in (.3,.6)]+[('local',b,n) for n in (200,800) for b in (0,1,2,3,4)]
    for ci,(regime,b,n) in enumerate(specs):
        beta=b if regime=='fixed' else b/np.sqrt(n);pop=population_circle(beta);popfine=population_circle(beta,M=32768)
        cell=f'{regime}_b{b}_n{n}';rr=[];xx=[]
        for rep in range(B):
            seed=SEED+9000000+ci*10000+rep;rng=np.random.default_rng(seed)
            x=circle_tilt_sample(n,beta,rng);F=circle_features(x)
            a=audit_refinement(analyze(F,np.ones(n),CF,cw,split=2,design='iid'));row=dict(cell=cell,replicate=rep,seed=seed,regime=regime,b=b,beta=beta,n=n,population_gap=pop['gap'],**record_fit(a))
            if rep<10:
                ar=analyze(F,np.ones(n),CF2,cw2,split=2,design='iid',tol=2e-12)
                a=audit_refinement(a,ar,scale=n);row.update(record_fit(a))
                row['scaled_refinement']=a['diagnostics']['scaled_refinement']
            row.update(lower=a['gap']-1.959963984540054*a['se'],upper=a['gap']+1.959963984540054*a['se'],error=a['gap']-pop['gap'])
            row['covered']=(row['lower']<=pop['gap']<=row['upper']) if a['valid'] else None
            rr.append(row);xx.append(x)
        csvwrite(out/f'{cell}_replicates.csv',rr);np.savez_compressed(out/f'{cell}_draws.npz',theta=np.array(xx))
        errs=np.array([r['error'] for r in rr]);cov=int(sum(r['covered'] is True for r in rr));lo,hi=wilson(cov,B)
        sr=dict(cell=cell,regime=regime,b=b,beta=beta,n=n,repeats=B,population_gap=pop['gap'],
          population_refinement=abs(popfine['gap']-pop['gap']),bias=float(errs.mean()),rmse=float(np.sqrt(np.mean(errs**2))),
          coverage=cov/B,coverage_mc_se=float(np.sqrt(cov/B*(1-cov/B)/B)),coverage_lo=lo,coverage_hi=hi,
          interval_claim='fixed alternative only' if regime=='fixed' else 'near-null interval reported diagnostically, not justified by fixed-alternative theorem',
          failures=sum(not r['valid'] for r in rr),
          asymptotic_local_power=float(ncx2.sf(chi2.ppf(.95,2),2,b*b*base['S'][0,0])) if regime=='local' else None)
        for method in ('p_gap','p_score','p_wald','naive_p'):
            k=sum(r[method]<.05 for r in rr);pl,pu=wilson(k,B)
            sr[f'{method}_rejection']=k/B;sr[f'{method}_mc_se']=np.sqrt(k/B*(1-k/B)/B);sr[f'{method}_lo']=pl;sr[f'{method}_hi']=pu
        summaries.append(sr);rows+=rr;csvwrite(out/'power_estimation_summary.csv',summaries)
        print(cell,'seconds',round(time.perf_counter()-start,1),flush=True)
    population_geometry(out)
    dump(out/'numerical_manifest.json',dict(mode=MODE,total_replicates=len(rows),failed=[r for r in rows if not r['valid']],
       refinement_status_counts={s:sum(r['refinement_status']==s for r in rows) for s in ('not_checked','ok','unavailable_refined_fit','excessive_scaled_difference')},
       max_moment_error=max(r['moment_error'] for r in rows),max_condition=max(r['condition'] for r in rows),
       max_abs_kl_identity=max(abs(r['gap_kl_error']) for r in rows),max_scaled_refinement=max(r.get('scaled_refinement',0) for r in rows),
       total_seconds=time.perf_counter()-start))
    if PLOTS: plot_b(out,summaries)


def population_geometry(out):
    rows=[]
    for name in ('uniform','vmf8','antipodal8','girdle12'):
        vals=[]
        for nz,np_ in ((48,96),(96,192)):
            X,qw=sphere_grid(nz,np_);F=sphere_features(X);z=X[:,2]
            logp=np.zeros(len(z)) if name=='uniform' else 8*z if name=='vmf8' else np.logaddexp(8*z,-8*z) if name=='antipodal8' else -12*z*z
            wp=qw*np.exp(logp-logsumexp(np.log(qw)+logp));a=analyze(F,wp,F,qw,split=3,design='importance',infer=False,tol=2e-12)
            Q=(X.T*wp)@X;r=wp@X
            # Fitted reduced-law Q from its exact quadrature probabilities.
            Q0=(X.T*a['reduced']['prob'])@X
            vals.append(dict(law=name,resultant=float(np.linalg.norm(r)),kappa=float(np.linalg.norm(a['reduced']['lambda_'])),
              raw_anisotropy=float(np.linalg.norm(Q-np.eye(3)/3)),residual_anisotropy=float(np.linalg.norm(Q-Q0)),I1=a['I1'],I2=a['gap'],
              residual_eigenvalues=np.linalg.eigvalsh(Q-Q0).tolist(),moment_error=max(a['full']['moment_error'],a['reduced']['moment_error']),
              descriptive_only=True,quadrature=f'{nz}x{np_}'))
        vals[1]['I2_refinement']=abs(vals[1]['I2']-vals[0]['I2']);rows.append(vals[1])
    csvwrite(out/'sphere_population_geometry.csv',rows)
    t,w,F=circle_grid(16384,L=4);p=w*(1+.8*np.cos(4*t));m=p@F;ds=[0];rr=[];prev=[]
    for L in range(1,5):
        fit=fit_dual(F[:,:2*L],w,m[:2*L],init=np.r_[prev,0,0],tol=2e-12);prev=fit['lambda_'];ds.append(fit['dual'])
        rr.append(dict(level=L,deficit=ds[-1],gap=ds[-1]-ds[-2],moment_error=fit['moment_error']))
    csvwrite(out/'cos4_population_hierarchy.csv',rr)


def plot_b(out,summary):
    from figure_style import plt,save_figure,COLORS,TEXT_WIDTH_IN
    fig,axes=plt.subplots(1,2,figsize=(TEXT_WIDTH_IN,2.8))
    fig.subplots_adjust(left=.095,right=.985,bottom=.28,top=.855,wspace=.35)
    for n,col,marker,style in ((200,COLORS['blue'],'o','-'),(800,COLORS['orange'],'s','--')):
        rr=[r for r in summary if r['regime']=='local' and int(r['n'])==n]
        axes[0].plot([float(r['b']) for r in rr],[float(r['p_gap_rejection']) for r in rr],
                     marker=marker,ls=style,color=col,label=rf'$n={n}$')
    axes[0].plot([float(r['b']) for r in rr],[float(r['asymptotic_local_power']) for r in rr],
                 color=COLORS['black'],ls=':',label='Reference')
    axes[0].set(xlabel=r'Local tilt $b$',ylabel='Rejection frequency',title='Local alternatives',
                ylim=(0,1),yticks=[0,.5,1],xticks=[0,1,2,3,4])
    for n,col,off,marker in ((200,COLORS['blue'],-.015,'o'),(800,COLORS['orange'],.015,'s')):
        rr=[r for r in summary if r['regime']=='fixed' and int(r['n'])==n]
        axes[1].errorbar([float(r['beta'])+off for r in rr],[float(r['coverage']) for r in rr],
            yerr=[[float(r['coverage'])-float(r['coverage_lo']) for r in rr],[float(r['coverage_hi'])-float(r['coverage']) for r in rr]],
            fmt=marker,color=col,capsize=3,elinewidth=1)
    axes[1].axhline(.95,color=COLORS['black'],ls=':')
    axes[1].set(xlabel=r'Fixed tilt $\beta$',ylabel='Interval coverage',title='Fixed alternatives',
                ylim=(.8,1),yticks=[.8,.9,.95,1],xticks=[.3,.6],xlim=(.25,.65))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.53,.015))
    meta=save_figure(fig,out/'power_coverage');plt.close(fig)
    return meta


if __name__=='__main__':
    if len(sys.argv)==1 or sys.argv[1]=='a':study_a()
    if len(sys.argv)==1 or sys.argv[1]=='b':study_b()
