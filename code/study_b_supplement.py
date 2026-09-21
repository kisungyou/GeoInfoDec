"""Prespecified descriptive weight stress and finite-resolution diagnostic."""
from study_ab import *
from run_config import ROOT,OUT,MODE,PLOTS,repetitions

def main():
    out=OUT/'study_b';out.mkdir(parents=True,exist_ok=True)
    stress_B=repetitions(100);tail_B=repetitions(400)
    dump(out/'supplement_config.json',dict(mode=MODE,seed=819263,pareto_shape=1.5,n=400,stress_repeats=stress_B,
        stress_laws='vM(0,4); equal antipodal vM(k7); equal trimodal vM(k5.5)',
        stress_interpretation='Independent Pareto raw weights have infinite second moment. Descriptive stress only; no Gaussian covariance or p-values.',
        finite_tail_law='1+0.8cos4theta',finite_tail_n=800,finite_tail_repeats=tail_B,
        planned_tests='I1,I2,I3,I4 and D4-D1; Holm over all five, no stopping at first nonsignificance',
        numerical='circle grid1024 for stress,512 for tail; tol2e-10, unpenalized; add-one scrambled Sobol reference for tail'))
    t,w,F=circle_grid(1024,L=4);rows=[];sumrows=[]
    for li,law in enumerate(('unimodal','antipodal','trimodal')):
        xs=[];ww=[]
        for rep in range(stress_B):
            rng=np.random.default_rng(819263+li*1000+rep)
            if law=='unimodal': x=rng.vonmises(0,4,400)
            elif law=='antipodal':x=rng.vonmises(np.pi*rng.integers(0,2,400),7)
            else:x=rng.vonmises(2*np.pi/3*rng.integers(0,3,400),5.5)
            raw=1+rng.pareto(1.5,400);xs.append(x);ww.append(raw);FX=circle_features(x,L=4)
            for wn,weight in [('equal',np.ones(400)),('pareto',raw)]:
                nw=weight/weight.sum();m=nw@FX;D=[0.];prev=[];fits=[]
                for L in range(1,5):
                    f=fit_dual(F[:,:2*L],w,m[:2*L],init=np.r_[prev,0,0]);prev=f['lambda_'];D.append(f['dual']);fits.append(f)
                row=dict(law=law,weights=wn,replicate=rep,ess=1/(nw@nw),max_weight=float(nw.max()),
                    D4=D[-1],moment_error=max(f['moment_error'] for f in fits),
                    success=all(f['success'] for f in fits),condition=max(f['condition'] for f in fits))
                row.update({f'I{j+1}':v for j,v in enumerate(np.diff(D))});rows.append(row)
        np.savez_compressed(out/f'stress_{law}_draws.npz',theta=np.array(xs),pareto_weights=np.array(ww))
    csvwrite(out/'pareto_stress_replicates.csv',rows)
    for law in ('unimodal','antipodal','trimodal'):
        for wn in ('equal','pareto'):
            rr=[r for r in rows if r['law']==law and r['weights']==wn]
            row=dict(law=law,weights=wn,repeats=len(rr),failures=sum(not r['success'] for r in rr))
            for key in ('ess','max_weight','I1','I2','I3','I4','D4'):
                vv=np.array([r[key] for r in rr]);row[key+'_mean']=float(vv.mean());row[key+'_sd']=float(vv.std(ddof=1));row[key+'_q05']=float(np.quantile(vv,.05));row[key+'_q95']=float(np.quantile(vv,.95))
            sumrows.append(row)
    csvwrite(out/'pareto_stress_summary.csv',sumrows)
    t,w,F=circle_grid(512,L=4);rows=[];xx=[]
    for rep in range(tail_B):
        rng=np.random.default_rng(819263+100000+rep);x=cos4_sample(800,.8,rng);FX=circle_features(x,L=4);xx.append(x);pvals=[];row=dict(replicate=rep,seed=819263+100000+rep)
        for L in range(1,5):
            a=analyze(FX[:,:2*L],np.ones(800),F[:,:2*L],w,split=2*(L-1),design='iid')
            pvals.append(a['p_gap']);row[f'I{L}']=a['gap'];row[f'p{L}']=a['p_gap'];row[f'valid{L}']=a['valid']
        a=analyze(FX,np.ones(800),F,w,split=2,design='iid');pvals.append(a['p_gap']);row.update(tail=a['gap'],p_tail=a['p_gap'],valid_tail=a['valid'])
        h=holm_adjust(pvals)
        for key,val in zip(('I1','I2','I3','I4','tail'),h):row['holm_'+key]=val
        row['would_stop_at_first']=row['p1']>.05
        rows.append(row)
    csvwrite(out/'finite_tail_holm_replicates.csv',rows);np.savez_compressed(out/'finite_tail_draws.npz',theta=np.array(xx))
    summary=dict(mode=MODE,n=800,repeats=tail_B,failures=sum(not all(r['valid'+str(j)] for j in range(1,5)) or not r['valid_tail'] for r in rows),
        first_gap_nonrejections=sum(r['would_stop_at_first'] for r in rows),
        holm_any_true_null_rejections=sum(any(r['holm_I'+str(j)]<.05 for j in range(1,4)) for r in rows))
    for key in ('I1','I2','I3','I4','tail'):
        k=sum(r['holm_'+key]<.05 for r in rows);lo,hi=wilson(k,tail_B)
        summary['holm_'+key]=dict(rejections=k,rate=k/tail_B,wilson_low=lo,wilson_high=hi)
    dump(out/'finite_tail_holm_summary.json',summary)
    print(summary)
if __name__=='__main__':main()
