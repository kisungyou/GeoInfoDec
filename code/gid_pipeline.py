"""Unpenalized GID fitting and design-aware residual inference.

All densities are relative to normalized surface measure. The quadrature is
positive and shared by nested models; refinement is a diagnostic, not a proof
of a uniform integration-error bound. No ridge, moment shrinkage, or negative
gap clipping is used. Covariances use the empirical moment as their center.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import chi2, qmc
from scipy.optimize import linprog


def circle_features(theta, L=2):
    theta=np.asarray(theta)
    if theta.ndim!=1 or not np.all(np.isfinite(theta)):
        raise ValueError('Angles must be a finite vector')
    return np.column_stack([f(k*theta) for k in range(1,L+1) for f in (np.cos,np.sin)])


def circle_grid(M=512,L=2):
    theta=np.arange(M)*(2*np.pi/M)
    return theta,np.full(M,1/M),circle_features(theta,L)


def sphere_grid(nz=24,nphi=48):
    z,w=np.polynomial.legendre.leggauss(nz)
    p=np.arange(nphi)*(2*np.pi/nphi)
    zz,pp=np.meshgrid(z,p,indexing='ij')
    rr=np.sqrt(1-zz.ravel()**2)
    X=np.column_stack([rr*np.cos(pp.ravel()),rr*np.sin(pp.ravel()),zz.ravel()])
    return X,np.repeat(w/(2*nphi),nphi)


def sphere_features(X):
    """Linear and Frobenius-orthonormal traceless quadratic features."""
    X=np.asarray(X,float)
    if X.ndim!=2 or X.shape[1]!=3 or not np.all(np.isfinite(X)) or np.max(abs(np.linalg.norm(X,axis=1)-1))>1e-7:
        raise ValueError('Sphere observations must be finite unit vectors in three dimensions')
    x,y,z=X.T
    return np.column_stack([x,y,z,(x*x-y*y)/np.sqrt(2),
       (x*x+y*y-2*z*z)/np.sqrt(6),np.sqrt(2)*x*y,np.sqrt(2)*x*z,np.sqrt(2)*y*z])


def partition(F,qw,lam):
    z=F@lam+np.log(qw)
    psi=logsumexp(z)
    p=np.exp(z-psi)
    mean=p@F
    centered=F-mean
    cov=(centered.T*p)@centered
    return float(psi),mean,(cov+cov.T)/2,p


def fit_dual(Fgrid,qweights,m,init=None,tol=2e-10,maxiter=80):
    """Damped Newton optimization of lambda @ m - log Z, without a penalty."""
    start=time.perf_counter()
    F=np.asarray(Fgrid,float); qw=np.asarray(qweights,float); m=np.asarray(m,float)
    if F.ndim!=2 or F.shape[1]!=m.size or len(qw)!=len(F):
        raise ValueError('Incompatible grid/moment dimensions')
    if np.any(qw<=0) or not np.isclose(qw.sum(),1,atol=1e-12):
        raise ValueError('Quadrature weights must be positive and normalized')
    if not np.all(np.isfinite(m)):
        raise ValueError('Nonfinite moment')
    lam=np.zeros(len(m)) if init is None else np.array(init,float).copy()
    status='iteration_limit'; iterations=0
    for iterations in range(maxiter+1):
        psi,mean,J,p=partition(F,qw,lam)
        g=mean-m
        eig=np.linalg.eigvalsh(J)
        if eig[0]<=1e-12 or np.linalg.norm(lam)>1000:
            status='boundary_or_ill_conditioned'; break
        if np.linalg.norm(g,np.inf)<=tol:
            status='ok'; break
        if iterations==maxiter: break
        step=np.linalg.solve(J,g)
        loss=psi-lam@m
        directional=g@step
        rate=1.
        for back in range(40):
            candidate=lam-rate*step
            zz=F@candidate+np.log(qw)
            newloss=logsumexp(zz)-candidate@m
            # Roundoff-aware Armijo: still require objective not to increase
            # beyond numerical precision when the predicted decrease is tiny.
            if newloss<=loss-1e-4*rate*directional+2e-15:
                lam=candidate; break
            rate*=.5
        else:
            status='line_search_failure'; break
    psi,mean,J,p=partition(F,qw,lam)
    eig=np.linalg.eigvalsh(J)
    err=float(np.linalg.norm(m-mean,np.inf))
    dual=float(lam@m-psi); entropy=float(lam@mean-psi)
    success=status=='ok' and err<=tol and eig[0]>1e-12
    return dict(lambda_=lam,dual=dual,psi=psi,mean=mean,cov=J,prob=p,
        success=success,status=status,moment_error=err,condition=float(eig[-1]/eig[0]) if eig[0]>0 else float('inf'),
        min_curvature=float(eig[0]),iterations=iterations,entropy=entropy,
        dual_entropy_discrepancy=dual-entropy,identity_d3_error=float((dual-entropy)-lam@(m-mean)),
        elapsed=time.perf_counter()-start)


def grid_feasibility(Fgrid,m):
    """Diagnostic LP; called explicitly, not used to alter the target."""
    F=np.asarray(Fgrid); m=np.asarray(m)
    A=np.vstack([np.ones(len(F)),F.T]); b=np.r_[1,m]
    out=linprog(np.zeros(len(F)),A_eq=A,b_eq=b,bounds=(0,None),method='highs')
    return dict(feasible=bool(out.success),status=int(out.status),message=out.message)


_QMC={}
def reference_sf(stat,eigs,power=14):
    """Tail of sum eig_j chi2_1,j.

    For q=2, an exact angular representation is evaluated by quadrature.

    For q>2 use a fixed scrambled Sobol Gaussian rule with 2**power points.
    It is a reproducible numerical reference approximation, not a fitted-data
    bootstrap. Eigenvalues below -1e-10 cause an explicit error; values
    at most 1e-12 are treated as numerical zero and discarded.
    """
    e=np.asarray(eigs,float)
    if not np.isfinite(stat) or e.ndim!=1 or not len(e) or not np.all(np.isfinite(e)):
        return float('nan')
    if np.min(e)<-1e-10: raise ValueError('Negative reference eigenvalue')
    e=e[e>1e-12]
    if not len(e): return float('nan')
    if stat<=0: return 1.
    if len(e)==1: return float(chi2.sf(stat/e[0],1))
    if len(e)==2:
        theta=(np.arange(256)+.5)*(2*np.pi/256)
        v=e[0]*np.cos(theta)**2+e[1]*np.sin(theta)**2
        return float(np.mean(np.exp(-stat/(2*v))))
    key=(len(e),power)
    if key not in _QMC:
        u=qmc.Sobol(len(e),scramble=True,seed=19417).random_base2(power)
        _QMC[key]=ndtri(u)**2
    samples=np.sum(_QMC[key]*e,axis=1)
    return float((1+np.count_nonzero(samples>=stat))/(len(samples)+1))


def _failed_analysis(F,w,m,low,full,split,design,reason,infer):
    """Retain fit diagnostics while marking every inferential output unavailable."""
    q=F.shape[1]-split; n=len(F); ess=1/float(w@w); nan=float('nan')
    a2=float(n) if design=='importance' else ess
    gap=full['dual']-(low['dual'] if split else 0.)
    out=dict(reduced=low,full=full,gap=gap,I1=low['dual'] if split else 0.,
        residual=np.full(q,nan),score=nan,wald=nan,a2=a2,ess=ess,n=n,
        S=np.full((q,q),nan),N=np.full((q,F.shape[1]),nan),Omega=np.full((q,q),nan),
        Sigma=np.full((F.shape[1],F.shape[1]),nan),spectrum=np.full(q,nan),se=nan,
        valid=False,status=reason,wald_valid=False,wald_status='unavailable_rank_or_fit',
        design=design,moment=m,
        diagnostics=dict(adjacent_kl=nan,gap_kl_error=nan,weighted_logscore=nan,
            gap_score_error=nan,lr_identity_error=nan,score_discrepancy=nan,
            max_weight=float(w.max()),covariance_min_eigenvalue=nan))
    if infer: out.update({k:nan for k in ('p_gap','p_score','p_wald','naive_p','kish_p')})
    return out


def analyze(F,w,Fgrid,qweights,split=3,design='importance',infer=True,reference_power=14,tol=2e-10):
    F=np.asarray(F,float); w=np.asarray(w,float)
    if F.ndim!=2 or not np.all(np.isfinite(F)) or len(F)!=len(w):
        raise ValueError('Observation features must be finite and compatible with weights')
    if np.any(w<0) or not np.all(np.isfinite(w)) or w.sum()<=0:
        raise ValueError('Weights must be finite, nonnegative, with positive total')
    if design not in ('iid','deterministic','importance'):
        raise ValueError('Unsupported design; compressed frequencies require expanded_counts_moments')
    w=w/w.sum(); n=len(F); q=F.shape[1]-split
    if q<1: raise ValueError('At least one added feature required')
    m=w@F
    low=fit_dual(Fgrid[:,:split],qweights,m[:split],tol=tol) if split else None
    init=np.r_[low['lambda_'],np.zeros(q)] if split else np.zeros(q)
    full=fit_dual(Fgrid,qweights,m,init=init,tol=tol)
    if not full['success'] or (split and not low['success']):
        reason='unavailable_fit:'+full['status']+('/'+low['status'] if split else '')
        return _failed_analysis(F,w,m,low,full,split,design,reason,infer)
    if split:
        psi,mu0,J,p0=partition(Fgrid,qweights,init)
        B=np.linalg.solve(J[:split,:split],J[:split,split:]).T
        N=np.column_stack([-B,np.eye(q)])
        S=J[split:,split:]-B@J[:split,split:]
        r=m[split:]-mu0[split:]
    else:
        psi,mu0,J,p0=partition(Fgrid,qweights,init); N=np.eye(q); S=J; r=m-mu0
    gap=full['dual']-(low['dual'] if split else 0.)
    llratio=F@(full['lambda_']-init)-full['psi']+psi
    adjacent_kl=float(full['prob']@(Fgrid@(full['lambda_']-init)-full['psi']+psi))
    s2=float(w@w); ess=1/s2
    centered=F-m
    if design in ('iid','deterministic'):
        a2=ess
        Sigma=(centered.T*(w*w/s2))@centered
    elif design=='importance':
        a2=float(n)
        Sigma=(centered.T*(n*w*w))@centered
    elif design=='frequency':
        raise ValueError('Use expanded_counts_moments for compressed frequencies')
    else: raise ValueError('Unknown design')
    Omega=N@Sigma@N.T; Omega=(Omega+Omega.T)/2
    ev,U=np.linalg.eigh(S)
    if ev[0]<=0:
        return _failed_analysis(F,w,m,low,full,split,design,'unavailable_model_curvature',infer)
    invroot=(U/np.sqrt(ev))@U.T
    spectrum=np.linalg.eigvalsh(invroot@Omega@invroot)
    if spectrum.min() < -1e-10 or not np.all(np.isfinite(spectrum)):
        return _failed_analysis(F,w,m,low,full,split,design,'unavailable_sampling_covariance',infer)
    score=float(a2*r@np.linalg.solve(S,r))
    oe=np.linalg.eigvalsh(Omega)
    valid=bool(full['success'] and (not split or low['success']) and gap>=0 and spectrum.max()>1e-12)
    wald=float(a2*r@np.linalg.solve(Omega,r)) if oe[0]>1e-12 else float('nan')
    gamma=full['lambda_']-init
    se=float(np.sqrt(max(0,gamma@Sigma@gamma)/a2))
    out=dict(reduced=low,full=full,gap=gap,I1=low['dual'] if split else 0.,residual=r,
        score=score,wald=wald,a2=a2,ess=ess,n=n,S=S,N=N,Omega=Omega,Sigma=Sigma,
        spectrum=spectrum,se=se,valid=valid,status='ok' if valid else 'unavailable_gap_or_zero_covariance',wald_valid=bool(valid and oe[0]>1e-12),
        wald_status='ok' if valid and oe[0]>1e-12 else 'unavailable_rank_or_fit',design=design,moment=m,
        diagnostics=dict(adjacent_kl=adjacent_kl,gap_kl_error=gap-adjacent_kl,
          weighted_logscore=float(w@llratio),gap_score_error=gap-float(w@llratio),
          lr_identity_error=float(2*n*gap-2*llratio.sum()) if np.ptp(w)<1e-15 else None,
          score_discrepancy=float(a2*gap-score/2),max_weight=float(w.max()),
          covariance_min_eigenvalue=float(oe[0])))
    if infer:
        out.update(p_gap=reference_sf(2*a2*gap,spectrum,reference_power) if valid else float('nan'),
          p_score=reference_sf(score,spectrum,reference_power) if valid else float('nan'),
          p_wald=float(chi2.sf(wald,q)) if valid else float('nan'),
          naive_p=float(chi2.sf(2*n*gap,q)) if valid else float('nan'),
          kish_p=float(chi2.sf(2*ess*gap,q)) if valid else float('nan'))
    return out


def expanded_counts_moments(F,counts):
    counts=np.asarray(counts)
    if np.any(counts<0) or np.any(counts!=np.floor(counts)) or counts.sum()<1:
        raise ValueError('Counts must be nonnegative integers with positive total')
    N=int(counts.sum()); m=counts@F/N; C=F-m
    return m,(C.T*(counts/N))@C,N


def sample_vmf(n,kappa,rng,mu=None):
    if kappa<1e-10: z=rng.uniform(-1,1,n)
    else: z=1+np.log(rng.uniform(np.exp(-2*kappa),1,n))/kappa
    p=rng.uniform(0,2*np.pi,n); r=np.sqrt(1-z*z)
    X=np.column_stack([r*np.cos(p),r*np.sin(p),z])
    if mu is None: return X
    mu=np.asarray(mu,float); mu=mu/np.linalg.norm(mu)
    # Orthogonal frame with mu as third column, including the poles.
    helper=np.array([1.,0,0]) if abs(mu[0])<.8 else np.array([0.,1,0])
    a=np.cross(mu,helper); a/=np.linalg.norm(a); b=np.cross(mu,a)
    return X@np.column_stack([a,b,mu]).T


def holm_adjust(pvalues):
    """Holm adjusted p-values for a fixed family; no independence assumption."""
    p=np.asarray(pvalues,float)
    if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('All planned p-values must be valid; do not drop failed tests')
    order=np.argsort(p);m=len(p);adj=np.minimum(1,np.maximum.accumulate(p[order]*(m-np.arange(m))))
    ans=np.empty(m);ans[order]=adj
    return ans


def audit_refinement(base, refined=None, scale=None, tolerance=1e-3):
    """Gate inference on a checked numerical refinement, retaining diagnostics.

    The operational tolerance was added during the final implementation audit.
    It was not selected by tuning statistical rejection/coverage results and is
    not a uniform integration-error certificate. Unchecked fits are explicitly
    marked not_checked; they are not relabeled as having passed refinement.
    """
    if not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('Refinement tolerance must be finite and positive')
    base['refinement_status']='not_checked'
    base['refinement_tolerance']=float(tolerance)
    base['refined_full_status']='not_checked'
    base['refined_reduced_status']='not_checked'
    if refined is None:
        return base
    scale=base['a2'] if scale is None else float(scale)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Refinement scale must be finite and positive')
    full=refined.get('full');low=refined.get('reduced');expects_low=base.get('reduced') is not None
    base['refined_full_status']=full.get('status','missing') if full is not None else 'missing'
    base['refined_reduced_status']=low.get('status','missing') if low is not None else ('missing' if expects_low else 'not_applicable')
    def finite_success(fit):
        return fit is not None and bool(fit.get('success',False)) and all(
            np.all(np.isfinite(np.asarray(fit.get(key,np.nan))))
            for key in ('dual','psi','lambda_','mean','cov','moment_error'))
    finite_gaps=np.isfinite(base.get('gap',np.nan)) and np.isfinite(refined.get('gap',np.nan))
    difference=scale*abs(base['gap']-refined['gap']) if finite_gaps else float('nan')
    base['diagnostics']['scaled_refinement']=float(difference)
    if not finite_success(full) or (expects_low and not finite_success(low)) or not np.isfinite(difference):
        status='unavailable_refined_fit'
    elif difference>tolerance:
        status='excessive_scaled_difference'
    else:
        status='ok'
    base['refinement_status']=status
    if status!='ok':
        base['valid']=False
        base['status']='unavailable_refinement:'+status
        base['wald_valid']=False
        base['wald_status']='unavailable_refinement'
        base['se']=float('nan')
        for key in ('p_gap','p_score','p_wald','naive_p','kish_p','analytic_p'):
            if key in base:base[key]=float('nan')
    return base
