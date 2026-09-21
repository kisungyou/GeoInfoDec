"""Executable numerical identities. Failures raise an assertion and are retained."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import chi2
from gid_pipeline import *

from run_config import ROOT,OUT,MODE,PLOTS,repetitions
CHECK_OUT=OUT/'study_b'
CHECK_OUT.mkdir(parents=True,exist_ok=True)
checks=[]

def check(name,value,tolerance):
    passed=bool(np.isfinite(value) and value<=tolerance)
    checks.append(dict(check=name,error=float(value),tolerance=float(tolerance),passed=passed))
    print(name,value,'PASS' if passed else 'FAIL')

rng=np.random.default_rng(919)
t,w,F=circle_grid(2048,L=4)
for L in range(1,5):
    f=fit_dual(F[:,:2*L],w,np.zeros(2*L),tol=2e-12)
    check(f'uniform_degree{L}',max(abs(f['dual']),np.linalg.norm(f['lambda_'])),1e-10)
# vM law exactly contained in every fitted family.
lam=np.array([1.7,.3,0,0,0,0,0,0]);ps,mu,J,p=partition(F,w,lam)
ds=[]
for L in range(1,5):ds.append(fit_dual(F[:,:2*L],w,mu[:2*L],tol=2e-12)['dual'])
check('vm_later_gaps',max(abs(np.diff(ds))),1e-10)
# Fourier counterexample and its model/sampling spectra.
cp=w*(1+.8*np.cos(4*t));m=cp@F;ds=[0]
for L in range(1,5):ds.append(fit_dual(F[:,:2*L],w,m[:2*L],tol=2e-12)['dual'])
check('cos4_first_three_gaps',max(abs(np.diff(ds)[:3])),1e-10)
check('cos4_fourth_positive',0 if ds[4]-ds[3]>.1 else 1,0)
S=.5*np.eye(2);Om=(F[:,2:4].T*cp)@F[:,2:4]
check('cos4_reference_eigenvalues',np.max(abs(np.linalg.eigvalsh(2*Om)-[.6,1.4])),1e-10)
# d=2 local coefficients: I1/r² ->1; I2/||Q-I/2||F² ->2.
eps=1e-4
f=fit_dual(F[:,:2],w,np.array([eps,0]),tol=2e-12)
check('local_first_order_d2',abs(f['dual']/eps**2-1),1e-5)
f=fit_dual(F[:,:4],w,np.array([0,0,eps,0]),tol=2e-12)
check('local_mean_zero_quadratic_d2',abs(f['dual']/(eps**2/2)-2),1e-5)
# Natural coordinates exactly zero at uniformity, no arbitrary direction.
check('uniform_natural_coordinates',np.linalg.norm(fit_dual(F[:,:2],w,[0,0])['lambda_']),1e-14)
# Derivatives against central differences.
FF=F[:,:4];la=np.array([.6,-.3,.2,.1]);ps,mm,JJ,_=partition(FF,w,la);h=2e-5
fd=np.array([(partition(FF,w,la+np.eye(4)[j]*h)[0]-partition(FF,w,la-np.eye(4)[j]*h)[0])/(2*h) for j in range(4)])
fdJ=np.column_stack([(partition(FF,w,la+np.eye(4)[j]*h)[1]-partition(FF,w,la-np.eye(4)[j]*h)[1])/(2*h) for j in range(4)])
check('logpartition_gradient',np.max(abs(fd-mm)),2e-8)
check('logpartition_hessian',np.max(abs(fdJ-JJ)),2e-8)
# Basis change, with a nonsingular block preserving retained coordinates.
x=rng.vonmises(.4,2,500);FX=circle_features(x);wt=np.exp(.4*np.cos(x));a=analyze(FX,wt,FF,w,split=2)
A=np.array([[2.,.3,0,0],[-.1,.7,0,0],[.5,.2,1.2,.4],[.1,-.4,-.3,.8]])
b=analyze(FX@A.T,wt,FF@A.T,w,split=2)
check('basis_gap',abs(a['gap']-b['gap']),1e-9)
check('basis_reference_spectrum',max(abs(a['spectrum']-b['spectrum'])),1e-8)
check('nested_KL_identity',abs(a['diagnostics']['gap_kl_error']),1e-8)
check('empirical_logscore_identity',abs(a['diagnostics']['gap_score_error']),1e-10)
check('approximate_parameter_D3',abs(a['full']['identity_d3_error']),1e-12)
# Check D3 also at a deliberately incomplete fit, not only an optimum.
ps,mm,_,_=partition(FF,w,la);memp=np.mean(FX,axis=0)
check('deliberately_approximate_D3',abs((la@memp-ps)-(la@mm-ps)-la@(memp-mm)),1e-12)
b=analyze(FX,37*wt,FF,w,split=2)
check('raw_weight_rescaling',max(abs(a['gap']-b['gap']),max(abs(a['spectrum']-b['spectrum']))),1e-10)
# Count compression retains the original independent observation count.
Fc=FX[:5];counts=np.array([3,7,1,9,4]);ex=np.repeat(Fc,counts,axis=0);mc,sc,N=expanded_counts_moments(Fc,counts)
check('frequency_moment_covariance',max(np.max(abs(mc-ex.mean(axis=0))),np.max(abs(sc-(ex-ex.mean(axis=0)).T@(ex-ex.mean(axis=0))/len(ex)))),1e-12)
check('frequency_original_n',abs(N-len(ex)),0)
# Correct model spectrum exactly one with population covariance and curvature.
J=partition(FF,w,np.array([2.,0,0,0]))[2];B=J[2:,:2]@np.linalg.inv(J[:2,:2]);N=np.c_[-B,np.eye(2)];S=J[2:,2:]-B@J[:2,2:];Om=N@J@N.T
check('factor_two_correct_model_spectrum',max(abs(np.linalg.eigvalsh(np.linalg.solve(S,Om))-1)),1e-10)
check('chi_square_reference_tail',abs(reference_sf(chi2.ppf(.95,2),[1,1])-.05),1e-12)
# S2 population, rotation and mean-zero coefficient.
X,qw=sphere_grid(48,96);FG=sphere_features(X)
x=sample_vmf(500,8,rng);aa=analyze(sphere_features(x),np.ones(500),FG,qw,split=3)
U,_=np.linalg.qr(rng.normal(size=(3,3)));bb=analyze(sphere_features(x@U),np.ones(500),FG,qw,split=3)
check('sphere_rotation_gap',abs(aa['gap']-bb['gap']),1e-8)
_,mu,J,p=partition(FG,qw,np.array([0,0,8,0,0,0,0,0]))
lo=fit_dual(FG[:,:3],qw,mu[:3],tol=2e-12);hi=fit_dual(FG,qw,mu,tol=2e-12)
check('sphere_vmf_population_gap',abs(hi['dual']-lo['dual']),1e-9)
ms=np.zeros(8);ms[3]=eps;ff=fit_dual(FG,qw,ms,tol=2e-12)
check('local_mean_zero_quadratic_d3',abs(ff['dual']/eps**2-15/4),2e-5)
ms=np.array([eps,0,0]);ff=fit_dual(FG[:,:3],qw,ms,tol=2e-12)
check('local_first_order_d3',abs(ff['dual']/eps**2-1.5),1e-5)
# Continuous points can be outside a coarse discrete moment body.
tc,wc,FC=circle_grid(8,L=2);off=circle_features(np.array([np.pi/8]))[0]
feas=grid_feasibility(FC,off)
check('grid_infeasibility_detected',0 if not feas['feasible'] else 1,0)
fb=fit_dual(FC,wc,off)
check('infeasible_fit_flagged',0 if not fb['success'] else 1,0)
fb=fit_dual(FF,w,FF[0])
check('boundary_fit_flagged',0 if not fb['success'] else 1,0)
# Common shrinkage has a different, exactly declared target.
shr=.2;ms=(1-shr)*(wt/wt.sum()@FX)
lo=fit_dual(FF[:,:2],w,ms[:2]);fu=fit_dual(FF,w,ms);eta=np.r_[lo['lambda_'],0,0]
kl=fu['prob']@(FF@(fu['lambda_']-eta)-fu['psi']+lo['psi'])
check('common_shrinkage_nested_identity',abs(fu['dual']-lo['dual']-kl),1e-8)
report=dict(mode=MODE,checks=checks,passed=sum(c['passed'] for c in checks),total=len(checks),
    note='Numerical checks diagnose implementation; they do not certify asymptotic error rates or uniform coverage.',
    boundary_status=fb['status'],grid_feasibility=feas)
(CHECK_OUT/'numerical_unit_checks.json').write_text(json.dumps(report,indent=2)+'\n')
if not all(c['passed'] for c in checks):raise SystemExit('NUMERICAL CHECK FAILURE')
