"""Targeted failure-injection checks for the final refinement-status audit."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from gid_pipeline import circle_grid,circle_features,analyze,audit_refinement

from run_config import ROOT,OUT,MODE,PLOTS,repetitions
CHECK_OUT=OUT/'study_b'
CHECK_OUT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(9281);x=rng.vonmises(0,2,100);F=circle_features(x)
t,w,Q=circle_grid(512);tt,ww,QQ=circle_grid(1024)
base=analyze(F,np.ones(100),Q,w,split=2,design='iid')
fine=analyze(F,np.ones(100),QQ,ww,split=2,design='iid',tol=2e-12)
assert base['valid'] and fine['valid']
checks=[]

def test(name,refined,expected):
    out=audit_refinement(deepcopy(base),refined,scale=100)
    good=out['refinement_status']==expected
    if expected in ('ok','not_checked'):
        good=good and out['valid']==base['valid'] and out['p_gap']==base['p_gap'] and out['se']==base['se']
    else:
        good=good and not out['valid'] and not out['wald_valid'] and np.isnan(out['se']) and all(np.isnan(out[k]) for k in ('p_gap','p_score','p_wald','naive_p','kish_p'))
    checks.append(dict(name=name,expected=expected,actual=out['refinement_status'],passed=bool(good),
        refined_full_status=out['refined_full_status'],refined_reduced_status=out['refined_reduced_status']))

test('unchecked_preserves_outputs',None,'not_checked')
test('successful_refinement_preserves_outputs',deepcopy(fine),'ok')
r=deepcopy(fine);r['full']['success']=False;r['full']['status']='injected_iteration_limit'
test('failed_full_fit_gates_every_inference_output',r,'unavailable_refined_fit')
r=deepcopy(fine);r['reduced']['success']=False;r['reduced']['status']='injected_boundary_failure'
test('failed_reduced_fit_gates_every_inference_output',r,'unavailable_refined_fit')
r=deepcopy(fine);r['reduced']=None
test('missing_required_reduced_fit_unavailable',r,'unavailable_refined_fit')
r=deepcopy(fine);r['gap']=float('nan')
test('nonfinite_refined_gap_unavailable',r,'unavailable_refined_fit')
r=deepcopy(fine);r['full']['lambda_'][0]=float('nan')
test('nonfinite_refined_parameter_unavailable',r,'unavailable_refined_fit')
r=deepcopy(fine);r['gap']=base['gap']+2e-3/100
test('excessive_scaled_difference_gates_every_inference_output',r,'excessive_scaled_difference')
r=deepcopy(fine);r['gap']=base['gap']+5e-4/100
test('subthreshold_scaled_difference_preserves_outputs',r,'ok')
result=dict(mode=MODE,passed=all(c['passed'] for c in checks),checks=checks,total=len(checks),tolerance=1e-3,
    provenance='Failure injection unit checks using a fixed small fitted fixture, not an additional calibration experiment.',
    policy_history='The gating threshold was added during the final implementation audit, not used to tune statistical results and not a uniform numerical error certificate.')
(CHECK_OUT/'refinement_gate_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(f"{sum(c['passed'] for c in checks)}/{len(checks)} refinement-gate checks passed")
assert result['passed']
