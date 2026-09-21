"""Independent identity and integrity checks for saved Study C outputs.

Use the same GID_MODE and GID_OUTPUT_ROOT as the corresponding analysis run.
Every validation attempt writes study_c_validation.json, including failures.
"""
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from gid_pipeline import sphere_grid, sphere_features, analyze
from run_config import ROOT, OUT, MODE, PLOTS, PROTOCOLS, REFERENCE


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate():
    protocol_bytes=(PROTOCOLS/'study_c.json').read_bytes()
    protocol=json.loads(protocol_bytes)
    require((OUT/'study_c_protocol.json').read_bytes()==protocol_bytes,
            'Run protocol differs from the frozen protocol')
    metadata=json.loads((OUT/'study_c_metadata.json').read_text())
    failures=json.loads((OUT/'study_c_failures.json').read_text())
    require(metadata['mode']==MODE, 'Saved mode does not match requested validation mode')
    require(metadata['seed']==protocol['seed'], 'Saved seed differs from frozen seed')
    require(metadata['protocol_sha256']==hashlib.sha256(protocol_bytes).hexdigest(),
            'Protocol hash mismatch')
    with (OUT/'study_c_queries.csv').open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    expected_queries=40 if MODE=='paper' else 4
    expected_count=3*expected_queries
    require(len(rows)==expected_count, f'Expected {expected_count} query-temperature rows, found {len(rows)}')
    require(metadata['evaluated_query_count']==expected_queries and
            metadata['frozen_query_count']==40 and
            metadata['planned_query_temperature_count']==expected_count and
            metadata['query_temperature_count']==expected_count, 'Metadata count mismatch')
    with np.load(OUT/'study_c_arrays.npz',allow_pickle=False) as a:
        names=['representation','query','reference','evaluation']
        perm=np.random.default_rng(protocol['seed']).permutation(1797)
        frozen=[perm[:600],perm[600:640],perm[640:1240],perm[1240:]]
        sets=[set(a[name+'_ids'].tolist()) for name in names]
        require(all(np.array_equal(a[name+'_ids'],ids) for name,ids in zip(names,frozen)),
                'Partition IDs or order differ from the frozen full-data partition')
        require(set.union(*sets)==set(range(1797)), 'Partition does not cover every dataset ID')
        require(all(not sets[i]&sets[j] for i in range(4) for j in range(i+1,4)),
                'Partitions overlap')
        require(metadata['split_sizes']==dict(zip(names,[600,40,600,557])), 'Split size metadata mismatch')
        for name,ids in zip(names,frozen):
            require(metadata['split_sha256'][name]==hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest(),
                    f'{name} split hash mismatch')
        evaluated=frozen[1][:expected_queries]
        require(np.array_equal(a['evaluated_query_ids'],evaluated), 'Evaluated IDs differ from mode-specific frozen queries')
        require(metadata['evaluated_query_ids']==evaluated.tolist(), 'Evaluated query metadata mismatch')
        expected_pairs=[(beta,order,int(q)) for beta in [6,3,9] for order,q in enumerate(evaluated)]
        actual_pairs=[(int(r['beta']),int(r['query_order']),int(r['query_id'])) for r in rows]
        require(actual_pairs==expected_pairs, 'Query-temperature pairing, uniqueness, or order mismatch')
        require(metadata['failure_count']==len(failures), 'Failure count metadata mismatch')
        failed_rows=[r for r in rows if r['status']!='ok']
        failed_pairs={(int(r['beta']),int(r['query_id'])) for r in failed_rows}
        manifest_pairs={(int(r['beta']),int(r['query_id'])) for r in failures}
        require(len(failures)==len(failed_rows) and manifest_pairs==failed_pairs,
                'Failure manifest does not match failed query rows')
        require(not failures, f'Analysis recorded {len(failures)} unsuccessful query-temperature fits')
        X=a['spherical_data']; F=sphere_features(X)
        require(X.shape==(1797,3) and np.all(np.isfinite(X)), 'Invalid spherical coordinates')
        bad=(a['projection_norm']<=1e-12)|(a['standardized_norm']<=1e-12)
        require(np.array_equal(a['excluded_ids'],np.flatnonzero(bad)), 'Excluded IDs disagree with normalization rule')
        require(metadata['excluded_ids']==a['excluded_ids'].tolist(), 'Excluded ID metadata mismatch')
        require(not np.any(bad[evaluated]), 'An evaluated query has an invalid norm')
        fitids=a['reference_valid_ids']; testids=a['evaluation_valid_ids']
        require(np.array_equal(fitids,frozen[2][~bad[frozen[2]]]), 'Valid reference IDs are inconsistent')
        require(np.array_equal(testids,frozen[3][~bad[frozen[3]]]), 'Valid evaluation IDs are inconsistent')
        require(not set(evaluated)&set(fitids) and not set(evaluated)&set(testids), 'Query self inclusion detected')
        grid,qw=sphere_grid(64,128); FG=sphere_features(grid)
        basegrid,basew=sphere_grid(48,96); FB=sphere_features(basegrid)
        errors=dict(shared_pipeline_score_relative=0.,shared_pipeline_wald_relative=0.,
                    unit_sphere=0.,mean_match=0.,raw_frobenius=0.,residual_frobenius=0.,
                    empirical_logscore_identity=0.,analytic_vmf_I1=0.,analytic_vmf_partition=0.,
                    finer_partition=0.,test_gain=0.,saved_test_delta=0.,shared_pipeline_score=0.,
                    shared_pipeline_wald=0.,shared_pipeline_gap=0.,saved_covariance=0.,weight_normalization=0.)
        errors['unit_sphere']=float(np.max(abs(np.linalg.norm(X[~bad],axis=1)-1)))
        for row in rows:
            beta=int(row['beta']); q=int(row['query_id']); key=f'beta{beta}_q{q}_'
            require(int(row['query_label'])==int(a['labels'][q]), 'Query label mismatch')
            require(np.array_equal(a[key+'fit_status'],np.ones(4,dtype=bool)), 'Saved optimizer status indicates failure')
            require(float(row['max_moment_error'])<=1e-8 and
                    float(row['grid_gap_change'])<=1e-7 and
                    float(row['grid_test_gain_change'])<=1e-7,
                    'Saved fit violates protocol accuracy tolerances')
            w=a[key+'fit_weights']; wt=a[key+'test_weights']
            require(w.shape==(len(fitids),) and wt.shape==(len(testids),), 'Weight dimensions mismatch')
            require(np.all(np.isfinite(w)) and np.all(w>0) and np.all(np.isfinite(wt)) and np.all(wt>0),
                    'Invalid weights')
            weight_error=max(abs(w.sum()-1),abs(wt.sum()-1),
                             np.max(abs(w-a[key+'raw_fit_weights']/a[key+'raw_fit_weights'].sum())),
                             np.max(abs(wt-a[key+'raw_test_weights']/a[key+'raw_test_weights'].sum())))
            m=a[key+'moment']; l1=a[key+'reduced_parameter']; l2=a[key+'full_parameter']
            p1=logsumexp(FB[:,:3]@l1+np.log(basew)); p2=logsumexp(FB@l2+np.log(basew))
            M=(X[fitids].T*w)@X[fitids]
            redprob=np.exp(FB[:,:3]@l1+np.log(basew)-p1)
            Mred=(basegrid.T*redprob)@basegrid
            k=float(np.linalg.norm(l1))
            analytic_partition=float(k+np.log1p(-np.exp(-2*k))-np.log(2*k)) if k>1e-8 else k*k/6
            delta=F[testids]@l2-p2-F[testids,:3]@l1+p1
            checks={
                'weight_normalization':weight_error,
                'mean_match':np.max(abs(w@F[fitids]-m)),
                'raw_frobenius':abs(np.linalg.norm(M-np.eye(3)/3)-float(row['raw_second_norm'])),
                'residual_frobenius':abs(np.linalg.norm(M-Mred)-float(row['residual_second_norm'])),
                'empirical_logscore_identity':abs(w@(F[fitids]@l2-p2-F[fitids,:3]@l1+p1)-float(row['I2'])),
                'analytic_vmf_I1':abs(l1@m[:3]-analytic_partition-float(row['I1'])),
                'analytic_vmf_partition':abs(p1-analytic_partition),
                'finer_partition':abs(logsumexp(FG@l2+np.log(qw))-p2),
                'test_gain':abs(wt@delta-float(row['test_gain'])),
                'saved_test_delta':np.max(abs(delta-a[key+'delta_test'])),
            }
            common=analyze(F[fitids],w,FB,basew,design='importance',infer=False)
            require(common['reduced']['success'] and common['full']['success'],
                    'Shared-pipeline validation fit failed')
            checks.update(shared_pipeline_score_relative=abs(common['score']-float(row['model_score']))/max(1.,abs(common['score'])),
                          shared_pipeline_wald_relative=abs(common['wald']-float(row['sandwich_wald']))/max(1.,abs(common['wald'])),
                          shared_pipeline_score=abs(common['score']-float(row['model_score'])),
                          shared_pipeline_wald=abs(common['wald']-float(row['sandwich_wald'])),
                          shared_pipeline_gap=abs(common['gap']-float(row['I2'])),
                          saved_covariance=np.max(abs(common['Sigma']/len(fitids)-a[key+'sampling_covariance'])))
            require(all(np.isfinite(err) for err in checks.values()), 'Non-finite validation discrepancy')
            for name,err in checks.items(): errors[name]=max(errors[name],float(err))
        thresholds={name:1e-7 for name in errors}
        # Raw scores can be large. Relative checks retain the fixed 1e-7
        # tolerance; absolute differences are also exposed in the report.
        thresholds['shared_pipeline_score']=1e-3
        thresholds['shared_pipeline_wald']=1e-5
        passed=all(errors[name]<=thresholds[name] for name in errors)
        return dict(passed=passed,mode=MODE,evaluated_queries=expected_queries,
                    expected_query_temperature_comparisons=expected_count,
                    query_temperature_comparisons=len(rows),pairwise_disjoint=True,
                    all_ids_partitioned=True,exact_frozen_partition=True,
                    exact_query_temperature_pairing=True,failure_count=len(failures),
                    excluded_ids=a['excluded_ids'].tolist(),max_discrepancies=errors,
                    thresholds=thresholds,finer_rule=[64,128],
                    scope='Saved-data integrity, independent identities, and numerical refinement; not population inference.')


def main():
    try:
        out=validate()
    except Exception as exc:
        out=dict(passed=False,mode=MODE,error_type=type(exc).__name__,error=str(exc),
                 scope='Validation stopped on a saved-data integrity or numerical error.')
    (OUT/'study_c_validation.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2,allow_nan=False))
    if not out['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
