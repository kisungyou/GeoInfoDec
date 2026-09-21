"""Frozen, descriptive held-out digits application (REVIEW Study C).

Run from the package root with GID_MODE=smoke (default) or GID_MODE=paper.
The frozen protocol is protocols/study_c.json; each output keeps a copy.
This module creates no population p-values and never treats queries as IID units.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 's14-gid-matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import spearmanr
import sklearn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
from threadpoolctl import threadpool_limits
from gid_pipeline import sphere_grid, sphere_features, fit_dual

from run_config import ROOT, OUT, MODE, PLOTS, PROTOCOLS, REFERENCE


def dump_json(path, obj):
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, default=default, allow_nan=False) + '\n')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def csv_write(path, rows, fieldnames=None):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or (list(rows[0]) if rows else []))
        w.writeheader(); w.writerows(rows)


def moments(F, weights):
    mu = weights @ F
    C = F - mu
    return mu, (C * weights[:, None]).T @ C


def reduced_geometry(Fgrid, qw, reduced):
    eta = np.r_[reduced['lambda_'], np.zeros(5)]
    logits = Fgrid @ eta + np.log(qw)
    prob = np.exp(logits - logsumexp(logits))
    mu, J = moments(Fgrid, prob)
    B = np.linalg.solve(J[:3, :3], J[:3, 3:]).T
    N = np.c_[-B, np.eye(5)]
    S = J[3:, 3:] - B @ J[:3, 3:]
    return mu, J, N, S


def fit_pair(Fgrid, qw, m, initial=None):
    reduced = fit_dual(Fgrid[:, :3], qw, m[:3],
                       init=None if initial is None else initial[0]['lambda_'])
    initial_full = np.r_[reduced['lambda_'], np.zeros(5)] if initial is None else initial[1]['lambda_']
    full = fit_dual(Fgrid, qw, m, init=initial_full)
    return reduced, full


def test_gain(pair, Ftest, w):
    red, full = pair
    delta = Ftest @ full['lambda_'] - full['psi'] - Ftest[:, :3] @ red['lambda_'] + red['psi']
    return float(w @ delta), delta


def analyze_query(beta, order, query_id, Ffit, Ftest, w, wt, grids, labels, init=None):
    start = time.perf_counter()
    m, _ = moments(Ffit, w)
    coarse = fit_pair(*grids[0], m)
    fine = fit_pair(*grids[1], m, initial=coarse)
    red, full = fine
    gap_coarse = coarse[1]['dual'] - coarse[0]['dual']
    gap = full['dual'] - red['dual']
    gain, delta = test_gain(fine, Ftest, wt)
    gain_coarse, _ = test_gain(coarse, Ftest, wt)
    mu, J, N, S = reduced_geometry(*grids[1], red)
    r = m[3:] - mu[3:]
    C = Ffit - m
    # Descriptive self-normalized sandwich covariance of the weighted mean.
    # This would have an IID ratio-estimator interpretation only with an
    # appropriate independent-unit model, which is NOT asserted here.
    V = (C * w[:, None]).T @ (C * w[:, None])
    Omega = N @ V @ N.T
    model_score = float(len(Ffit) * r @ np.linalg.solve(S, r))
    wald = float(r @ np.linalg.solve(Omega, r))
    residual_norm = float(np.linalg.norm(r))
    raw_norm = float(np.linalg.norm(m[3:]))
    errors = [float(f['moment_error']) for f in [*coarse, *fine]]
    grid_error = max(abs(gap-gap_coarse), abs(gain-gain_coarse))
    statuses = [bool(f['success']) for f in [*coarse, *fine]]
    ok = all(statuses) and max(errors) <= 1e-8 and grid_error <= 1e-7
    row = dict(beta=beta, query_order=order, query_id=int(query_id), query_label=int(labels[query_id]),
               status='ok' if ok else 'accuracy_failure', I1=float(red['dual']), I2=float(gap),
               test_gain=gain, raw_second_norm=raw_norm, residual_second_norm=residual_norm,
               model_score=model_score, sandwich_wald=wald, fit_ess=float(1/(w@w)),
               test_ess=float(1/(wt@wt)), mean_length=float(np.linalg.norm(m[:3])),
               max_moment_error=max(errors), grid_gap_change=abs(gap-gap_coarse),
               grid_test_gain_change=abs(gain-gain_coarse),
               min_schur_eigenvalue=float(np.linalg.eigvalsh(S).min()),
               min_omega_eigenvalue=float(np.linalg.eigvalsh(Omega).min()),
               full_condition=float(full['condition']),
               iterations=sum(int(f['iterations']) for f in [*coarse, *fine]),
               elapsed_seconds=time.perf_counter()-start)
    extra = dict(reduced_parameter=red['lambda_'], full_parameter=full['lambda_'],
                 moment=m, reduced_model_moment=mu, residual=r, reduced_full_curvature=J,
                 sampling_covariance=V, residual_map=N, schur=S, omega=Omega,
                 delta_test=delta, fit_status=np.asarray(statuses))
    return row, extra


def summarize(rows, planned_queries):
    fields = ['I1', 'I2', 'raw_second_norm', 'residual_second_norm', 'model_score', 'sandwich_wald']
    summaries = []
    policies = []
    for beta in [6, 3, 9]:
        block = [r for r in rows if r['beta']==beta and r['status']=='ok']
        if not block: continue
        gain = np.asarray([r['test_gain'] for r in block])
        i1 = np.asarray([r['I1'] for r in block])
        low_idx = np.argsort(i1, kind='stable')[:max(1,len(block)//4)]
        result = dict(beta=beta, successful_queries=len(block), failed_queries=planned_queries-len(block),
                      positive_gain_queries=int((gain>0).sum()), negative_gain_queries=int((gain<0).sum()),
                      mean_gain=float(gain.mean()), median_gain=float(np.median(gain)),
                      min_gain=float(gain.min()), max_gain=float(gain.max()),
                      q25_gain=float(np.quantile(gain,.25)), q75_gain=float(np.quantile(gain,.75)),
                      median_I1=float(np.median(i1)), median_I2=float(np.median([r['I2'] for r in block])),
                      low_I1_queries=len(low_idx), low_I1_max=float(i1[low_idx].max()),
                      low_I1_mean_gain=float(gain[low_idx].mean()),
                      low_I1_positive=int((gain[low_idx]>0).sum()),
                      low_I1_query_ids=[block[i]['query_id'] for i in low_idx],
                      low_I1_median_I2=float(np.median([block[i]['I2'] for i in low_idx])),
                      spearman={field:float(spearmanr([r[field] for r in block],gain).statistic)
                                if len(block)>1 and np.ptp(gain)>0 and np.ptp([r[field] for r in block])>0 else None
                                for field in fields},
                      rank_overlap_with_I2={})
        idx_i2 = set(np.argsort(-np.asarray([r['I2'] for r in block]),kind='stable')[:len(block)//2].tolist())
        for field in fields:
            vals=np.asarray([r[field] for r in block])
            idx=np.argsort(-vals,kind='stable')[:len(block)//2]
            result['rank_overlap_with_I2'][field]=len(idx_i2 & set(idx.tolist()))
            policies.append(dict(beta=beta, policy='top_half_'+field, selected=len(idx),
                                 selected_mean_gain=float(gain[idx].mean()) if len(idx) else 0.,
                                 per_query_policy_gain=float(gain[idx].sum()/len(block)),
                                 negative_selected=int((gain[idx]<0).sum()),
                                 query_ids=';'.join(str(block[i]['query_id']) for i in idx)))
        for name,selected in [('always_vMF',np.array([],dtype=int)),('always_FB',np.arange(len(block)))]:
            policies.append(dict(beta=beta,policy=name,selected=len(selected),
                                 selected_mean_gain=float(gain[selected].mean()) if len(selected) else 0.,
                                 per_query_policy_gain=float(gain[selected].sum()/len(block)),
                                 negative_selected=int((gain[selected]<0).sum()),
                                 query_ids=';'.join(str(block[i]['query_id']) for i in selected)))
        summaries.append(result)
    return summaries, policies


def plot(rows):
    from figure_style import plt,save_figure,COLORS,TEXT_WIDTH_IN
    from matplotlib.lines import Line2D
    block = [r for r in rows if float(r['beta'])==6 and r['status']=='ok']
    if not block: return None
    fig,axes=plt.subplots(1,3,figsize=(TEXT_WIDTH_IN,2.75),sharey=True)
    fig.subplots_adjust(left=.10,right=.987,bottom=.34,top=.94,wspace=.24)
    low=set(np.argsort([float(r['I1']) for r in block],kind='stable')[:max(1,len(block)//4)])
    for ax,field,label in zip(axes,['raw_second_norm','residual_second_norm','I2'],
                            ['Raw second-moment\nnorm','vMF residual\nnorm',r'Information gap $I_2$'+'\n(nats)']):
        for selected,color,marker in ((False,COLORS['blue'],'o'),(True,COLORS['orange'],'D')):
            idx=[i for i in range(len(block)) if (i in low)==selected]
            ax.scatter([float(block[i][field]) for i in idx],[float(block[i]['test_gain']) for i in idx],
                       c=color,marker=marker,s=25,alpha=.90,edgecolors='white',linewidths=.3,zorder=3)
        ax.axhline(0,color=COLORS['gray'],linewidth=.7,zorder=0)
        ax.set_xlabel(label)
        ax.set(ylim=(-.025,.465),yticks=[0,.2,.4])
    axes[0].set(xticks=[.2,.4,.6],ylabel='Held-out gain (nats)')
    axes[1].set(xticks=[.1,.15,.2])
    axes[2].set(xticks=[0,.1,.2,.3])
    handles=[Line2D([],[],ls='',marker=m,color=c,markersize=4.5) for c,m in ((COLORS['blue'],'o'),(COLORS['orange'],'D'))]
    fig.legend(handles,['Other queries','Lowest first-gap quartile'],ncol=2,
               loc='lower center',bbox_to_anchor=(.54,.015))
    meta=save_figure(fig,OUT/'study_c_heldout');plt.close(fig)
    return meta


def write_report(meta, summaries, policies, rows):
    primary=next((s for s in summaries if s['beta']==6),None)
    evaluated=meta['evaluated_query_count']
    text=['# Study C: frozen held-out digits analysis','',
          f"Mode: **{MODE}**. Evaluated queries: {evaluated} of the frozen 40-query pool; planned query-temperature comparisons: {3*evaluated}.", '',
          '## Protocol and scope','',
          'A fixed permutation assigns 600 images to representation training, 40 to the query pool, 600 to reference fitting, and 557 to held-out evaluation. The four sets are pairwise disjoint. Standardization and three-dimensional PCA use only the representation set. Projected vectors are normalized onto the sphere; query weights use cosine similarity in the original standardized 64-dimensional feature space. The primary inverse temperature is 6; 3 and 9 are sensitivity analyses.', '',
          ('Paper mode evaluates all 40 frozen queries.' if MODE=='paper' else 'Smoke mode evaluates the first four frozen queries at all three temperatures, retaining the complete training, reference, and evaluation pools. These diagnostic results do not replace the paper results.'), '',
          'The estimand concerns this constructed spherical representation. Queries share reference and evaluation images. The source contains repeated handwriting observations, and writer identifiers are unavailable. Results are descriptive: no population p-values, independent-query standard errors, or population confidence intervals are reported. Labels identify queries but do not define geometric adequacy. The protocol is prespecified within this revision, not externally preregistered.', '',
          '## Results for this run','',
          '| Inverse temperature | Successful / planned | Positive gains | Mean gain | Median gain | Range | Median second gap |',
          '|---:|---:|---:|---:|---:|---|---:|']
    for s in summaries:
        text.append(f"| {s['beta']} | {s['successful_queries']}/{evaluated} | {s['positive_gain_queries']}/{s['successful_queries']} | {s['mean_gain']:.4f} | {s['median_gain']:.4f} | {s['min_gain']:.4f} to {s['max_gain']:.4f} | {s['median_I2']:.4f} |")
    text += ['', 'Summaries use successful analyses only. The failure manifest retains every unsuccessful planned analysis. All gains and information gaps use natural logarithms; negative test gains are preserved.', '']
    if primary is not None:
        text += ['| Reference summary | Descriptive Spearman association with held-out gain | Top-half mean gain | Negative gains among selected |',
                 '|---|---:|---:|---:|']
        for field,rho in primary['spearman'].items():
            p=next(p for p in policies if p['beta']==6 and p['policy']=='top_half_'+field)
            rho_text=f'{rho:.3f}' if rho is not None else 'undefined'
            text.append(f"| {field} | {rho_text} | {p['selected_mean_gain']:.4f} | {p['negative_selected']}/{p['selected']} |")
        budget=primary['successful_queries']//2
        text += ['',f"The lower first-gap quartile contains {primary['low_I1_queries']} successful queries, with a maximum first gap of {primary['low_I1_max']:.4f} nats. Its mean held-out gain is {primary['low_I1_mean_gain']:.4f} nats, and {primary['low_I1_positive']} gains are positive. Its median second gap is {primary['low_I1_median_I2']:.4f} nats.", '',
                 f"The top-half comparison selects {budget} of {primary['successful_queries']} successful primary queries using reference information only. The score and information-gap selections share {primary['rank_overlap_with_I2']['model_score']} of these {budget} queries. The complete policy results include always-vMF and always-Fisher-Bingham. This describes one fixed finite benchmark and does not establish a tuned deployment rule, a unique predictive advantage, a useful acceptance cutoff, or an optimal complexity penalty.", '']
    valid=[r for r in rows if r['status']=='ok']
    text += ['## Numerical and reproducibility records','',
             f"Successful query-temperature analyses: {len(valid)}/{len(rows)}. Failed analyses: {meta['failure_count']}.", '']
    if valid:
        text += [f"Among successful analyses, the maximum moment residual is {max(r['max_moment_error'] for r in valid):.3g}; maximum coarse/fine changes are {max(r['grid_gap_change'] for r in valid):.3g} for the second gap and {max(r['grid_test_gain_change'] for r in valid):.3g} for test gain.", '']
    text += ['The shared solver fits unpenalized dual objectives. The integration rule combines Gauss-Legendre nodes in the vertical coordinate with equally spaced azimuths, using 24 by 48 nodes and a 48 by 96 refinement. Each query records status and numerical errors. No ridge, target shrinkage, silent gap clipping, or significance threshold is used.', '',
             'The raw second-moment norm is the Frobenius norm of the second moment minus one third of the identity. The residual norm subtracts the fitted vMF second moment. The model score uses the reduced-model Schur complement and reference sample count. The sandwich Wald descriptor uses the normalized squared-weight empirical covariance. Its numerical value is reported without claiming a population null distribution.', '',
             f"Seed: {meta['seed']}. Data SHA-256: `{meta['data_sha256']}`. Split SHA-256: `{meta['permutation_sha256']}`. Analysis and numerical-data writing time: {meta['elapsed_seconds']:.2f} seconds (excludes imports, report writing, and plotting). Total optimization iterations: {sum(r['iterations'] for r in rows)}.", '',
             'Machine-readable files retain every planned query-temperature row, all policy comparisons, all four partition ID lists, evaluated query IDs, raw and normalized weights, frozen standardization/PCA parameters, spherical coordinates, fitted natural parameters, moment covariances, and evaluation log-score differences where available. The failure manifest records all unsuccessful analyses; an empty list explicitly records no failures.', '',
             '## Data attribution','',
             'E. Alpaydin and C. Kaynak (1998), Optical Recognition of Handwritten Digits, UCI Machine Learning Repository, DOI: [10.24432/C50P49](https://doi.org/10.24432/C50P49). The UCI repository identifies the dataset license as [CC BY 4.0](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). The scikit-learn [load_digits documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) identifies its bundled 1,797-image data as the original UCI test subset. These metadata were checked on 2026-09-16.', '',
             'Any requested figure is generated from these derived data; it contains no externally sourced artwork. The data remain attributable to Alpaydin and Kaynak under the UCI dataset license. No new license is asserted for the manuscript or its code.', '',
             f"Reproduce with `GID_MODE={MODE} python code/study_c_digits.py`, followed by `GID_MODE={MODE} python code/study_c_validate.py`. Set `GID_PLOTS=1` to render figures. The output directory may be set with `GID_OUTPUT_ROOT`; both commands must use the same setting. No data download is required once scikit-learn is installed. Environment versions are in `study_c_metadata.json`.", '',
             f"The frozen protocol is copied from `protocols/study_c.json`. Saved paper outputs for comparison are in `{REFERENCE.name}`. Validation checks mode-specific counts and query pairing, partition disjointness, Frobenius identities, the analytic vMF normalizer, the empirical log-score identity, shared-pipeline matrix scaling, and an additional 64 by 128 integration rule. Consult `study_c_validation.json` after running validation for its actual status."]
    (OUT/'study_c_report.md').write_text('\n'.join(text)+'\n')
    tex=[r'\begin{tabular}{@{}rrrrrr@{}}',r'\toprule',
         rf'$\beta$ & $+\Delta$/{evaluated} & Mean $\Delta$ & Median $\Delta$ & Min. $\Delta$ & Max. $\Delta$\\',r'\midrule']
    for s in summaries:
        tex.append(f"{s['beta']} & {s['positive_gain_queries']} & {s['mean_gain']:.3f} & {s['median_gain']:.3f} & {s['min_gain']:.3f} & {s['max_gain']:.3f}\\\\")
    tex += [r'\bottomrule',r'\end{tabular}']
    (OUT/'study_c_table.tex').write_text('\n'.join(tex)+'\n')


def failed_row(beta, order, query_id, labels, status, elapsed=0.):
    row=dict(beta=beta,query_order=order,query_id=int(query_id),query_label=int(labels[query_id]),status=status)
    for field in ('I1','I2','test_gain','raw_second_norm','residual_second_norm','model_score',
                  'sandwich_wald','fit_ess','test_ess','mean_length','max_moment_error',
                  'grid_gap_change','grid_test_gain_change','min_schur_eigenvalue',
                  'min_omega_eigenvalue','full_condition'):
        row[field]=None
    row.update(iterations=0,elapsed_seconds=elapsed)
    return row


def main():
    started=time.perf_counter()
    protocol_path=PROTOCOLS/'study_c.json'
    protocol_bytes=protocol_path.read_bytes()
    protocol=json.loads(protocol_bytes)
    (OUT/'study_c_protocol.json').write_bytes(protocol_bytes)
    data=load_digits()
    pixels=np.asarray(data.data,dtype=np.float64)
    labels=np.asarray(data.target,dtype=np.int64)
    rng=np.random.default_rng(protocol['seed'])
    perm=rng.permutation(len(pixels))
    split={'representation':perm[:600],'query':perm[600:640],
           'reference':perm[640:1240],'evaluation':perm[1240:]}
    assert len(np.unique(np.concatenate(list(split.values()))))==len(pixels)
    scaler=StandardScaler().fit(pixels[split['representation']])
    Z=scaler.transform(pixels)
    pca=PCA(n_components=3,svd_solver='full').fit(Z[split['representation']])
    projection=pca.transform(Z)
    norm=np.linalg.norm(projection,axis=1)
    znorm=np.linalg.norm(Z,axis=1)
    bad=(norm <= 1e-12)|(znorm <= 1e-12)
    X=np.divide(projection,norm[:,None],out=np.zeros_like(projection),where=norm[:,None]>1e-12)
    Zunit=np.divide(Z,znorm[:,None],out=np.zeros_like(Z),where=znorm[:,None]>1e-12)
    # No replacement or reallocation after exclusions; IDs remain in the split file.
    reference=split['reference'][~bad[split['reference']]]
    evaluation=split['evaluation'][~bad[split['evaluation']]]
    F=sphere_features(X)
    grids=[]
    for nz,nphi in [protocol['quadrature']['coarse'],protocol['quadrature']['fine']]:
        grid,qw=sphere_grid(nz=nz,nphi=nphi)
        grids.append((sphere_features(grid),qw))
    arrays={**{k+'_ids':v for k,v in split.items()},'reference_valid_ids':reference,
            'evaluation_valid_ids':evaluation,'excluded_ids':np.flatnonzero(bad),
            'spherical_data':X,'projection_norm':norm,'standardized_norm':znorm,
            'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_,
            'pca_components':pca.components_,'pca_mean':pca.mean_,
            'pca_explained_variance_ratio':pca.explained_variance_ratio_,
            'labels':labels}
    evaluated=split['query'] if MODE=='paper' else split['query'][:4]
    arrays['evaluated_query_ids']=evaluated
    rows=[]; failures=[]
    for beta in [6,3,9]:
        for order,q in enumerate(evaluated):
            query_started=time.perf_counter()
            if bad[q]:
                row=failed_row(beta,order,q,labels,'invalid_query_norm')
                rows.append(row); failures.append(row)
                continue
            try:
                raw=np.exp(beta*(Zunit[reference]@Zunit[q])); w=raw/raw.sum()
                rawt=np.exp(beta*(Zunit[evaluation]@Zunit[q])); wt=rawt/rawt.sum()
                row,extra=analyze_query(beta,order,q,F[reference],F[evaluation],w,wt,grids,labels)
                key=f'beta{beta}_q{int(q)}_'
                arrays[key+'raw_fit_weights']=raw; arrays[key+'fit_weights']=w
                arrays[key+'raw_test_weights']=rawt; arrays[key+'test_weights']=wt
                arrays.update({key+k:v for k,v in extra.items()})
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                row=failed_row(beta,order,q,labels,'analysis_exception',time.perf_counter()-query_started)
                failures.append(dict(**row,error_type=type(exc).__name__,error=str(exc)))
            else:
                if row['status']!='ok': failures.append(row)
            rows.append(row)
        print(f"beta={beta}: completed {sum(r['beta']==beta for r in rows)} queries",flush=True)
    csv_write(OUT/'study_c_queries.csv',rows)
    np.savez_compressed(OUT/'study_c_arrays.npz',**arrays)
    summaries,policies=summarize(rows,len(evaluated))
    dump_json(OUT/'study_c_summary.json',summaries)
    csv_write(OUT/'study_c_policies.csv',policies,fieldnames=['beta','policy','selected',
              'selected_mean_gain','per_query_policy_gain','negative_selected','query_ids'])
    dump_json(OUT/'study_c_failures.json',failures)
    meta=dict(mode=MODE,evaluated_query_count=len(evaluated),evaluated_query_ids=evaluated,
              frozen_query_count=len(split['query']),planned_query_temperature_count=3*len(evaluated),
              seed=protocol['seed'],protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),
              source_sha256={name:hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()
                             for name in ('gid_pipeline.py','study_c_digits.py','study_c_validate.py','run_config.py')},
              data_sha256=digest(pixels),label_sha256=digest(labels),permutation_sha256=digest(perm),
              split_sha256={k:digest(v) for k,v in split.items()},
              split_sizes={k:len(v) for k,v in split.items()},
              excluded_ids=np.flatnonzero(bad),dataset='sklearn.datasets.load_digits, all 1797 images',
              dataset_source='https://doi.org/10.24432/C50P49',dataset_license='CC BY 4.0 (UCI metadata)',
              source_access_date='2026-09-16',
              timing_scope='Dataset loading, preprocessing, all coarse/fine fits, diagnostics, numerical result writing; excludes import startup, report writing, and plotting',
              sum_query_analysis_seconds=sum(r['elapsed_seconds'] for r in rows),
              independent_unit_scope='Finite benchmark only; writer IDs unavailable; queries share pools',
              preprocessing=protocol['representation'],pca_explained_variance_ratio=pca.explained_variance_ratio_,
              versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                            sklearn=sklearn.__version__,matplotlib=matplotlib.__version__),
              executable=Path(sys.executable).name,elapsed_seconds=time.perf_counter()-started,
              failure_count=len(failures),query_temperature_count=len(rows))
    dump_json(OUT/'study_c_metadata.json',meta)
    write_report(meta,summaries,policies,rows)
    if PLOTS: plot(rows)
    print(json.dumps(summaries,indent=2),flush=True)
    if failures: raise SystemExit('Study C has failed fits: inspect study_c_failures.json')


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
