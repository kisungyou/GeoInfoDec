"""Generate mode-labelled reports and LaTeX tables from the executed outputs."""
from study_ab import *
from run_config import ROOT,OUT,MODE,PLOTS,repetitions
import platform,hashlib,scipy


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def mode_note():
    if MODE=='smoke':
        return ('Smoke mode is an execution check with five replicates per cell. '
                'Its rates and intervals are retained to check the full reporting path; '
                'they are not paper results or evidence of calibration, coverage, or power.')
    return ('Paper mode uses the full frozen replication counts. Rates and intervals '
            'below are calculated from this run, including every configured replicate in the denominator.')


def refinement_note(manifest,limit):
    counts=manifest['refinement_status_counts']
    return (f"Up to the first {limit} available replicates per cell undergo grid refinement. "
            f"Recorded statuses: {counts['ok']} passed, {counts['not_checked']} not checked, "
            f"{counts['unavailable_refined_fit']} unavailable refined fits, and "
            f"{counts['excessive_scaled_difference']} excessive scaled differences. "
            'Checked rows require successful finite fits and n times the absolute gap change at most 0.001. '
            'A failed check makes inference unavailable while retaining diagnostics. '
            'This operational threshold was added during the final implementation audit, '
            'without tuning statistical results; it is not a uniform integration-error certificate.')


def main():
    oa=OUT/'study_a';ob=OUT/'study_b'
    A=read_csv(oa/'rejection_summary.csv');B=read_csv(ob/'power_estimation_summary.csv')
    ca=json.loads((oa/'config.json').read_text());cb=json.loads((ob/'config.json').read_text())
    ma=json.loads((oa/'numerical_manifest.json').read_text());mb=json.loads((ob/'numerical_manifest.json').read_text())
    supplement=json.loads((ob/'supplement_config.json').read_text())
    holm=json.loads((ob/'finite_tail_holm_summary.json').read_text())
    checks=json.loads((ob/'numerical_unit_checks.json').read_text())
    for metadata in (ca,cb,ma,mb,supplement,holm,checks):
        if metadata.get('mode')!=MODE:
            raise ValueError('Output mode does not match GID_MODE; regenerate all studies in one output root')
    keys=list(dict.fromkeys(r['cell'] for r in A))
    grouped={key:{r['method']:r for r in A if r['cell']==key} for key in keys}
    tex=['\\begin{tabular}{llrrrrrr}','\\toprule','Domain and law & $n$ & Gap & Score & Wald & Naive & Kish & MC s.e.\\\\','\\midrule']
    labels={'uniform_direct':'Uniform, direct','vm2_direct':'von Mises, direct','cos4_direct':'Fourth harmonic, direct','uniform_importance':'Uniform, importance','vm2_importance':'von Mises, importance','vm8_direct':'vMF(8), direct'}
    for key in keys:
        di=grouped[key];r0=next(iter(di.values()))
        vals=[float(di[k]['rejection_rate']) for k in ('p_gap','p_score','p_wald','naive_p','kish_p')]
        tex.append(('Circle: ' if r0['domain']=='circle' else 'Sphere: ')+labels[r0['law']]+' & '+r0['n']+' & '+' & '.join(f'{v:.3f}' for v in vals)+' & '+f"{float(di['p_gap']['mc_se']):.3f}"+r'\\')
    tex+=['\\bottomrule','\\end{tabular}'];(oa/'calibration_table.tex').write_text('\n'.join(tex)+'\n')
    tex=['\\begin{tabular}{rrrrrr}','\\toprule',r'$n$ & $\beta$ & $I_2$ & Bias & RMSE & Coverage\\','\\midrule']
    for r in B:
        if r['regime']=='fixed':tex.append(' & '.join([r['n'],r['b']]+[f"{float(r[k]):.4f}" for k in ['population_gap','bias','rmse','coverage']])+r'\\')
    tex+=['\\bottomrule','\\end{tabular}'];(ob/'estimation_table.tex').write_text('\n'.join(tex)+'\n')
    count_descriptions=[]
    for domain in ('circle','sphere'):
        selected=[c for c in ca['cells'] if c['domain']==domain]
        for count in sorted(set(c['repeats'] for c in selected)):
            group=[c for c in selected if c['repeats']==count]
            count_descriptions.append(f"{len(group)} {domain} cells with {count} replicates each")
    aa=[f'# Study A: {MODE} fitted-gap calibration','',mode_note(),'',
        'All values come from retained per-replicate records. The sampling configuration was saved before execution. Failed fits and unavailable inference remain in the records and in configured rejection denominators.',
        '',f"Executed {ma['total_replicates']} replicates: "+'; '.join(count_descriptions)+f". Unavailable fits/inference: {len(ma['failed'])}.",
        '', '## Executed rates', '',
        '| Cell | Repeats | Fitted gap rate (95% Wilson interval) | Score | Wald | Naive | Kish |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in keys:
        d=grouped[key];g=d['p_gap']
        aa.append(f"| {key} | {g['repeats']} | {float(g['rejection_rate']):.3f} ({float(g['wilson_low']):.3f}, {float(g['wilson_high']):.3f}) | "+' | '.join(f"{float(d[k]['rejection_rate']):.3f}" for k in ('p_score','p_wald','naive_p','kish_p'))+' |')
    aa+=['', 'Ordinary chi-square and Kish-scaled comparisons are diagnostics. Scalar ESS does not generally supply the covariance required for informative-weight or misspecified moment-null inference. The fourth-harmonic null has exact doubled-reference eigenvalues 1.4 and 0.6.',
        '', '## Numerical evidence', '',
        f"Maximum fitted moment residual: {ma['max_moment_error']:.3g}. Maximum observed Hessian condition: {ma['max_condition']:.3g}. Maximum absolute gap versus adjacent KL discrepancy: {ma['max_abs_kl_identity']:.3g}.",
        '', refinement_note(ma,25),
        '',f"Maximum observed n-scaled grid refinement difference: {ma['max_scaled_refinement']:.3g}. Up to the first ten available replicates per cell also undergo rotation; the maximum observed n-scaled rotation difference was {ma['max_scaled_rotation']:.3g}.",
        '',f"Spherical tails use an add-one scrambled Sobol Gaussian rule with 16,384 nodes. Increasing to 65,536 nodes on checked replicates changed p-values by at most {ma['max_reference_refinement']:.4g}. Circle tails use 256-angle integration. These approximate an asymptotic reference law, not an exact randomization test.",
        '',f"Cumulative solver time: {ma['fit_seconds']:.2f} seconds; complete study time excluding plotting: {ma['total_seconds']:.2f} seconds. Hardware-dependent timings are descriptive.",
        '', 'The stored score is a² rᵀS⁻¹r; both it and 2a²I use eigenvalues of S⁻¹ᐟ²ΩS⁻¹ᐟ². The stored remainder is a²I minus score/2.',
        '', 'NPZ files preserve observations and raw weights. CSV files preserve seeds, p-values, diagnostics, statuses, rates, Monte Carlo standard errors, and Wilson intervals.']
    (oa/'REPORT.md').write_text('\n'.join(aa)+'\n')
    fixed=[r for r in B if r['regime']=='fixed'];local=[r for r in B if r['regime']=='local']
    geometry=read_csv(ob/'sphere_population_geometry.csv');hierarchy=read_csv(ob/'cos4_population_hierarchy.csv')
    stress=read_csv(ob/'pareto_stress_summary.csv')
    bb=[f'# Study B: {MODE} interpretation, power, and estimation','',mode_note(),'',
        f"Executed {mb['total_replicates']} main-study repetitions across {len(B)} fixed/local cells, with {cb['repeats']} replicates per cell. Unavailable fits/inference: {len(mb['failed'])}. Circle alternatives have density proportional to exp(2 cos(theta) + beta cos(2 theta)).",
        '', '## Fixed alternatives','','| n | beta | Repeats | Population I2 | Bias | RMSE | 95% interval coverage |','|---:|---:|---:|---:|---:|---:|---:|']
    for r in fixed:
        bb.append('| '+' | '.join([r['n'],r['b'],r['repeats']]+[f"{float(r[k]):.4f}" for k in ['population_gap','bias','rmse','coverage']])+' |')
    bb+=['', 'Intervals use the fixed nonzero-gap delta method. Near-null interval outputs in the full CSV are diagnostic only, and no finite-sample or uniform near-null coverage guarantee follows. Coverage Wilson intervals and Monte Carlo standard errors use the recorded cell replication counts.',
        '', '## Local alternatives','','| n | b | Repeats | Fitted gap rejection | Model chi-square rejection | Limiting power |','|---:|---:|---:|---:|---:|---:|']
    for r in local:
        bb.append('| '+' | '.join([r['n'],r['b'],r['repeats']]+[f"{float(r[k]):.4f}" for k in ['p_gap_rejection','naive_p_rejection','asymptotic_local_power']])+' |')
    bb+=['', 'The sequence is beta=b/sqrt(n). The limiting noncentral chi-square approximation is an asymptotic comparison, not a power-dominance claim.',
        '', '## Population interpretation','','| Law | Resultant | Raw anisotropy | Residual anisotropy | I1 | I2 |','|---|---:|---:|---:|---:|---:|']
    for r in geometry:
        bb.append('| '+r['law']+' | '+' | '.join(f"{float(r[k]):.4f}" for k in ('resultant','raw_anisotropy','residual_anisotropy','I1','I2'))+' |')
    bb+=['', 'These deterministic population calculations retain the full quadrature settings in both modes: 48x96, independently refined to 96x192. They do not use reduced smoke integration accuracy.',
        '', 'Fourth-harmonic population gaps, in level order: '+', '.join(f"{float(r['gap']):.4f}" for r in hierarchy)+' nats.',
        '', '## Fixed-family finite-tail assessment','',
        f"The complete planned family is I1, I2, I3, I4, and D4-D1. It was evaluated in {holm['repeats']} samples of size {holm['n']}; {holm['failures']} samples had unavailable tests. The first gap was nonsignificant in {holm['first_gap_nonrejections']} samples.",
        '', '| Holm-adjusted comparison | Rejections | Repeats | Rate | 95% Wilson interval |','|---|---:|---:|---:|---:|']
    for key in ('I1','I2','I3','I4','tail'):
        r=holm['holm_'+key]
        bb.append(f"| {key} | {r['rejections']} | {holm['repeats']} | {r['rate']:.4f} | ({r['wilson_low']:.4f}, {r['wilson_high']:.4f}) |")
    fwer=holm['holm_any_true_null_rejections'];total=holm['repeats'];fl,fu=wilson(fwer,total)
    bb += ['',f"At least one true-null comparison was rejected in {fwer}/{total} samples ({fwer/total:.4f}; Wilson interval {fl:.4f}, {fu:.4f}). The theoretical family-wise guarantee is asymptotic; this run does not establish exact finite-sample control.",
        '', '## Descriptive heavy-weight stress','',
        f"Three circle laws were each evaluated in {supplement['stress_repeats']} paired samples of size {supplement['n']}, giving {sum(int(r['repeats']) for r in stress)} weighted profiles. Equal weights and independent Pareto weights of tail index {supplement['pareto_shape']} use the same observations within each pair. The Pareto weights have infinite second moment; these effect-size and ESS summaries do not validate Gaussian inference.",
        '', '## Numerical audit','',refinement_note(mb,10),'',
        f"{checks['passed']} of {checks['total']} numerical identity checks passed. Both modes run the full checks, including basis/rotation invariance, the fourth-harmonic spectrum, derivatives, count compression, and boundary/grid-infeasibility statuses.",
        '',f"Maximum moment residual in the main B simulation: {mb['max_moment_error']:.3g}; maximum absolute adjacent-KL discrepancy: {mb['max_abs_kl_identity']:.3g}; maximum n-scaled refinement change: {mb['max_scaled_refinement']:.3g}. These are diagnostics, not certified uniform error or coverage bounds."]
    gate_path=ob/'refinement_gate_checks.json'
    if gate_path.exists():
        gate=json.loads(gate_path.read_text())
        if gate.get('mode')!=MODE:raise ValueError('Refinement-check mode does not match GID_MODE')
        bb+=['',f"{sum(c['passed'] for c in gate['checks'])} of {gate['total']} refinement-gate failure-injection checks passed."]
    (ob/'REPORT.md').write_text('\n'.join(bb)+'\n')
    source_names=['gid_pipeline.py','run_config.py','study_ab.py','study_b_supplement.py','numerical_checks.py','summarize_ab.py','refinement_gate_checks.py']
    dump(ob/'environment_manifest.json',dict(mode=MODE,plots=PLOTS,python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,
        actual_counts=dict(study_a=ma['total_replicates'],study_b=mb['total_replicates'],stress_profiles=sum(int(r['repeats']) for r in stress),finite_tail=holm['repeats']),
        source_sha256={n:hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest() for n in source_names if (ROOT/'code'/n).is_file()},
        command_environment=dict(GID_MODE=MODE,GID_PLOTS='1' if PLOTS else '0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
        output_selection='GID_OUTPUT_ROOT when provided; otherwise results/GID_MODE',
        commands=['python code/study_ab.py','python code/study_b_supplement.py','python code/numerical_checks.py','python code/refinement_gate_checks.py','python code/summarize_ab.py']))


if __name__=='__main__':main()
