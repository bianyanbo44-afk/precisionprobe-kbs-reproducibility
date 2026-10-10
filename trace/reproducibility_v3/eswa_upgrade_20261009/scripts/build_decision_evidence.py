"""Descriptive decision evidence from retained records; no new inference."""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_retrospective import expected_review
from build_paper_assets import tex_table, configure_plot, MODEL_NAMES, require_matplotlib_panel_alignment

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'analysis/validation'
TABLES=BASE/'manuscript/generated'
FIGURES=BASE/'manuscript/figures'

def main():
    frame=pd.read_csv(OUT/'task_features.csv',float_precision='round_trip')
    capacities=[]; public_rows=[]; curves=[]; summaries=[]
    same_stage=[]
    for condition,cell in frame.groupby('condition',sort=True):
        model,precision=condition.rsplit('_',1)
        label=MODEL_NAMES[model]+' '+precision.upper()
        n=len(cell); q=math.ceil(n*.25)
        f=int((cell.public_failure>0).sum()); ef=int((cell.greedy_failure>0).sum())
        counts={}
        for budget in [.1,.25,.5]:
            counts[budget]={}
            for score in ['dsde_4','failure_then_nll','public_then_nll','public_then_dsde']:
                metric=expected_review(cell.error,cell[score],budget)
                counts[budget][score]=metric['reviewed_errors_expected']
                capacities.append(dict(condition=condition,budget=budget,n=n,execution_failures=ef,
                                       public_failures=f,score=score,**metric))
        assert np.isclose(counts[.25]['failure_then_nll'],q)
        assert np.isclose(counts[.25]['public_then_nll'],q)
        same_stage.append([label,ef,f,
                           f"{counts[.25]['failure_then_nll']:.0f}/{counts[.25]['public_then_nll']:.0f}",
                           f"{counts[.5]['failure_then_nll']:.0f}/{counts[.5]['public_then_nll']:.0f}"])
        passing=cell[cell.public_failure==0]
        left=expected_review(passing.error,passing.dsde_4,.25)
        right=expected_review(passing.error,passing.mean_nll,.25)
        extra=(passing.generation_seconds_4-passing.greedy_generation_seconds
               +passing.execution_seconds_4-passing.greedy_execution_seconds)
        baseline=passing.greedy_generation_seconds+passing.greedy_execution_seconds
        summaries.append(dict(condition=condition,n=len(passing),review_slots=left['reviewed'],
                              hidden_errors=int(passing.error.sum()),dsde4_captured=left['reviewed_errors_expected'],
                              nll_captured=right['reviewed_errors_expected'],mean_target_cost_seconds=float(baseline.mean()),
                              mean_extra_sampling_seconds=float(extra.mean())))
        public_rows.append([label,len(passing),left['reviewed'],int(passing.error.sum()),
                            f"{left['reviewed_errors_expected']:.1f}",f"{right['reviewed_errors_expected']:.1f}",
                            f"{extra.mean():.2f}"])
        for actual_q in range(1,n+1):
            # This interior fraction avoids ceil round-off at integer boundaries.
            budget=(actual_q-.25)/n
            left=expected_review(cell.error,cell.public_then_dsde,budget)
            right=expected_review(cell.error,cell.public_then_nll,budget)
            assert left['reviewed']==right['reviewed']==actual_q
            delta=left['reviewed_errors_expected']-right['reviewed_errors_expected']
            if actual_q<=f:
                assert abs(delta)<1e-10
            curves.append(dict(condition=condition,review_slots=actual_q,review_fraction=actual_q/n,
                               public_failures=f,public_failure_fraction=f/n,
                               additional_errors_captured=delta))
    pd.DataFrame(capacities).to_csv(OUT/'same_stage_capacity_metrics.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'public_pass_decision_counts_and_costs.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'guarded_capacity_curve.csv',index=False)
    tex_table(TABLES/'same_stage_capacity_table.tex','lrrrr',
              ['Artifact','Exec. fail.','Public fail.','25\\% E/P','50\\% E/P'],same_stage,
              'Target-check information and error capture in the same validation panel.',
              'tab:same-stage-capacity',
              'Every condition contains 175 targets. Exec. fail. counts targets failing to execute on a public input; Public fail. also counts incorrect successful outputs. '
              'E/P reports errors captured by Exec/NLL and Public/NLL, respectively. The 25\\% queue has 44 positions and the 50\\% queue 88. '
              'At 10\\% (18 positions), both methods capture 18 errors in every condition. Capacity comparisons other than 25\\% are descriptive secondary results.', 'small')
    tex_table(TABLES/'public_pass_decision_table.tex','lrrrrrr',
              ['Artifact','$n$','$q$','Errors','DSDE4','NLL','Extra s'],public_rows,
              'Public-passing targets: actual review positions, expected hidden errors captured and recorded incremental sampling cost.',
              'tab:public-pass-decision',
              'Review is 25\\% of each public-passing subset. DSDE4 and NLL columns are expected hidden-error counts captured, including fractional boundary ties. '
              'Extra s is the subset mean of four alternative-generation requests plus their public executions after target generation and checks. '
              'The cost is a descriptive subset summary of the recorded schedule, excluding comparator, score and ranking overhead. '
              'The paired intervals in Table~\\ref{tab:public-pass} remain the inferential summary.', 'footnotesize')
    configure_plot()
    figure,axes=plt.subplots(2,3,figsize=(7.2,4.8),sharey=True)
    data=pd.DataFrame(curves)
    for ax,condition in zip(axes.flat,[m+'_'+p for p in ['q4','q8'] for m in ['deepseek','qwen25','qwen3']]):
        cell=data[data.condition==condition]
        model,precision=condition.rsplit('_',1)
        boundary=cell.public_failure_fraction.iloc[0]
        ax.axvspan(0,100*boundary,color='#eeeeee',zorder=0)
        ax.axhline(0,color='#888888',linewidth=.7)
        ax.axvline(100*boundary,color='#555555',linestyle=':',linewidth=.9)
        ax.plot(100*cell.review_fraction,cell.additional_errors_captured,color='#0072B2',linewidth=1.2)
        ax.set_title(MODEL_NAMES[model]+' '+precision.upper(),loc='left')
        ax.set_xlim(0,100);ax.grid(alpha=.15);ax.set_axisbelow(True)
    for ax in axes[1]:ax.set_xlabel('Review capacity (%)')
    for ax in axes[:,0]:ax.set_ylabel('Additional errors captured\n(Public/DSDE4 minus Public/NLL)')
    figure.tight_layout()
    require_matplotlib_panel_alignment(figure,axes,'guarded_capacity_curve')
    figure.savefig(FIGURES/'guarded_capacity_curve.pdf')
    figure.savefig(FIGURES/'guarded_capacity_curve.png',dpi=300)
    plt.close(figure)
    print(json.dumps({'same_stage_rows':len(capacities),'public_pass_summaries':summaries,
                      'capacity_curve_points':len(curves),'scope':'Descriptive retained-record additions under Amendment 07.'},indent=2))

if __name__=='__main__':main()
