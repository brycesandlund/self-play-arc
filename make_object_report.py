"""Build XP003's tracked report, figures, and compact data from local run files."""
import collections
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from arcplay.object_lab import CommunicationNet, make_object, representation, scenes, selection_metrics
from arcplay.object_experiments import save
from arcplay.object_followups import score_prediction

ROOT=Path(__file__).resolve().parent
RUNS=ROOT/"runs"
OUT=ROOT/"reports"


def read(run,name):
    return json.loads((RUNS/run/(name+".json")).read_text())


def aggregate(rows,keys,fields):
    groups=collections.defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,group in groups.items():
        entry=dict(zip(keys,key));entry["n"]=len(group)
        for name,path in fields.items():
            values=[]
            for row in group:
                value=row
                for k in path.split("."):value=value[k]
                values.append(value)
            entry[name]=statistics.mean(values)
            entry[name+"_sd"]=statistics.stdev(values) if len(values)>1 else 0
        result.append(entry)
    return result


def table(rows,keys,fields):
    headers=list(keys)+list(fields)
    out=["| "+" | ".join(headers)+" |","|"+"---|"*len(headers)]
    for r in rows:
        out.append("| "+" | ".join([str(r[k]) for k in keys]+[f"{100*r[v]:.1f}%" for v in fields.values()])+" |")
    return "\n".join(out)


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    audit=read("xp003-audit-v2","audit")
    inverse=read("xp003-infer-v2","infer")
    learners=[r for r in read("xp003-learners-v2","learners") if r['task'] not in ('smallest','leftmost')]
    learners+=read("xp003-learners-signfix","learners")
    named=[r for r in read("xp003-named-inverse","named_inverse") if r['task'] not in ('smallest','leftmost')]
    named+=read("xp003-named-signfix","named_inverse")
    refine=read("xp003-refine","refine")+read("xp003-refine-replication","refine")
    soft=read("xp003-softness-v2","softness")
    comp=read("xp003-composition-v2","composition")
    controls=read("xp003-controls","controls")
    gradients=read("xp003-gradients-v2","gradients")
    explore=read("xp003-explore","explore")
    population=read('xp003-population','population')
    population_long=read('xp003-population-long','population')
    parents={r['seed']:r for r in explore if r['method']=='parent'}
    for r in explore:
        if 'selected_objects' in r:
            r['object_choice_change']=statistics.mean(a!=b for a,b in zip(r['selected_objects'],parents[r['seed']]['selected_objects']))
    aggregates={
      'audit':aggregate(audit,['mode','features','temperature'],{'whole_objects':'regimes.repeated.whole_object_fraction','same_color':'regimes.same.whole_object_fraction','irregular':'regimes.irregular.whole_object_fraction','canvas30':'regimes.canvas30.whole_object_fraction'}),
      'inverse':aggregate(inverse,['features','action','hard_targets','support_count'],{'train':'support.exact_grids','query':'query.exact_grids','irregular':'ood.exact_grids','identity':'identity.exact_grids'}),
      'learners':aggregate(learners,['task','support_count','variant'],{'train':'support.exact_grids','query':'query.exact_grids','irregular':'ood.exact_grids'}),
      'named':aggregate(named,['task','support_count'],{'train':'support.exact_grids','query':'query.exact_grids','irregular':'ood.exact_grids'}),
      'controls':aggregate(controls,['task','support_count','variant'],{'train':'support.exact_grids','query':'query.exact_grids','canvas30':'canvas30.exact_grids'}),
      'refine':aggregate(refine,['objective'],{'before':'before.exact_grids','after':'after.exact_grids','whole_before':'before.whole_object_fraction','whole_after':'after.whole_object_fraction','edits_after':'after.boards_with_edits'}),
      'softness':aggregate(soft,['propagation','soft_mode','noise'],{'whole':'whole_object_fraction','agreement':'clean_output_agreement'}),
      'composition':aggregate(comp,['soft_mode','depth'],{'agreement':'hard_intermediate_agreement','confidence':'active_confidence'}),
      'grouping':aggregate(read('xp003-grouping','grouping'),['truth_mode','support_count'],{'train':'support.exact_grids','query':'query.exact_grids','gate':'foreground_gate'}),
      'grouping_untied':aggregate(read('xp003-grouping-untied','grouping'),['truth_mode','support_count'],{'train':'support.exact_grids','query':'query.exact_grids','gate':'foreground_gate'}),
      'population':aggregate(population,['repulsion'],{'before':'before.exact_grids','after':'after.exact_grids','agreement_before':'before.pairwise_output_agreement','agreement_after':'after.pairwise_output_agreement','edits_after':'after.boards_with_edits'}),
      'population_long':aggregate(population_long,['repulsion'],{'before':'before.exact_grids','after':'after.exact_grids','agreement_before':'before.pairwise_output_agreement','agreement_after':'after.pairwise_output_agreement','edits_after':'after.boards_with_edits'}),
      'explore':aggregate([r for r in explore if 'selected_objects' in r],['method'],{'whole':'whole_object_fraction','changed':'object_choice_change','grid_changed':'changed_from_parent'}),
    }
    rng=np.random.default_rng(79001)
    for row in aggregates['refine']:
        rs=[r for r in refine if r['objective']==row['objective']]
        delta=np.array([r['after']['exact_grids']-r['before']['exact_grids'] for r in rs])
        bootstrap=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
        row['paired_delta']=float(delta.mean())
        row['paired_delta_bootstrap95']=np.quantile(bootstrap,[.025,.975]).tolist()
    # Evaluate selection behavior on fresh boards, so palette changes do not count
    # as exploration and near-identical score vectors are not mistaken for novelty.
    x,ids=scenes(128,12,78001)
    rep=representation(x)
    diversity=[]
    for objective in ('initial','entropy','learnability','guarded_learnability'):
        rs=[r for r in refine if r['objective']==('entropy' if objective=='initial' else objective)]
        signatures=[];dominants=[]
        for r in rs:
            score=torch.tensor(r['initial_score' if objective=='initial' else 'final_score'])
            model=make_object(0,temperature=.1)
            with torch.no_grad():model.score.copy_(score)
            signatures.append(selection_metrics(model,rep,ids)['selected_objects'])
            top=int(score.abs().argmax());dominants.append(('area','x','y','width','height')[top]+('+' if score[top]>0 else '-'))
        sig=np.array(signatures)
        pair=(sig[:,None]==sig[None,:]).mean(2)
        diversity.append({'objective':objective,'rules':len(sig),'distinct_signatures':len(np.unique(sig,axis=0)),
                          'pairwise_selection_agreement':float(pair[np.triu_indices(len(sig),1)].mean()),
                          'dominant_feature_counts':dict(collections.Counter(dominants))})
    aggregates['diversity']=diversity
    inventory=[]
    for path in sorted(RUNS.glob('xp003-*')):
        if not path.is_dir():continue
        r={'run_id':path.name}
        if (path/'summary.json').exists():r.update(json.loads((path/'summary.json').read_text()))
        else:r['status']='interrupted'
        for name in ('INVALIDATED','PARTIALLY_INVALIDATED'):
            if (path/(name+'.json')).exists():r[name.lower()]=json.loads((path/(name+'.json')).read_text())
        inventory.append(r)
    summary={'experiment':'XP003','aggregates':aggregates,'inventory':inventory,
             'completed_run_reported_updates':sum(r.get('updates',0) for r in inventory),
             'completed_run_process_seconds':sum(r.get('seconds',0) for r in inventory),
             'valid_inverse_fits':len(inverse),'valid_neural_learner_fits':len(learners)+len(controls),
             'refinement_teachers':64,'neural_learner_parameters':sum(p.numel() for p in CommunicationNet().parameters()),
             'cloud_cost_usd':0}
    save(OUT/'xp003-summary.json',summary)
    save(OUT/'xp003-inverse-details.json',inverse)
    save(OUT/'xp003-refinement-details.json',refine)
    save(OUT/'xp003-learner-details.json',learners+controls)
    save(OUT/'xp003-population-details.json',{'short':population,'long':population_long})
    palette=ListedColormap(['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25','#eeeeee'])
    def board(ax,z,title):
        ax.imshow(z.argmax(0).numpy(),cmap=palette,vmin=0,vmax=10,interpolation='nearest')
        ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    sample,_=scenes(3,12,80001,'same')
    fig,axes=plt.subplots(4,6,figsize=(13,9))
    for row,seed in enumerate((0,1,2,3)):
        teacher=make_object(seed,temperature=.03)
        with torch.no_grad():y=teacher(sample)
        for j in range(3):
            board(axes[row,j*2],sample[j],f'Seed {seed}: input {j+1}')
            board(axes[row,j*2+1],y[j],f'Output {j+1}')
    fig.suptitle('Random shared object rules: disconnected objects all have the same color',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'xp003-random-objects.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    ax=axes[0,0]
    ar=[r for r in aggregates['audit'] if r['temperature']==.03 and r['features']=='geometry']
    ax.bar([r['mode'] for r in ar],[100*r['whole_objects'] for r in ar],color=['#8a8a8a','#2b8cbe','#cf7b51'])
    ax.set(title='Random rules: whole-object selection',ylabel='% boards',ylim=(0,105))
    ax=axes[0,1]
    for feature,action,label in [('geometry','paint','Geometry + recolor'),('all','paint','Geometry/color + recolor'),('geometry','move','Geometry + move')]:
        rs=[r for r in aggregates['inverse'] if r['features']==feature and r['action']==action and r['hard_targets']]
        ax.plot([r['support_count'] for r in rs],[100*r['query'] for r in rs],'o-',label=label)
    ax.set(title='Known-family inverse fit; hard demonstrations',xlabel='Demonstrations',ylabel='Exact query grids (%)',ylim=(0,105));ax.legend(fontsize=8)
    ax=axes[1,0]
    for mode in ('base','prior','sharp2','sharp'):
        rs=[r for r in aggregates['composition'] if r['soft_mode']==mode]
        ax.plot([r['depth'] for r in rs],[100*r['agreement'] for r in rs],'o-',label=mode)
    ax.set(title='Soft composition vs rounded-intermediate reference',xlabel='Composition depth',ylabel='Agreement (%)',ylim=(0,105));ax.legend(fontsize=8)
    ax=axes[1,1]
    tasks=['recolor','shift','ray','box'];xs=np.arange(4)
    for i,variant in enumerate(('local','pool','scan','both')):
        values=[next(r['query'] for r in aggregates['controls'] if r['task']==t and r['support_count']==8 and r['variant']==variant) for t in tasks]
        ax.bar(xs+(i-1.5)*.2,np.array(values)*100,.2,label=variant)
    ax.set(title='Matched neural learners; 8 demonstrations',ylabel='Exact query grids (%)',ylim=(0,105));ax.set_xticks(xs,tasks);ax.legend(fontsize=8)
    fig.suptitle('XP003 synthetic diagnostics — no ARC benchmark data',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'xp003-results.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,label in (('population','160 steps / 4 banks'),('population_long','1,000 steps / 2 banks')):
        rs=aggregates[name]
        axes[0].plot([r['repulsion'] for r in rs],[100*r['after'] for r in rs],'o-',label=label)
        axes[1].plot([r['repulsion'] for r in rs],[100*r['agreement_after'] for r in rs],'o-',label=label)
    axes[0].set(title='Held-out short-adaptation accuracy',ylabel='Exact grids (%)',xlabel='Disagreement penalty weight',ylim=(50,100))
    axes[1].set(title='Similarity of rules in each bank',ylabel='Pairwise output agreement (%)',xlabel='Disagreement penalty weight',ylim=(15,100))
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'xp003-population.png',dpi=160);plt.close(fig)
    gradient_rows=[]
    for mode in ('base','prior','sharp2','sharp'):
        rs=[r for r in gradients if r['soft_mode']==mode and r['noise']==.01 and r['temperature']==.3]
        gradient_rows.append({'mode':mode,'median_mask_gradient':statistics.median(r['mask_input_logit_gradient_norm'] for r in rs)})
    save(OUT/'xp003-gradient-summary.json',gradient_rows)
    inv_table=table(aggregates['inverse'],['features','action','hard_targets','support_count'],{'Support':'train','Query':'query','Irregular query':'irregular'})
    learner_table=table([r for r in aggregates['learners'] if r['support_count']==8],['task','variant'],{'Support':'train','Query':'query','Irregular query':'irregular'})
    control_table=table([r for r in aggregates['controls'] if r['support_count']==8],['task','variant'],{'Support':'train','Query':'query','30×30 query':'canvas30'})
    refine_table=table(aggregates['refine'],['objective'],{'Before':'before','After':'after','Whole-object after':'whole_after'})
    named_table=table(aggregates['named'],['task','support_count'],{'Query':'query','Irregular query':'irregular'})
    report=f'''# XP003 — Object representations, neural task refinement, and CompressARC-inspired layers

**Synthetic research experiment. No ARC examples, external training data, pretrained weights, or cloud compute.**
All reported inference percentages are on generated tasks, not ARC benchmark scores.

## Findings

1. A connectivity prior plus pooled object features gives random weights a useful task family. On the audited separated monochrome boards, geometry-only selectors at temperature 0.03 selected one complete object in all tested cases, including repeated colors and 30×30 canvases. Color masks and endpoint-gated ray scans do not provide equivalent object representations.
2. The family is learnable from a few demonstrations, but remains ambiguous. Three hard examples give about 70% exact new-board agreement for random geometry/recolor rules; sixteen give about 92%. The learner is explicitly supplied the correct architecture family, and these are per-task fits, not the existing in-context solver.
3. Differentiating through a short learner adaptation improves held-out learnability in this restricted family. The improvement replicates on new teacher and board seeds. Diversity measurements below show what remains of the rule bank.
4. Soft composition has a real numerical/representational problem. A differentiable power transform before region construction improves stability. Fourth-power sharpening severely weakens gradients that could change the regions through board logits; square-power sharpening is a better candidate for further study, but also attenuates them.
5. CompressARC-inspired cumulative scans help ray extension strongly. They do not substitute for object representations in selection tasks, and combining every layer is not uniformly better.

![Random object transformations](xp003-random-objects.png)
![Results](xp003-results.png)

## What was built, and what is an explicit prior

Each cell seeds a soft region. Four-neighbor recurrent max propagation is gated by color compatibility. On one-hot inputs this implements bounded same-color connectivity; it is not learned segmentation. All candidate masks remain real-valued. There is no connected-component labeling routine or symbolic rule interpreter in the forward pass. The synthetic instance IDs are used only for evaluation.

Each region supplies log area, centroid x/y, and x/y spread. The optional color-aware version adds nine pooled color channels. Features are standardized across candidate objects on each board. A shared random linear vector scores the candidates, and a softmax selects their weighted mask. The action recolors that mask or moves it one cell in a learned mixture of cardinal directions. Inputs use black and colors 1–3 (or distinct colors in controls); paint uses reserved colors 5–9, so identity is not the easy default.

Every object has multiple seeds. Subtracting log region area from candidate logits compensates for this multiplicity. Without it, larger objects get extra votes even when their scores are identical. This correction is exact for duplicate hard component masks and approximate for soft/partial regions.

Connectivity, foreground/background, coordinate moments, shared selection, and edit structure are **human-specified architectural priors**. Only scorer/action weights are randomly initialized in the main task-family experiment. This establishes a useful narrow neural parameterization, not discovery of an ARC prior from nothing. No edits were made to the original three-network self-play training loop.

## Data and segmentation audits

Default scenes contain four separated objects on 12×12 grids: rectangles, lines, and connected polyominoes. Train, query, irregular-shape, and 30×30 seeds are separate. Scene geometry and palette remain limited. Large canvases retain similarly small objects; this is not a large-object stress test.

The region audit uses 64 boards/regime (four boards for its initial 30×30 check). Color masks exactly cover only 28.5% of objects when colors repeat, and none when all four objects share a color. Local propagation with four or more steps offers at least one exact candidate for every object in these small-shape datasets. Full task transforms use 24 steps, so all seeds within each tested small object agree. These are distinct metrics: best-candidate coverage is weaker than all-seed consistency.

The explicit winding-path control fails to converge from any seed in 24 steps; 144 steps recover the whole path from every seed. Finite propagation depth remains a limitation. Candidate storage is quadratic in cell count, with recurrent computation multiplying that cost, so this all-cell prototype needs a more economical seed mechanism before scaling.

A same-color prior fragments multicolored semantic objects. A foreground-connectivity alternative recovers separated multicolor shapes, but merges touching objects of different colors. In a follow-up 64-board audit, multicolor object coverage is 3.9% with same-color grouping versus 100% with foreground grouping; touching-object coverage is 100% versus 62.9%. Neither definition is universally the right object.

## Few-shot inverse learning

Each random teacher is fixed. An independently initialized student in the same family is fitted to 3, 8, or 16 demonstrations by gradient descent on grid distributions. Two Adam restarts, 400 steps/restart, learning rate 0.03; only support loss chooses the restart. Student temperature anneals 0.4→0.03. Results average 24 teacher seeds, each with 32 held-out rectangle queries and 32 irregular queries. No query enters fitting or restart selection.

Hard-target controls remove soft-label side information by converting the teacher output to one-hot targets **for this diagnostic only**. Soft labels do help, but do not explain all the transfer. Nearly perfect support fits alongside imperfect query fits expose underdetermination and/or inductive-bias mismatch.

{inv_table}

## CompressARC-inspired architectural ablations

These are **conditional board-to-board learners fitted separately per task**, not a CompressARC reproduction. There is no joint-puzzle latent decoder, KL objective, or pretrained shared solver. We borrowed two ideas: pooled/broadcast features and directional cumulative maxima. Only four cardinal directions are implemented here, not their complete eight-direction multitensor system.

Four residual blocks; width 16; identical coordinate channels, identity-logit skip, and **{summary['neural_learner_parameters']:,} parameters** in every variant. Each has the same projections; controls feed repeated local features where the alternatives feed pooled or scanned features. Variants: local only, global/row/column pooling, cardinal scans, or both. Adam 0.003, 800 steps, three seeds, summed CE scaled by grid area. Each task has 32 independent query boards. Named selection tasks reject tied/ambiguous extrema using only teacher masks, not instance labels.

Eight-demo object tasks:

{learner_table}

Specialized object-family inverse fits on the identical task datasets (eight optimizer seeds, two restarts):

{named_table}

Positive controls establish that the scan layers can be useful and the optimization setup can learn simple rules. Ray targets extend color-1 features rightward through background; box targets use intersecting directional propagations. These are architectural diagnostic tasks, not official ARC tasks. Eight-demo results:

{control_table}

Scans help tasks aligned with their primitive, but extra global machinery can overfit on very few examples. These finite experiments do not establish a universal architecture ranking. The full per-seed results include three-demo conditions and variability.

## Differentiable teacher refinement

Optimize only the teacher's five geometry preference weights; painting color 9 and region construction stay fixed. Each outer step generates three fresh support boards and five fresh query boards. A fresh known-family student starts with zero scoring weights and takes five differentiable SGD updates. Optimize teacher query KL after adaptation; compare plain target-entropy minimization and a guarded objective (post-adaptation KL + 0.2 target entropy + 0.0001 mean-square teacher weight).

This is a diagnostic short-adaptation meta-objective, **not** the original retained shared solver, not the difficulty-fringe objective, and not an amortized ARC solver. All direct target gradients and gradients through student updates are retained. A finite-difference test checks the meta-gradient.

120 outer steps per teacher/objective. Initial study: 16 teachers. Replication: 48 new teachers and independent board seeds. Evaluation: 12 unseen adaptation episodes per teacher, each with three support and sixteen query boards. Aggregated over 64 teachers:

{refine_table}

The compact summary includes paired teacher-bootstrap intervals for accuracy changes. Whole-object and edit rates guard against improvements from turning everything into identity or a soft mixture. They do not prove broad task usefulness.

Selection diversity on 128 additional boards (palette differences excluded):

''' + '\n'.join(f"- {r['objective']}: {r['distinct_signatures']}/{r['rules']} distinct selection signatures; mean pairwise agreement {r['pairwise_selection_agreement']:.1%}. Dominant-feature counts: `{r['dominant_feature_counts']}`." for r in diversity) + f'''

Distinct signatures are a weak novelty metric: near-identical rules can remain technically distinct. Pairwise agreement and feature concentration matter too. The improvement is within a five-dimensional rule family, not evidence of indefinitely open-ended exploration.

Pixel-loss weighting is another limitation: errors on a small selected region cost less than errors on a large one. The increase in small-area/extent preferences is consistent with this incentive, although these runs do not isolate it causally. Preserving one complete object edit does not remove all task-simplification incentives.

## Coupled population follow-up

To address that concentration, jointly refine banks of sixteen teachers. Each still has its own student and independent inner updates; vectorization is tested against independent predictions and gradients. Add a differentiable penalty on pairwise cosine similarity of selected masks on the same query inputs. The penalty uses no instance IDs. Mask normalization prevents larger selected areas from being rewarded merely for changing more pixels.

Four initial banks (64 teachers), 160 outer steps per setting, matched seeds across disagreement strengths. Fresh held-out adaptation uses twelve episodes of three support and sixteen query boards. Agreement means two generators produce the same complete output on the same board; lower values indicate more behavioral diversity. All paint color 9, so color changes do not inflate diversity.

''' + table(aggregates['population'],['repulsion'],{'Accuracy before':'before','Accuracy after':'after','Agreement before':'agreement_before','Agreement after':'agreement_after','Boards edited':'edits_after'}) + '''

A longer follow-up uses the first two banks and 1,000 steps per setting. This checks sustained optimization, with fewer independent banks:

''' + table(aggregates['population_long'],['repulsion'],{'Accuracy before':'before','Accuracy after':'after','Agreement before':'agreement_before','Agreement after':'agreement_after','Boards edited':'edits_after'}) + '''

![Population learnability/diversity tradeoff](xp003-population.png)

This is a restricted demonstration of gradient-based task refinement with an explicit diversity tradeoff. It is not evidence that the bank continues discovering new families indefinitely. The 16-teacher banks operate over five geometry preference weights; the representation and action remain fixed.

## Soft inputs and composition

The base construction fails even at 0.1% uniform color contamination: tiny candidate regions can receive pathological feature scores. The prior variant floors the log-area feature at one cell, weights candidate statistics by squared foreground confidence, and adds log foreground confidence to the score before temperature scaling. The sharp2/sharp variants additionally normalize squared/fourth-powered color probabilities **inside region construction**, while preserving the original soft board for rendering. No rounding is used by these transforms.

Soft-input stress results below use 1% uniform contamination and local/max propagation, averaged across 32 rules × 32 boards:

''' + table([r for r in aggregates['softness'] if r['noise']==.01 and r['propagation']=='local'],['soft_mode'],{'Whole-object mask':'whole','Clean-output agreement':'agreement'}) + '''

Repeated soft recolor/move composition is compared with a rounded-intermediate reference. That reference is a stability diagnostic, not an independently correct target. Depth-four agreement:

''' + table([r for r in aggregates['composition'] if r['depth']==4],['soft_mode'],{'Agreement':'agreement','Active confidence':'confidence'}) + '''

Median gradient norm of a random mask probe with respect to input logits (1% contamination, selector temperature 0.3; 24 rules):

''' + '\n'.join(f"- {r['mode']}: `{r['median_mask_gradient']:.3g}`." for r in gradient_rows) + '''

Gradients through the rendered output include a direct copy path and can conceal frozen region decisions; this measurement isolates the mask path. Fourth-power sharpening's excellent stability comes with almost no mask gradient in this test. Squaring is less extreme, but remains a tradeoff. These are probe norms, not a guarantee of useful optimization directions.

## Exploration and grouping choice

Random weight perturbations and interpolation preserve whole-object selection much better than arbitrary pixel networks. The summary reports both changed object choices and changed complete outputs, so a new paint color is not mistaken for a new selection rule. Composition can change topology, so its mask family may change at later stages.

An additional learner used a differentiable scalar gate between same-color and foreground region banks, with a shared scorer and paint head. It fit grid pairs from a mix of multicolored and touching-object scenes. The first run inferred the correct gate in all tested cases, but tied largest components produced soft half-paint targets with unstable argmax agreement. A follow-up rejects tied/ambiguous selections, uses a separate query seed, and fits full grid pairs:

''' + table(aggregates['grouping_untied'],['truth_mode','support_count'],{'Support':'train','Query':'query','Foreground gate':'gate'}) + '''

These results support retaining multiple object interpretations and learning which applies from demonstrations. They do not establish arbitrary object parsing: the gate has only two explicit candidate interpretations, and every task uses the same one across its boards.

## Verification, failed runs, and scope

- Exact-zero candidate masking, duplicate-seed correction, named control semantics, probability normalization, first/second derivatives, a selection finite difference, and a meta-gradient finite difference have dedicated tests.
- The first audit/inverse runs used finite log-epsilon for absent candidates. Large scores could override it and select empty regions. Those runs are explicitly invalidated and excluded; v2 reruns use exact zero-prior masking. The first neural-learner run was interrupted.
- A named-control sign bug made “smallest” behave as largest and “leftmost” as rightmost. Those rows are excluded and replaced by dedicated signfix reruns. The random-rule studies and other named tasks were unaffected. A regression test now checks the intended extrema.
- Max/cummax operations are differentiable almost everywhere; they are not globally smooth. Passing higher derivative tests does not make discrete changes in topology easy to optimize.
- No official ARC transfer test, no existing in-context solver retraining, and no full CompressARC reproduction were performed. No cloud jobs or new charges were incurred.

## Artifacts and reproduction

Code: `arcplay/object_lab.py`, `arcplay/object_experiments.py`, `arcplay/object_followups.py`, and `tests/test_objects.py`. `make_object_report.py` recreates this report from local runs. Each run stores its configuration and exact source snapshots/hashes. `runs/` remains gitignored; tracked compact summaries and per-seed inference/refinement/learner details are in `reports/xp003-*.json`.

Example commands (use the project's Python environment):

```sh
python -m arcplay.object_experiments segmentation --output runs/new-segmentation --seeds 64
python -m arcplay.object_experiments audit --output runs/new-audit --seeds 64
python -m arcplay.object_experiments infer --output runs/new-infer --seeds 24 --steps 400
python -m arcplay.object_experiments learners --output runs/new-learners --seeds 3 --steps 800
python -m arcplay.object_followups refine --output runs/new-refine --seeds 16 --steps 120
python -m arcplay.object_followups controls --output runs/new-controls --seeds 3 --steps 800
python -m arcplay.object_followups softness --output runs/new-softness
python -m arcplay.object_followups composition --output runs/new-composition
python -m arcplay.object_followups gradients --output runs/new-gradients
python -m arcplay.object_followups population --output runs/new-population --seeds 4 --steps 160
python -m arcplay.object_followups grouping --output runs/new-grouping --unambiguous
python -m pytest -q
```

Run inventory and numerical aggregates: [xp003-summary.json](xp003-summary.json). Full local artifacts: `runs/xp003-*`. Completed-run update counts include superseded/invalidated work and exclude partially completed interrupted iterations and inner-loop expansion. Process seconds sum concurrent jobs and are not wall-clock duration.

Architecture inspiration: [CompressARC](https://github.com/iliao2345/CompressARC/tree/83a22218024d46273eb32b769a906340202ffb4d), [paper](https://arxiv.org/abs/2512.06104). Its code/data were not used as training examples. This prototype is an independently implemented, narrower ablation of selected ideas.
'''
    (OUT/'XP003_OBJECTS.md').write_text(report)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('aggregates','inventory')},indent=2))


if __name__=='__main__':main()
