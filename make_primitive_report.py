"""Build XP002 aggregate tables and a self-contained, inspectable grid gallery."""
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from arcplay.primitive_lab import boards, comparison, make_transform

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/"reports"
PALETTE=["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25","#dddddd"]


def load_run(name,file):
    path=ROOT/"runs"/name/file
    return json.loads(path.read_text()) if path.exists() else []


def mean(rows,key):
    def value(row):
        for k in key.split("."): row=row[k]
        return float(row)
    return float(np.mean([value(r) for r in rows])) if rows else None


def table(headers,rows):
    return "| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in rows)


def percent(v): return f"{100*v:.1f}%"


def build():
    REPORT.mkdir(exist_ok=True)
    audit=load_run("xp002-audit","audit.json")+load_run("xp002-audit-object","audit.json")
    large=load_run("xp002-audit","canvas30.json")+load_run("xp002-audit-object","canvas30.json")
    explore=load_run("xp002-explore","explore.json")+load_run("xp002-explore-object","explore.json")
    inference=load_run("xp002-infer","infer.json")+load_run("xp002-infer-object","infer.json")
    calibrated=load_run("xp002-infer-calibrated","infer.json")
    coverage=load_run("xp002-coverage","coverage.json")
    refine=load_run("xp002-refine","refine.json")+load_run("xp002-refine-object","refine.json")
    inputs=load_run("xp002-inputs","optimize_inputs.json")+load_run("xp002-inputs-object","optimize_inputs.json")
    gradients=load_run("xp002-gradients","gradients.json")
    families=list(dict.fromkeys(r["family"] for r in audit))
    summaries={}
    for p in sorted((ROOT/"runs").glob("xp002-*/summary.json")):
        summaries[p.parent.name]=json.loads(p.read_text())

    family_summary=[]
    identity_baselines={}
    query_boards=boards(32,seed=9302)
    torch.set_num_threads(2)
    with torch.no_grad():
        for family in families:
            identity_baselines[family]=float(np.mean([comparison(query_boards,make_transform(family,seed,.07)(query_boards))["exact_grids"] for seed in range(12)]))
    (REPORT/"xp002-inverse-identity-baselines.json").write_text(json.dumps(identity_baselines,indent=2)+"\n")
    for f in families:
        cold=[r for r in audit if r["family"]==f and r["temperature"]==.07]
        infer=[r for r in inference if r["family"]==f]
        family_summary.append({"family":f,"parameters":cold[0]["parameters"],
            "screen_pass_fraction":mean(cold,"screen_pass"),"active_confidence":mean(cold,"active_confidence"),
            "changed_cells":mean(cold,"changed_cells"),"nonuniform_boards":mean(cold,"nonuniform_boards"),
            "infer_train_kl":mean(infer,"support.kl"),"infer_query_exact":mean(infer,"query.exact_grids"),
            "infer_query_kl":mean(infer,"query.kl"),"infer_iid_exact":mean(infer,"iid_query.exact_grids"),
            "identity_query_exact":identity_baselines[f],
            "infer_canvas30_exact":mean(infer,"canvas30.exact_grids")})
    method_summary=[]
    for method in dict.fromkeys(r["method"] for r in explore):
        subset=[r for r in explore if r["method"]==method and r["parent_screen_pass"]]
        method_summary.append({"method":method,"parent_count":len(subset),
                               "screen_pass_fraction":mean(subset,"screen_pass"),
                               "distance":mean(subset,"distance_from_parent")})
    refined_summary=[]
    for f in families:
        for obj in ("sharpness","learnability","guarded_learnability"):
            s=[r for r in refine if r["family"]==f and r["objective"]==obj]
            if not s:continue
            refined_summary.append({"family":f,"objective":obj,"seeds":len(s),
                "before_kl":mean(s,"learner.before.kl"),"after_kl":mean(s,"learner.after.kl"),
                "initial_pass":mean(s,"initial.screen_pass"),"final_pass":mean(s,"final.screen_pass"),
                "final_nonuniform":mean(s,"final.nonuniform_boards"),
                "final_confidence":mean(s,"final.active_confidence"),"final_change":mean(s,"final.changed_cells")})
    solver_results={}
    for p in sorted((ROOT/"runs").glob("xp002-solver-*/solver.json")):
        solver_results[p.parent.name]=json.loads(p.read_text())
    result={"families":family_summary,"exploration":method_summary,"refinement":refined_summary,
            "coverage":coverage,"calibrated_inference":calibrated,"solver":solver_results,
            "run_inventory":summaries,"counts":{"random_tasks":len(audit),"canvas30_tasks":len(large),
            "exploration_candidates":len(explore),"base_inference_tasks":len(inference),
            "refinement_runs":len(refine),"input_optimization_runs":len(inputs)}}
    (REPORT/"xp002-summary.json").write_text(json.dumps(result,indent=2)+"\n")

    fig,ax=plt.subplots(figsize=(11,6))
    matrix=np.array([[mean([r for r in audit if r["family"]==f and r["temperature"]==t],"screen_pass") for t in (1.,.3,.07)] for f in families])
    ax.imshow(matrix,vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
    ax.set_xticks(range(3),["Temperature 1.0","Temperature 0.3","Temperature 0.07"])
    ax.set_yticks(range(len(families)),families)
    for i in range(len(families)):
        for j in range(3):ax.text(j,i,percent(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]>.6 else "black",fontsize=8)
    ax.set_title("Random initialization: fraction passing a structural screen\n24 seeds per cell; screening is not ARC performance")
    fig.tight_layout();fig.savefig(REPORT/"xp002-screening.png",dpi=160);plt.close(fig)

    samples={}
    for run in ("xp002-audit","xp002-audit-object"):
        p=ROOT/"runs"/run/"examples.pt"
        if p.exists():samples.update(torch.load(p,weights_only=True))
    groups={"routing":["color_bg","transport","separable","dihedral","axis_affine","mirror_copy"],
            "conditional":["gated_transport","dilate","erode","flood","relative_motion","count_gate"],
            "baselines":["conv","residual_conv","color","affine","content_attention"]}
    for name,group in groups.items():
        fig,axs=plt.subplots(len(group),6,figsize=(12,len(group)*1.8),squeeze=False)
        for row,family in enumerate(group):
            x,y=samples[f"{family}/0.07/0"]
            for b in range(3):
                for j,values in enumerate((x,y)):
                    ax=axs[row,2*b+j]
                    ax.imshow(values[b].argmax(0),cmap=ListedColormap(PALETTE),vmin=0,vmax=10,interpolation="nearest")
                    ax.set_xticks([]);ax.set_yticks([])
                    ax.set_title("input" if j==0 else f"output · confidence {values[b].max(0).values.mean():.2f}",fontsize=8)
            axs[row,0].set_ylabel(family,fontsize=9)
        fig.suptitle("Random neural transformations · seed 0 · temperature 0.07\nArgmax display only; outputs and training remain soft",fontsize=12)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(REPORT/f"xp002-{name}.png",dpi=160);plt.close(fig)

    # A before/after figure makes collapse visible rather than hiding it in a score.
    rsamples={}
    for run in ("xp002-refine","xp002-refine-object"):
        rsamples.update(torch.load(ROOT/"runs"/run/"examples.pt",weights_only=True))
    selected=[("conv","learnability"),("color_bg","learnability"),("transport","learnability"),
              ("axis_affine","guarded_learnability"),("color_bg","sharpness"),("flood","sharpness")]
    fig,axs=plt.subplots(len(selected),3,figsize=(8,12))
    for row,(family,obj) in enumerate(selected):
        triples=rsamples[f"{family}/0/{obj}"]
        for col,(title,values) in enumerate(zip(("input","initial output","refined output"),triples)):
            ax=axs[row,col];ax.imshow(values[0].argmax(0),cmap=ListedColormap(PALETTE),vmin=0,vmax=10)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f"{title} · confidence {values[0].max(0).values.mean():.2f}",fontsize=8)
        axs[row,0].set_ylabel(f"{family}\n{obj}",fontsize=9)
    fig.suptitle("Gradient refinement examples · seed 0\nA better learning loss does not ensure a useful task")
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(REPORT/"xp002-refinement.png",dpi=160);plt.close(fig)

    # Interactive, completely local gallery, with no third-party assets or JS.
    gallery=[]
    for key,(x,y) in samples.items():
        f,t,s=key.split("/")
        gallery.append({"family":f,"temperature":t,"seed":s,"input":x.argmax(1).tolist(),
                        "output":y.argmax(1).tolist(),"confidence":y.max(1).values.mean((-2,-1)).tolist()})
    for key,values in rsamples.items():
        f,s,obj=key.split("/")
        x,before,after=values
        gallery.append({"family":"refined:"+f,"temperature":obj,"seed":s,"input":x.argmax(1).tolist(),
                        "output":after.argmax(1).tolist(),"before":before.argmax(1).tolist(),
                        "confidence":after.max(1).values.mean((-2,-1)).tolist()})
    gallery_doc='''<!doctype html><meta charset="utf-8"><title>XP002 neural transformation gallery</title>
<style>body{font:16px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;background:#f6f7fa;color:#182238}h1{font-size:28px}select{padding:8px;margin:10px 12px 16px 0}#grids{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}.pair{background:white;padding:18px;border-radius:12px;display:flex;gap:20px;flex-wrap:wrap}canvas{width:150px;height:150px;image-rendering:pixelated}small{display:block;margin:8px 0}.note{line-height:1.5;color:#435068}button{padding:7px}</style>
<h1>Random neural transformations & gradient refinement</h1><p class="note">No ARC examples or pretrained weights. The architecture encodes the named operations. These are argmax displays of soft targets; confidence is the mean maximum color probability, including background. A crisp-looking image can hide an uncertain distribution. Initializations shown are seeds 0–2, without selection; refined examples cover seeds 0–7.</p>
<label>Family <select id="family"></select></label><label>Temperature / objective <select id="temperature"></select></label><label>Seed <select id="seed"></select></label><div id="grids"></div>
<script>const data=__DATA__,colors=__COLORS__;const f=document.querySelector('#family'),t=document.querySelector('#temperature'),s=document.querySelector('#seed');
function options(e,values){e.replaceChildren(...[...new Set(values)].map(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;return o}))}
function draw(board,title){const box=document.createElement('div'),c=document.createElement('canvas'),label=document.createElement('small');c.width=board[0].length*20;c.height=board.length*20;const ctx=c.getContext('2d');board.forEach((r,y)=>r.forEach((v,x)=>{ctx.fillStyle=colors[v];ctx.fillRect(x*20,y*20,19,19)}));label.textContent=title;box.append(c,label);return box}
function render(){const d=data.find(d=>d.family===f.value&&d.temperature===t.value&&d.seed===s.value),root=document.querySelector('#grids');root.replaceChildren();d.input.forEach((b,i)=>{const p=document.createElement('div');p.className='pair';p.append(draw(b,'Input'));if(d.before)p.append(draw(d.before[i],'Before refinement'));p.append(draw(d.output[i],'Output · confidence '+d.confidence[i].toFixed(3)));root.append(p)})}
function seeds(){options(s,data.filter(d=>d.family===f.value&&d.temperature===t.value).map(d=>d.seed));render()}
function temperatures(){options(t,data.filter(d=>d.family===f.value).map(d=>d.temperature));if([...t.options].some(o=>o.value==='0.07'))t.value='0.07';seeds()}
options(f,data.map(d=>d.family));f.value='color_bg';f.onchange=temperatures;t.onchange=seeds;s.onchange=render;temperatures();</script>'''
    (REPORT/"xp002-gallery.html").write_text(gallery_doc.replace("__DATA__",json.dumps(gallery)).replace("__COLORS__",json.dumps(PALETTE)))

    screening=table(["Family","Params","Sharp screen pass","Active confidence","Changed cells","3-demo exact queries","Identity exact"],
        [[r['family'],r['parameters'],percent(r['screen_pass_fraction']),percent(r['active_confidence']),percent(r['changed_cells']),percent(r['infer_query_exact']),percent(r['identity_query_exact'])] for r in family_summary])
    exploration=table(["Proposal","Passing parents","Child screen pass","Mean output distance"],
        [[r['method'],r['parent_count'],percent(r['screen_pass_fraction']),f"{r['distance']:.3f}"] for r in method_summary])
    refinement=table(["Family","Objective","Fresh learner KL: before → after","Screen: before → after"],
        [[r['family'],r['objective'],f"{r['before_kl']:.3f} → {r['after_kl']:.3f}",f"{percent(r['initial_pass'])} → {percent(r['final_pass'])}"] for r in refined_summary])
    coverage_rows=[]
    for f in dict.fromkeys(r['family'] for r in coverage):
        base=[r for r in inference if r['family']==f]
        a=[r for r in coverage if r['family']==f and r['support_mode']=='objects']
        b=[r for r in coverage if r['family']==f and r['support_mode']=='iid']
        coverage_rows.append([f,percent(mean(base,'query.exact_grids')),percent(mean(a,'query.exact_grids')),percent(mean(b,'query.exact_grids'))])
    solver_rows=[]
    for name,run in solver_results.items():
        for f,v in run['final'].items():
            solver_rows.append([name.removeprefix('xp002-solver-'),f,percent(v['matched']['exact_grids']),percent(v['matched']['balanced_cell_accuracy']),percent(v['wrong_demonstrations']['exact_grids']),percent(v['identity']['exact_grids'])])
    calibration=[]
    for f in dict.fromkeys(r['family'] for r in calibrated):
        a=[r for r in inference if r['family']==f];b=[r for r in calibrated if r['family']==f]
        calibration.append([f,f"{mean(a,'support.kl'):.3f} → {mean(b,'support.kl'):.3f}",f"{percent(mean(a,'query.exact_grids'))} → {percent(mean(b,'query.exact_grids'))}"])
    gradient_rows=[]
    for f in families:
        rates=[]
        for t in (1.,.3,.07):
            a=[r for r in gradients if r['family']==f and r['temperature']==t]
            rates.append(percent(mean(a,'near_zero')))
        gradient_rows.append([f,*rates])
    input_rows=[]
    for f in families:
        a=[r for r in inputs if r['family']==f]
        if a:input_rows.append([f,f"{percent(mean(a,'initial.changed_cells'))} → {percent(mean(a,'final.changed_cells'))}",percent(mean(a,'input_confidence')),f"{percent(mean(a,'initial.screen_pass'))} → {percent(mean(a,'final.screen_pass'))}"])
    inventory=table(["Run","Elapsed seconds","Work"],[[name,f"{r['seconds']:.1f}",json.dumps({k:v for k,v in r.items() if k in ('tasks','updates','outer_updates','candidates','full_size_tasks')})] for name,r in summaries.items()])
    body=f'''# XP002 — Random neural transformation primitives

Run date: 2026-10-12. No external data, pretrained weights, or official ARC evaluation.
All experiments ran locally on CPU; **additional cloud spending: $0**.

## Findings

Random initialization can produce coherent tasks when the neural architecture gives its weights shared spatial or color meanings. This is strong evidence for structured neural task families, not for arbitrary random neural networks as ARC generators. The tested operations are architectural priors; sampling a rotation from a dihedral module does not discover rotation.

Local exploration mostly preserves those families. Gradient refinement is feasible and numerically checked, but optimizing short-horizon learnability frequently rewards collapse or softening. The existing in-context solver also struggles with several tasks that a matched-family inverse learner solves readily.

Inspect the [interactive gallery](xp002-gallery.html), [aggregate JSON](xp002-summary.json), and [initialization heatmap](xp002-screening.png). The gallery includes uncertain outputs and failures as well as useful examples. It shows all first-three-seed examples collected, not just selected successes.

## Scope and protocols

- **{len(families)} architectures, {len(audit):,} random initializations**, 24 seeds × three temperatures, plus {len(large)} full 30×30 audits. Random weights only; no hypernetwork. Each task uses one fixed transformation across its instances.
- **{len(explore):,} exploration candidates**, including restart, perturbation, related/unrelated weight interpolation, two- and three-stage composition, and gated composition. This count includes unchanged-parent controls.
- **{len(inference)} base inverse-learning tasks**, 12 teacher seeds/family; 3 fixed demonstrations, 32 independent object-board queries, 16 IID queries, and 8 full-size queries. Two fresh student initializations, 300 Adam steps each. Best restart is selected using demonstration loss only.
- {len(calibrated)} optimizer follow-ups, {len(coverage)} coverage controls, **{len(refine)} transformation-refinement runs**, and **{len(inputs)} input-optimization runs**. See the inventory for smoke runs and all training steps.
- Neural solver controls use the existing 43,115-parameter solver, three demos, 8×8 grids, and 10,000 updates/run. Test transformation seeds are disjoint from training seeds. These are synthetic generalization checks, not ARC scores.
- Inputs: synthetic rectangles, lines and dots on black backgrounds; separate IID controls. Ten real colors plus X, which is unused on these full square canvases. Targets stay soft; argmax is only for diagnostics and pictures. Input-board optimization also stays soft.
- Deliberate omissions: variable output sizes, rectangular padding, connected-component segmentation, object identity across disconnected same-color regions, arbitrary counting, multi-step symbolic rule discovery, and official ARC transfer.

The structural screen requires active-cell confidence > .85, >5% changed cells, cross-input output variation > .02, and nonuniform outputs on >75% of boards. **It is only a heuristic.** Sparse useful edits can fail the 5% rule, and a destructive geometric warp can pass. Active cells are the union of input/output argmax foreground. Mean confidence alone is inflated by background. Exact-grid accuracy below compares argmax outputs; soft KL is recorded separately.

## Initializations and three-demo inverse fitting

{screening}

The three-demo column gives the learner the correct architecture family. It does **not** give it the teacher weights or query labels. This measures parameter identifiability and optimization within a family, not unknown-family inference or the existing solver's capability. The last column copies the query input unchanged. Low-temperature shared routing is particularly successful. Plain convolutions have diffuse random outputs; a strong residual identity bias produces mostly identity.

**Sparse conditional rules need the identity control.** Selected-color transport's 96.1% exceeds its 54.7% identity baseline. But relative movement's 54.2% is below its 64.6% identity baseline, and count-gate's 95.1% is below 97.1% identity. Those two inference results are failures; visually interesting occasional outputs do not establish a useful task distribution. This input prior rarely activates the count threshold. Morphology/propagation also have substantial identity fractions, so their raw exact accuracies overstate the difficulty solved.

![Shared spatial and color operators](xp002-routing.png)

![Conditional and propagation operators](xp002-conditional.png)

![Unstructured baselines](xp002-baselines.png)

Architectural details: color maps are column-softmax matrices; transport is a shared convex mixture of 25 spatial offsets; separable routing factors row/column offsets; dihedral routing mixes eight fixed symmetries. Affine models use Gaussian coordinate attention; axis-affine starts near signed diagonal matrices, but all entries are trainable. Content attention pools raw color values. Gated transport moves a soft color-selected mask. Morphology uses learned neighbor gates with soft probabilistic union/intersection. Flood uses four recurrent local expansion steps through background, with other colors acting as barriers. Relative motion translates a selected color mask toward another color's centroid; it is not general object segmentation. Count-gate thresholds selected-color **area fraction**, not an abstract size-independent count. Mirror-copy places transformed foreground into background only. Those semantics are supplied by the architecture.

### Optimization and demonstration coverage controls

The initial uniform learning rate (.06) was unsuitable for convolutional inverse learners and some coordinate models. A follow-up used .003, 600 steps, and four restarts; all three changes are confounded, so this only establishes whether a stronger fitting attempt helps.

{table(['Family','Demonstration KL: base → follow-up','Query exact: base → follow-up'],calibration)}

The three object demonstrations contain colors 0,1,2,4,5,6,7,9; **3 and 8 are absent**. Their mapping cannot be recovered from those demonstrations. Eight object demos cover all real colors. Three IID demos also cover colors but can be poor evidence for sparse propagation behavior.

{table(['Family','3 object demos','8 object demos','3 IID demos'],coverage_rows)}

This is evidence that the board distribution controls identifiability, independently of the transformation being simple.

## Exploration

These rates are conditioned on the unchanged parent passing the screen on an independent probe set. Distance is mean per-cell total variation between the parent's and child's soft outputs. High preservation with near-zero distance is not broad exploration.

{exploration}

Small perturbations and related interpolation preserve structure more reliably. Deeper composition can erase information. Gated composition often lowers crispness because it mixes two different color distributions; failing this screen is not evidence that its gradients cannot be useful. Semantic parameter coordinates make interpolation more meaningful than arbitrary hidden-unit averaging.

## Differentiable refinement

Start from temperature .3 and optimize the transformation weights for 100 Adam steps (.03 learning rate), eight seeds per family. Each outer step uses new procedural support/query boards. The diagnostic student is a small two-layer convolutional network trained on three examples with five differentiable SGD steps (per-cell CE, learning rate .8); it is reset to a new initialization each outer step. Final measurements use 24 held-out queries and three fresh student seeds. This diagnostic is **not the original progress self-play objective**, and is not a trained universal solver.

Objectives: (1) minimize target entropy; (2) minimize the fresh learner's post-adaptation soft KL; (3) that KL plus .2×entropy and squared hinge penalties for low output response, low spatial variance, and too little change from input. See source for the exact constants. These hand-designed guards are tested hypotheses, not guarantees of meaningfulness. No accuracy claim is made from training loss alone.

{refinement}

![Before and after gradient refinement](xp002-refinement.png)

Read KL reductions alongside confidence and diversity: making the target constant or broad can make it easier to match. Sharpening is effective in some structured families, but cannot turn an unconstrained network into a useful family by itself. The guards tested here are insufficient to reliably prevent degeneracy. Positive affine results also need caution: easy geometric distortion is not necessarily an ARC-like semantic rule.

### Gradient sensitivity

Fraction of initializations with parameter-gradient norm < 1e-6 under a fixed random linear output probe (not the learning objective). This is a local sensitivity diagnostic, not a guarantee of useful gradients.

{table(['Family','T=1','T=.3','T=.07'],gradient_rows)}

## Optimizing input boards

Hold the random transformation fixed. Optimize eight soft input-board logits for 100 steps, rewarding output/input change and cross-board output variation, penalizing input entropy and changes in foreground occupancy. Initialization comes from structured boards. This is input search, not learned task generation, and can create irregular patterns. It does not guarantee demonstrations that identify the rule.

{table(['Family','Changed cells: before → after','Final input confidence','Screen: before → after'],input_rows)}

## Existing solver controls

The solver receives only demonstrations and the query. Single-family runs train on 32 random transformation seeds; mixed runs train on 128 (32 each for color_bg, transport, dihedral, flood). Each family is evaluated on 16 unseen transformation seeds × four queries. Mismatched-demo controls replace whole coherent demonstration sets with another task's demonstrations. Balanced cell accuracy averages over target colors present in the evaluation set, reducing background dominance.

{table(['Run','Test family','Exact queries','Balanced cells','Wrong-demo exact','Identity exact'],solver_rows)}

The oracle-family inverse learner and this in-context solver are different inference mechanisms. A gap between them is evidence of a limitation in this solver/training setup, not proof that the tasks are intrinsically unlearnable. Single-family controls help separate mixed-task interference from basic inference difficulty. The 3-demo color coverage ceiling and soft-label uncertainty still apply.

## Reproduction and run inventory

Run from the repository with its existing Python environment:

```sh
python -m pytest -q
python -m arcplay.primitive_experiments audit --output runs/new-audit
python -m arcplay.primitive_experiments explore --seeds 24 --output runs/new-explore
python -m arcplay.primitive_experiments infer --seeds 12 --steps 300 --output runs/new-infer
python -m arcplay.primitive_experiments refine --seeds 8 --steps 100 --output runs/new-refine
python -m arcplay.primitive_followups coverage --output runs/new-coverage
python -m arcplay.primitive_followups optimize_inputs --output runs/new-inputs
python -m arcplay.primitive_followups gradients --output runs/new-gradients
python -m arcplay.primitive_followups solver --regime mixed --steps 10000 --output runs/new-solver
python make_primitive_report.py
```

The original sweep had 14 families; relative_motion, count_gate and mirror_copy were follow-ups. Current default commands include all 17. Run folders contain configurations, raw JSON and examples. Early core-experiment source snapshots were captured before adding the three later families; later launches record source snapshots and hashes automatically. Fixed seeds reproduce inputs and initialization within this environment. Fitting uses soft targets and selects restarts on support only. Early smoke runs and the high-learning-rate failure are retained. `runs/` is Git-ignored; report artifacts outside it provide aggregate results and the visual gallery.

Verification: **44 tests passed**, including probability conservation and first/second derivatives for all 17 architectures, routing boundaries, rejection of constant/identity tasks by the diagnostic screen, and a finite-difference check through student updates. See [test output](xp002-tests.txt). All 1,224 initialization outputs passed the probability-validity checks. Static figures were visually inspected. Automatic browser review blocked the optional HTML gallery preview because the browser does not permit local `file:` URLs; the self-contained gallery is saved but was not browser-tested.

{inventory}

Additional cloud spend is zero; no Modal workers or schedules were launched. Elapsed times above are per-process wall times and overlap across concurrently running local experiments. This is a broad, finite architecture sweep, not an exhaustion of all possible neural inductive biases.
'''
    (REPORT/"XP002_PRIMITIVES.md").write_text(body)
    print(json.dumps(result['counts'],indent=2))


if __name__=="__main__":build()
