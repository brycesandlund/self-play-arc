"""Rebuild XP005 static scientific figures/tables from saved measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parent
REPORTS=ROOT/'reports'


def read(name):
    p=ROOT/'runs'/name/'summary.json'
    return json.loads(p.read_text()) if p.exists() else None


def compact(value):
    """Retain scores/provenance, leaving per-board diagnostics in raw runs."""
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()
                if k not in ('metadata', 'log', 'parameter_novelty')}
    return value


def generate():
    REPORTS.mkdir(exist_ok=True)
    runs={p.parent.name:json.loads(p.read_text()) for p in sorted((ROOT/'runs').glob('xp005-*/summary.json'))}
    base=[];large=[]
    for name in ('a','b','c','d','e'):
        data=read('xp005-fit-'+name)
        if data:base+=data['records']
    for name in ('a','b','c'):
        data=read('xp005-large-'+name)
        if data:large+=data['records']
    base=sorted(base,key=lambda r:r['index']);large=sorted(large,key=lambda r:r['index'])
    latent=runs.get('xp005-latent')
    novel={r['index'] for r in latent['heldout_zero_context']['records'] if r['novel_parameters']} if latent else set()
    inference=[]
    def inference_row(label, rows, key='scores'):
        subset=[r for r in rows if r['index'] in novel]
        inference.append(dict(method=label,tasks=len(rows),
                              exact=sum(r[key]['exact'] for r in rows)/len(rows),
                              novel_tasks=len(subset),
                              novel_exact=sum(r[key]['exact'] for r in subset)/len(subset) if subset else None))
    icl=runs.get('xp005-icl');adapt=runs.get('xp005-adapt')
    if icl:
        inference_row('In context, 4 demos',icl['evaluation']['records'])
        inference_row('Wrong task demos, 4',icl['wrong_support']['records'])
    if adapt:
        inference_row('In context, 8 demos',adapt['records'],'before')
        inference_row('Full-weight adaptation, 8 demos',adapt['records'],'after')
    if latent:
        for demos,fit in latent['fitted'].items():
            inference_row(f'Continuous task-vector fitting, {demos} demos',fit['evaluation']['records'])
    measurements=dict(runs=compact(runs),inference=inference)
    (REPORTS/'xp005-measurements.json').write_text(json.dumps(measurements,indent=2,allow_nan=False)+'\n')
    lines=['| Inference | All 58 task variants | 40 new parameterizations |','|---|---:|---:|']
    for r in inference:
        lines.append(f"| {r['method']} | {r['exact']:.1%} | {r['novel_exact']:.1%} |")
    (REPORTS/'xp005-inference-table.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    if base:
        groups=[('Catalogue primitives',[r for r in base if r['index']<21]),
                ('Composed transformations',[r for r in base if 21<=r['index']<29]),
                ('Earlier neural teachers',[r for r in base if r['index']>=29])]
        fig,axes=plt.subplots(1,3,figsize=(17,10),gridspec_kw={'width_ratios':[1.1,1,1.2]})
        for ax,(title,rows) in zip(axes,groups):
            yy=np.arange(len(rows));names=[r['name'].replace('_v0','').replace('neural_','').replace('_',' ')+(' *' if r['index'] in (0,30,38,43,44) else '') for r in rows]
            ax.barh(yy-.18,[100*r['scores']['train']['exact'] for r in rows],height=.35,color='#9dbcd4',label='Training boards')
            ax.barh(yy+.18,[100*r['scores']['test']['exact'] for r in rows],height=.35,color='#24577a',label='Held-out boards')
            ax.set_yticks(yy,names);ax.invert_yaxis();ax.set_xlim(0,104);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Exact output grids (%)');ax.set_title(title,pad=12);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        fig.suptitle('One neural architecture, independently fitted computational weights',fontsize=17,y=.98)
        fig.text(.5,.94,'256 training + 32 validation boards per task; 64 test boards; all 12×12. * Identity/constant controls.',ha='center',fontsize=10)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
        fig.tight_layout(rect=(0,.045,1,.91));fig.savefig(REPORTS/'xp005-fits.png',dpi=150);plt.close(fig)
    if large:
        fig,ax=plt.subplots(figsize=(10,11))
        yy=np.arange(len(large));names=[r['name'].replace('_',' ') for r in large]
        ax.barh(yy-.18,[100*r['before']['exact'] for r in large],height=.35,color='#a7b7c6',label='Original fitted network')
        ax.barh(yy+.18,[100*r['scores']['test']['exact'] for r in large],height=.35,color='#16776b',label='After 1,024 additional training boards')
        ax.set_yticks(yy,names);ax.invert_yaxis();ax.set_xlim(0,104);ax.set_xlabel('Exact output grids (%)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_title('More examples: paired evaluation on fresh query boards\n126–128 fresh queries per task; same architecture',fontsize=13,pad=16)
        ax.legend(loc='lower right',frameon=True,fontsize=9)
        fig.tight_layout();fig.savefig(REPORTS/'xp005-more-data.png',dpi=150);plt.close(fig)
    lines=['| Transformation | Train exact | Test exact | Test foreground | Test extent |','|---|---:|---:|---:|---:|']
    for r in base:
        s=r['scores'];fg=s['test']['foreground']
        lines.append(f"| {r['name']} | {s['train']['exact']:.1%} | {s['test']['exact']:.1%} | {fg:.1%} | {s['test']['extent']:.1%} |" if fg is not None else f"| {r['name']} | {s['train']['exact']:.1%} | {s['test']['exact']:.1%} | — | {s['test']['extent']:.1%} |")
    (REPORTS/'xp005-family-table.md').write_text('\n'.join(lines)+'\n')
    if large:
        lines=['| Transformation | Before | After continuation | Fresh test boards |','|---|---:|---:|---:|']
        for r in large:lines.append(f"| {r['name']} | {r['before']['exact']:.1%} | {r['scores']['test']['exact']:.1%} | {r['scores']['test']['examples']} |")
        (REPORTS/'xp005-continuation-table.md').write_text('\n'.join(lines)+'\n')
    chosen=[1,8,18,19,20,24]
    available=[]
    for index in chosen:
        name=('a' if index<10 else 'b' if index<20 else 'c')
        p=ROOT/f'runs/xp005-large-{name}/examples_{index:03d}.pt'
        if not p.exists():
            name='a' if index<10 else 'b' if index<21 else 'c'
            p=ROOT/f'runs/xp005-fit-{name}/examples_{index:03d}.pt'
        if p.exists():available.append((index,p))
    if available:
        cmap=ListedColormap(['#000000','#0074d9','#ff4136','#2ecc40','#ffdc00','#aaaaaa','#f012be','#ff851b','#7fdbff','#870c25','#f0f0f0'])
        fig,axes=plt.subplots(len(available),3,figsize=(8,2.45*len(available)),squeeze=False)
        name_by_id={r['index']:r['name'].replace('_v0','').replace('_',' ') for r in base}
        for row,(index,path) in enumerate(available):
            d=torch.load(path,weights_only=False);xs,ys=d['x'].argmax(1),d['y'].argmax(1);ps=d['pred']
            same=(ps==ys).flatten(1).all(1)
            # Alternate success/failure when those examples are present.
            candidates=torch.where(same if row%2==0 and index!=20 else ~same)[0];i=int(candidates[0]) if len(candidates) else 0
            for col,labels in enumerate((xs[i],ys[i],ps[i])):
                ax=axes[row,col];ax.imshow(labels,cmap=cmap,vmin=0,vmax=10,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
                if row==0:ax.set_title(['Query input','Target','Neural prediction'][col])
            axes[row,0].set_ylabel(name_by_id[index],fontsize=10)
            axes[row,2].set_xlabel('Exact' if same[i] else 'Incorrect',color='#16776b' if same[i] else '#a02c2c')
        fig.suptitle('Examples of learned execution and remaining failures',fontsize=14,y=.997)
        fig.tight_layout();fig.savefig(REPORTS/'xp005-examples.png',dpi=150);plt.close(fig)
    print(json.dumps({'runs':len(runs),'independent_tasks':len(base),'continued_tasks':len(large)}))


if __name__=='__main__':generate()
