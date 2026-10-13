"""Regenerate XP004 figures, tracked measurements, and the run inventory."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parent;RUNS=ROOT/'runs';REPORTS=ROOT/'reports'
REPORTS.mkdir(exist_ok=True)

def read(name):return json.loads((RUNS/name/'summary.json').read_text())
def pct(x):return f'{100*x:.1f}%'

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7f8fb','axes.facecolor':'#f7f8fb'})
    cm=ListedColormap(['#15181e','#2676d9','#ef493f','#36ba78','#f5d749','#92979e','#ca49ac','#f39b35','#75d1db','#842438','#e1e5ee'])
    examples=torch.load(RUNS/'xp004-audit/examples.pt',weights_only=False)+torch.load(RUNS/'xp004-advanced/examples.pt',weights_only=False)
    wanted=['recolor','copy','bbox','outline','ray','crop','rotate','scale','repeat_h','count','link','align','sort_pack','shape_match','fill_holes']
    mapping={name:(x,y,t) for name,x,y,t in examples}
    from arcplay.catalogue import Rule,OPS,encode_board,execute
    ax,_,_=mapping['align']
    ay=execute(encode_board(ax[None]),Rule(OPS.index('align'),src=0,dst=2).code()[None])[0]
    mapping['align']=(ax,ay,ay.argmax(0))
    fig,axes=plt.subplots(5,6,figsize=(15,12.2),layout='constrained')
    for j,name in enumerate(wanted):
        x,y,target=mapping[name]
        for k,a in enumerate((x.argmax(0),y.argmax(0))):
            ax=axes[j//3,2*(j%3)+k];ax.imshow(a,vmin=0,vmax=10,cmap=cm,interpolation='nearest')
            ax.set_xticks(np.arange(a.shape[1]+1)-.5,minor=True);ax.set_yticks(np.arange(a.shape[0]+1)-.5,minor=True)
            ax.grid(which='minor',color='white',alpha=.16,linewidth=.4);ax.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
            ax.set_title((name.replace('_',' ').capitalize()+' · input') if k==0 else 'Output',fontsize=10,pad=6)
            for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle('XP004 · Architectural catalogue\nProcedural parameters; neural tensor execution. Light gray = outside output grid.',fontsize=15)
    fig.savefig(REPORTS/'xp004-catalogue.png',dpi=160);plt.close(fig)
    rows=[('grid','Grid loss, mix masks'),('late-grid','Grid loss, mix renders'),('hybrid','Grid + parameter loss'),('raw','Parameter loss, raw demos'),('code','Parameter loss, object features')]
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),sharey=True,layout='constrained')
    for ax,split,title in zip(axes,['train','heldout'],['Seen operation–selector bindings','Withheld operation–selector bindings']):
        for i,(name,label) in enumerate(rows):
            d=read('xp004-'+name+'-final')[split]['overall']
            ax.barh(i-.17,d['before']*100,.31,color='#326bc4',label='In context' if i==0 else None)
            ax.barh(i+.17,d['after']*100,.31,color='#ed9d38',label='+ 80 gradient steps × 2 starts' if i==0 else None)
            for offset,key in [(-.17,'before'),(.17,'after')]:ax.text(d[key]*100+1,i+offset,f'{d[key]*100:.1f}',va='center',fontsize=9)
        ax.set_title(title);ax.set_xlim(0,100);ax.set_xlabel('Exact query grids (%)');ax.set_yticks(range(len(rows)),[l for n,l in rows]);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle('One demonstration encoder and shared 18-operation executor\n4 demos; 108 puzzles × 4 new queries per split; paired board draws',fontsize=14)
    fig.savefig(REPORTS/'xp004-solvers.png',dpi=160);plt.close(fig)
    controls=[('Structured encoder only',read('xp004-code-final')['heldout']['overall']['before']),
              ('Early routing · 80 steps × 2',read('xp004-code-final')['heldout']['overall']['after']),
              ('Rendered routing · 80 steps × 2',read('xp004-cached-matched-v2')['heldout']['after']),
              ('Rendered routing · 240 steps × 3',read('xp004-cached-code')['heldout']['after']),
              ('No pretrained encoder · 240 × 3',read('xp004-cached-untrained')['heldout']['after']),
              ('8 demos · rendered routing',read('xp004-cached-demos8')['heldout']['after'])]
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for i,(label,value) in enumerate(controls):
        ax.barh(i,value*100,.62,color='#326bc4' if i<2 else '#238571')
        ax.text(min(value*100+1,102),i,f'{100*value:.1f}%',va='center')
    ax.set_xlim(0,110);ax.set_yticks(range(len(controls)),[x[0] for x in controls]);ax.invert_yaxis();ax.set_xlabel('Exact query grids (%)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('Routing after rendering changes inference substantially\nExplicit 894-setting family; withheld bindings; no query targets used',pad=14)
    fig.savefig(REPORTS/'xp004-routing.png',dpi=160);plt.close(fig)
    inventory=[]
    measurements={}
    for path in sorted(RUNS.glob('xp004-*')):
        if not path.is_dir():continue
        config=json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else {}
        status='complete' if (path/'summary.json').exists() else 'interrupted' if (path/'interrupted.json').exists() else 'incomplete'
        summary=json.loads((path/'summary.json').read_text()) if (path/'summary.json').exists() else {}
        inventory.append(dict(run=path.name,status=status,steps=summary.get('steps'),config=config,seconds=summary.get('seconds')))
        measurements[path.name]=summary
    (REPORTS/'xp004-measurements.json').write_text(json.dumps(measurements,indent=2,allow_nan=False)+'\n')
    (REPORTS/'xp004-inventory.json').write_text(json.dumps(inventory,indent=2,allow_nan=False)+'\n')
    table=['| Run | Status | Training updates |','|---|---|---:|']
    table += [f"| `{r['run']}` | {r['status']} | {r['steps'] if r['steps'] is not None else '—'} |" for r in inventory]
    (REPORTS/'xp004-run-inventory.md').write_text('\n'.join(table)+'\n')
    summaryrows=['| Model / training loss | Seen, in context | Seen + adaptation | Withheld, in context | Withheld + adaptation |','|---|---:|---:|---:|---:|']
    for name,label in rows:
        d=read('xp004-'+name+'-final');a,b=d['train']['overall'],d['heldout']['overall']
        summaryrows.append(f"| {label} | {pct(a['before'])} | {pct(a['after'])} | {pct(b['before'])} | {pct(b['after'])} |")
    (REPORTS/'xp004-solver-table.md').write_text('\n'.join(summaryrows)+'\n')
    family=['| Family | In context | Early adaptation | Rendered adaptation | No pretrained encoder |','|---|---:|---:|---:|---:|']
    a=read('xp004-code-final')['heldout']['records'];b=read('xp004-cached-code')['heldout']['records'];c=read('xp004-cached-untrained')['heldout']['records']
    from arcplay.catalogue import OPS
    for op in OPS:
        aa=[r for r in a if r['family']==op];bb=[r for r in b if r['family']==op];cc=[r for r in c if r['family']==op]
        values=[np.mean([r['before']['exact'] for r in aa]),np.mean([r['after']['exact'] for r in aa]),np.mean([r['after'] for r in bb]),np.mean([r['after'] for r in cc])]
        family.append('| '+op+' | '+' | '.join(pct(x) for x in values)+' |')
    (REPORTS/'xp004-family-table.md').write_text('\n'.join(family)+'\n')
    print('Wrote figures, measurements, and tables for',len(inventory),'runs')

if __name__=='__main__':main()
