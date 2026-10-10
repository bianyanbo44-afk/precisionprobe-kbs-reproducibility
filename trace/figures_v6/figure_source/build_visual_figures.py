"""Reproduce TRACE publication figures from the retained source tables.

No inference, resampling, selection or smoothing is performed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, Patch
import numpy as np
import pandas as pd
from audit_panel_alignment import require_matplotlib_panel_alignment

MM=1/25.4
BLUE='#0072B2'
ORANGE='#D55E00'
INK='#22272B'
GREY='#69747C'
LIGHT='#F0F2F3'
MODELS=['deepseek','qwen25','qwen3']
MODEL_NAMES=['DeepSeek','Qwen2.5','Qwen3']
plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':9,'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'axes.linewidth':.65,'axes.edgecolor':INK,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
    'xtick.major.size':3,'ytick.major.size':3,'xtick.major.width':.65,
    'ytick.major.width':.65,'savefig.facecolor':'white','figure.facecolor':'white',
})

def export(fig, out, name, multipanel=False):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,json_out=out/f'{name}.alignment.json',tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,require_panel_labels=multipanel,strict=True)
    fig.savefig(out/f'{name}.pdf')
    fig.savefig(out/f'{name}.svg')
    fig.savefig(out/f'{name}.png',dpi=600)
    fig.savefig(out/f'{name}.tiff',dpi=1000,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def panel_grid(height):
    fig,axes=plt.subplots(2,3,figsize=(180*MM,height*MM),sharey=True)
    fig.subplots_adjust(left=.115,right=.975,bottom=.145,top=.79,
                        wspace=.23,hspace=.46)
    for j,name in enumerate(MODEL_NAMES):
        pos=axes[0,j].get_position()
        fig.text(pos.x0+pos.width/2,.842,name,ha='center',va='center',
                 fontsize=10,fontweight='bold')
    for i,p in enumerate(['Q4','Q8']):
        pos=axes[i,0].get_position()
        fig.text(.064,pos.y0+pos.height/2,p,rotation=90,rotation_mode='anchor',
                 ha='center',va='center',fontsize=9.5,fontweight='bold')
    for k,ax in enumerate(axes.flat):
        ax.text(-.03,1.06,chr(97+k),transform=ax.transAxes,
                fontsize=11,fontweight='bold',ha='left',va='bottom')
        ax.tick_params(pad=3)
    return fig,axes

def workflow(out):
    fig=plt.figure(figsize=(180*MM,88*MM))
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,180),ylim=(0,88));ax.set_axis_off()
    def txt(x,y,s,size=9,weight='normal',ha='center',va='center',color=INK):
        return ax.text(x,y,s,fontsize=size,fontweight=weight,ha=ha,va=va,
                       color=color,linespacing=1.35)
    def box(x,y,w,h,title,body,accent=BLUE,fill='#F6F8F9'):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor='#A1AAB0',linewidth=.6))
        ax.plot([x,x+w],[y+h,y+h],color=accent,linewidth=1.4,solid_capstyle='butt')
        txt(x+w/2,y+h-4.3,title,9.5,'bold')
        txt(x+w/2,y+h/2-2,body,9)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,
                                    linewidth=.8,color=GREY,shrinkA=1.3,shrinkB=1.3))
    txt(4,85,r'$n$ incoming targets; review positions $q=\lceil bn\rceil$',ha='left',color=GREY)
    box(4,47,37,28,'1  Target checks','Generate one target\nRun available checks\nCount known failures: f')
    arrow((41,60),(54,60))
    txt(67,79,'2  Capacity boundary',9.5,'bold')
    ax.add_patch(Polygon([(54,60),(67,74),(80,60),(67,46)],
                         facecolor='white',edgecolor=GREY,linewidth=.8))
    txt(67,60,r'$f\geq q$ ?',11)
    arrow((67,46),(67,41));txt(73,43.5,'Yes',9,ha='left')
    box(46,23,42,18,'Queue saturated','Review q known errors\nMaximum error capture: q',
        accent=ORANGE,fill='#FCF5EF')
    arrow((80,60),(97,60));txt(88.5,65,'No',9)
    box(97,47,79,28,'3  Order passing targets',
        'Reserve f slots; h = q - f remain\nNLL vs. DSDE-derived scores (K = 1, 2, 4)\nPaired error capture + sampling cost')
    arrow((136.5,47),(136.5,41))
    box(97,23,79,18,'Review allocation','Choose ordering from paired evidence\nReview h passing targets; f + h = q')
    ax.add_patch(Rectangle((4,2),172,18,facecolor='#F6F8F9',edgecolor='#A1AAB0',linewidth=.6))
    txt(7,16.5,'4  Acceptance under an error budget (when required)',9.5,'bold',ha='left')
    txt(29,10.1,'Disjoint reference\nScore percentiles',8.9)
    arrow((49,10.1),(59,10.1))
    txt(85,10.1,'Disjoint calibration\nBound accepted-error risk',8.9)
    arrow((113,10.1),(123,10.1))
    txt(150,10.1,'Feasible rule or\nzero acceptance',8.9)
    txt(7,3.7,'Held-out tasks evaluate coverage and accepted error.',8.8,ha='left',color=GREY)
    export(fig,out,'trace_workflow')

METHODS=[
 ('mean_nll','NLL',dict(marker='D',s=25,facecolors='#707A81',edgecolors='#505A62',linewidths=.7,zorder=4)),
 ('public_then_nll','Public/NLL',dict(marker='s',s=65,facecolors='none',edgecolors=INK,linewidths=1,zorder=7)),
 ('dsde_1','DSDE1',dict(marker='^',s=43,facecolors='none',edgecolors=BLUE,linewidths=1,zorder=5)),
 ('dsde_2','DSDE2',dict(marker='v',s=35,facecolors='#78B7D4',edgecolors=BLUE,linewidths=.7,zorder=4)),
 ('dsde_4','DSDE4',dict(marker='o',s=30,facecolors=ORANGE,edgecolors=ORANGE,linewidths=.7,zorder=5)),
 ('public_then_dsde','Public/DSDE4',dict(marker='h',s=100,facecolors='none',edgecolors=ORANGE,linewidths=1,zorder=8))]

def cost_plot(source,out,receipt):
    data=pd.read_csv(source/'validation_cost_source.csv',float_precision='round_trip')
    fig,axes=panel_grid(112)
    display=[]
    max_by_model={m:float(data[data.condition.str.startswith(m+'_')].mean_total_seconds.max()) for m in MODELS}
    xmaxs={'deepseek':22,'qwen25':12,'qwen3':95}
    xticks={'deepseek':[0,10,20],'qwen25':[0,5,10],'qwen3':[0,40,80]}
    for ax,condition in zip(axes.flat,[m+'_'+p for p in ['q4','q8'] for m in MODELS]):
        model=condition.rsplit('_',1)[0]
        cell=data[data.condition==condition].set_index('score')
        baseline=float(cell.loc['public_then_nll','residual_error_per_task'])
        ax.axhline(0,color=GREY,linestyle=(0,(3,3)),linewidth=.65,zorder=1)
        for score,label,style in METHODS:
            row=cell.loc[score]
            x=float(row.mean_total_seconds)
            y=100*(float(row.residual_error_per_task)-baseline)
            assert y>=-1e-10 and int(row.n)==175 and int(row.reviewed)==44
            ax.scatter(x,y,label=label,**style)
            display.append({'condition':condition,'score':score,'acquisition_seconds':x,
                            'excess_residual_per_100':y,'baseline_residual_per_100':100*baseline})
        assert xmaxs[model]>max_by_model[model]
        ax.set(xlim=(0,xmaxs[model]),ylim=(-.30,2.2),xticks=xticks[model],yticks=[0,1,2])
    handles,labels=axes[0,0].get_legend_handles_labels()
    # Keep the three target/sampling pairs aligned in a compact shared key.
    fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.54,.995),
               frameon=False,handletextpad=.6,columnspacing=2.0,labelspacing=.75)
    fig.text(.027,.46,'Excess residual errors vs. Public/NLL\n(per 100 incoming tasks)',
             rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=9.5,linespacing=1.3)
    fig.text(.545,.027,'Mean acquisition time (s per task)',ha='center',fontsize=9.5)
    receipt['cost_display']=display
    export(fig,out,'validation_cost_frontier',True)

def capacity_plot(source,out,receipt):
    data=pd.read_csv(source/'guarded_capacity_source.csv',float_precision='round_trip')
    assert len(data)==1050
    fig,axes=panel_grid(110)
    for ax,condition in zip(axes.flat,[m+'_'+p for p in ['q4','q8'] for m in MODELS]):
        cell=data[data.condition==condition].sort_values('review_slots')
        assert len(cell)==175 and list(cell.review_slots)==list(range(1,176))
        boundary=float(cell.public_failure_fraction.iloc[0])*100
        color=BLUE if condition.endswith('q4') else ORANGE
        ax.axvspan(0,boundary,color=LIGHT,zorder=0)
        ax.axhline(0,color='#90999F',linewidth=.65,zorder=1)
        ax.axvline(boundary,color=GREY,linestyle=(0,(1.4,2)),linewidth=.8,zorder=2)
        # Every integer queue-size observation is retained; no interpolation smoothing.
        ax.plot(100*cell.review_fraction,cell.additional_errors_captured,
                color=color,linewidth=1.05,zorder=3)
        ax.set(xlim=(0,100),ylim=(-2.9,2.5),xticks=[0,25,50,75,100],yticks=[-2,0,2])
    key=[Patch(facecolor=LIGHT,edgecolor='#B4BBC0',linewidth=.5),
         Line2D([],[],color=GREY,linestyle=(0,(1.4,2)),linewidth=.8)]
    fig.legend(key,['Known failures fill queue','Known-failure capacity'],loc='upper center',
               bbox_to_anchor=(.55,.96),ncol=2,frameon=False,columnspacing=1.6,handlelength=1.6)
    fig.text(.027,.46,'Additional errors captured\n(Public/DSDE4 minus Public/NLL)',
             rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=9.5,linespacing=1.3)
    fig.text(.545,.027,'Review capacity (%)',ha='center',fontsize=9.5)
    receipt['capacity_points']=len(data)
    export(fig,out,'guarded_capacity_curve',True)

def forest_plot(source,out,receipt):
    data=pd.read_csv(source/'retrospective_comparison_source.csv',float_precision='round_trip')
    assert len(data)==10 and np.isfinite(data[['difference','lower','upper']]).all().all()
    groups=['deepseek_humaneval','deepseek_mbpp','qwen_humaneval','qwen_mbpp','qwen3_mbpp60']
    names=['DeepSeek / HumanEval+','DeepSeek / MBPP+','Qwen2.5 / HumanEval+','Qwen2.5 / MBPP+','Qwen3 / MBPP+']
    fig,ax=plt.subplots(figsize=(180*MM,96*MM))
    fig.subplots_adjust(left=.34,right=.975,bottom=.245,top=.86)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=10)
    for i,group in enumerate(groups):
        for p,offset in [('q4',.15),('q8',-.15)]:
            row=data[(data.panel==group)&(data.precision==p)].iloc[0]
            value=float(row.difference)*100
            lo=float(row.lower)*100;hi=float(row.upper)*100
            assert lo<=value<=hi
            ax.errorbar(value,4-i+offset,xerr=[[value-lo],[hi-value]],
                        fmt='o' if p=='q4' else 's',color=BLUE if p=='q4' else ORANGE,
                        markersize=4.5,elinewidth=1,capsize=2,capthick=.8,zorder=3)
    ax.axvline(0,color=GREY,linestyle=(0,(3,3)),linewidth=.7,zorder=1)
    ax.set(yticks=list(range(4,-1,-1)),yticklabels=names,xlim=(-14.5,11),
           ylim=(-.55,4.6),xticks=[-10,-5,0,5,10])
    ax.set_xlabel('DSDE4 minus Exec/NLL\nResidual errors per 100 incoming tasks',labelpad=8)
    handles=[Line2D([],[],marker='o',color=BLUE,lw=1,markersize=4.5),
             Line2D([],[],marker='s',color=ORANGE,lw=1,markersize=4.5)]
    fig.legend(handles,['Q4','Q8'],ncol=2,loc='upper center',bbox_to_anchor=(.67,.995),
               frameon=False,columnspacing=2.0)
    fig.text(.34,.035,'Qwen3 matched targets: Q4 n = 58; Q8 n = 60.',fontsize=9)
    receipt['retrospective_estimates']=len(data)
    export(fig,out,'retrospective_differences')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent/'figures')
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    receipt={'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
               for p in args.source_dir.glob('*.csv')},'data_and_statistics_recomputed':False,
             'method_points_shifted_or_jittered':False,'bootstrap_intervals':'original source columns'}
    workflow(args.out_dir)
    cost_plot(args.source_dir,args.out_dir,receipt)
    capacity_plot(args.source_dir,args.out_dir,receipt)
    forest_plot(args.source_dir,args.out_dir,receipt)
    (args.out_dir/'visual_data_integrity.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('Exported four source-linked figures with editable fonts and alignment receipts.')

if __name__=='__main__':main()
