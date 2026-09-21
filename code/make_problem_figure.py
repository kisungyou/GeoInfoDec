"""Render the introductory illustration from its retained clouds and estimates.

No fitting or simulation is performed. reference_results/problem_clouds.npz and
reference_results/problem_figure.json preserve the original executed illustration.
"""
from pathlib import Path
import json
import numpy as np
from matplotlib.patches import Patch
from figure_style import plt,save_figure,COLORS,TEXT_WIDTH_IN

from run_config import OUT, REFERENCE


def plot_problem(clouds,records):
    fig,axs=plt.subplots(2,3,figsize=(TEXT_WIDTH_IN,3.15),
        gridspec_kw={'height_ratios':[1.3,.75]})
    fig.subplots_adjust(left=.095,right=.988,bottom=.19,top=.915,hspace=.23,wspace=.28)
    for j,(X,row) in enumerate(zip(clouds,records)):
        ax=axs[0,j]
        ax.scatter(X[:,0],X[:,2],c=X[:,1],cmap='viridis',s=3,alpha=.68,linewidths=0)
        ax.add_patch(plt.Circle((0,0),1,facecolor='none',edgecolor=COLORS['gray'],linewidth=.65))
        ax.set(xlim=(-1.08,1.08),ylim=(-1.08,1.08),aspect='equal',xticks=[],yticks=[])
        ax.set_axis_off();ax.set_title(row['scenario'],pad=5)
        axs[1,j].bar([0,1],[row['I1'],row['I2']],
            color=[COLORS['blue'],COLORS['orange']],width=.55)
        axs[1,j].set(xticks=[0,1],xticklabels=[r'$I_1$',r'$I_2$'],ylim=(0,2),yticks=[0,1,2])
        if j==0:axs[1,j].set_ylabel('Fitted gap (nats)')
    fig.legend([Patch(facecolor=COLORS['blue']),Patch(facecolor=COLORS['orange'])],
               [r'Mean gap $I_1$',r'Residual gap $I_2$'],ncol=2,loc='lower center',
               bbox_to_anchor=(.55,.015))
    meta=save_figure(fig,OUT/'figures/problem');plt.close(fig)
    return meta


def main():
    data=json.loads((REFERENCE/'problem_figure.json').read_text())
    with np.load(REFERENCE/'problem_clouds.npz') as stored:
        clouds=[stored[str(j)] for j in range(len(data['results']))]
    return plot_problem(clouds,data['results'])

if __name__=='__main__':main()
