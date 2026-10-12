"""Fixed scene32 illustration plus all40-scene pair distances; no cherry-picked scene."""
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];d=r/'runs/intervention_20261012/contact_paired_production_20261013';o=r/'runs/contact_mechanism_figures_20261013';o.mkdir(exist_ok=True)
z=np.load(d/'full_trajectories_with_repeats.npz');mass=[.01,.02,.04];force=[.01,.02,.03];times=np.arange(121)*.004
key=lambda sc,mode,m,f:f's{sc}_{mode}_m{m}_f{f}_r0_fullstates'
fig,axes=plt.subplots(1,3,figsize=(15,4.4));colors=plt.get_cmap('tab10').colors
for ax,mode in zip(axes[:2],['on','off']):
 for i,m in enumerate(mass):
  for j,f in enumerate(force):
   x=z[key(32,mode,m,f)][:,0,0];ax.plot(times,(x-x[0])*1000,color=colors[3*i+j],label=f'm={m:g}, F={f:g}',lw=1.4,ls=['-','--',':'][i])
 ax.axvline(.12,color='gray',ls=':',lw=1);ax.set(title=f'Robot collision {mode.upper()} | fixed scene 32',xlabel='Time (s)',ylabel='Object x displacement (mm)');ax.grid(alpha=.2)
pairs=[((.01,.01),(.02,.02)),((.02,.01),(.04,.02))];summary=[]
for sc in range(40):
 for i,(a,b) in enumerate(pairs):
  vals=[]
  for mode in ['on','off']:
   A=z[key(sc,mode,*a)][:,0,:3];B=z[key(sc,mode,*b)][:,0,:3];vals.append(float(np.sqrt(np.mean(np.sum((A-B)**2,axis=1)))*1000))
  summary.append([sc,i,*vals])
s=np.array(summary)
for i in range(2):
 mask=s[:,1]==i;axes[2].scatter(s[mask,2],s[mask,3],s=23,alpha=.7,label=f'Equal F/m pair {i+1}',marker=['o','x'][i])
lim=max(.1,s[:,2:].max()*1.1);axes[2].plot([0,lim],[0,lim],color='gray',ls=':',lw=1);axes[2].set(xlabel='Collision ON position RMS difference (mm)',ylabel='Collision OFF position RMS difference (mm)',title='Equal-ratio pairs | all 40 scenes',xlim=(-.02*lim,lim),ylim=(-.02*lim,lim));axes[2].legend(fontsize=8);axes[2].grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=9,fontsize=7,frameon=False);fig.tight_layout(rect=(0,.1,1,1));fig.savefig(o/'contact_scaling_symmetry.png',dpi=180);fig.savefig(o/'contact_scaling_symmetry.svg');np.savez(o/'plotted_pairs.npz',scene=s[:,0],pair=s[:,1],on_position_rms_mm=s[:,2],off_position_rms_mm=s[:,3]);print('figure saved',s[:,2:].max(0))

fig,ax=plt.subplots(figsize=(8,4.3))
for i,(a,b) in enumerate(pairs):
 curves=[]
 for sc in range(40):
  A=z[key(sc,'on',*a)][:,0,:3];B=z[key(sc,'on',*b)][:,0,:3];curves.append(np.linalg.norm(A-B,axis=1)*1000)
 curves=np.array(curves);q=np.percentile(curves,[25,50,75],axis=0);ax.plot(times,q[1],color=colors[i],label=f'Collision ON, equal-ratio pair {i+1}');ax.fill_between(times,q[0],q[2],color=colors[i],alpha=.15)
ax.plot(times,np.zeros(121),color='black',ls='--',label='Collision OFF, both pairs: exactly zero');ax.axvline(.12,color='gray',ls=':',lw=1);ax.set(xlabel='Time (s)',ylabel='3D position difference (mm)',title='Equal F/m pairs: median and IQR across all 40 scenes');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(o/'equal_ratio_contact_effect.png',dpi=180);fig.savefig(o/'equal_ratio_contact_effect.svg')
