"""Figures generated solely from recorded experiment measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'results/benchmark.json').read_text())['rows']
names={'reference':'Direct','query_chunked':'Query chunks','recomputed_backward':'Recomputed backward'}
colors={'reference':'#236b8e','query_chunked':'#d48124','recomputed_backward':'#247650'}
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,ax=plt.subplots(figsize=(8,5.8),layout='constrained')
for method,label in names.items():
 r=[x for x in rows if x['method']==method and x['mode']=='forward_backward' and not x['causal']]
 ax.plot([x['tokens'] for x in r],[x['saved_intermediate_storage_bytes']/2**20 for x in r],marker='o',ms=8,lw=2.5,ls='--' if method=='query_chunked' else '-',label=label,color=colors[method])
ax.set(xscale='log',yscale='log',xlabel='Sequence length N',ylabel='Saved intermediate storage (MiB)',title='What survives the forward pass?')
ax.set_xticks([128,512,1024],['128','512','1,024']); ax.set_yticks([.00390625,.015625,.03125,.5,8,32],['4 KiB','16 KiB','32 KiB','0.5 MiB','8 MiB','32 MiB'])
ax.legend(loc='center left');ax.grid(alpha=.18,which='major')
fig.text(.5,-.035,'Unmasked · float32 · batch 1 · 8 heads · excludes pre-existing inputs\nUnique storage saved by autograd, not peak process/device memory.',ha='center',fontsize=10)
fig.savefig(ROOT/'results/storage.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5.2),layout='constrained')
for method,label in names.items():
 r=[x for x in rows if x['method']==method and x['mode']=='forward_backward' and not x['causal']]
 x=[a['tokens'] for a in r]; y=[a['median_ms'] for a in r]
 ax.plot(x,y,marker='o',lw=2.5,label=label,color=colors[method])
 ax.fill_between(x,[a['min_ms'] for a in r],[a['max_ms'] for a in r],alpha=.1,color=colors[method])
ax.set(xlabel='Sequence length N',ylabel='Forward + backward (ms)',title='Recomputation costs time in this CPU baseline')
ax.set_xticks([128,512,1024]);ax.legend();ax.grid(alpha=.2)
fig.text(.5,-.035,'1 CPU thread · float32 · batch 1 · 8 heads · dimension 64\nMedian and min–max of 20 runs; not a GPU performance claim.',ha='center',fontsize=10)
fig.savefig(ROOT/'results/latency.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,6.5));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
def box(x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.15',facecolor=color,edgecolor='none'))
 ax.text(x+w/2,y+h-.35,title,ha='center',va='top',weight='bold',fontsize=13,color='#182b38')
 ax.text(x+w/2,y+.2,body,ha='center',va='bottom',fontsize=10,color='#243d49',linespacing=1.6)
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#536770',lw=2))
ax.text(5,6.8,'Same gradients. Different retained state.',ha='center',fontsize=19,weight='bold')
box(.3,4.8,4.1,1.1,'Ordinary autograd','Forward: compute probability blocks','#e9f1f5')
box(5.6,4.8,4.1,1.1,'Custom backward','Forward: no autograd graph','#e9f4ef')
box(.3,2.6,4.1,1.2,'Retain every block','P₁, P₂, … together contain N² values','#dcebf3')
box(5.6,2.6,4.1,1.2,'Retain one scalar per query','Save Lᵢ = logsumexp(Sᵢ) and Q, K, V','#d9eee3')
box(.3,.4,4.1,1.2,'Backward: reuse P','Use saved probabilities','#e9f1f5')
box(5.6,.4,4.1,1.2,'Backward: reconstruct P','exp(Sᵢ − Lᵢ), block by block','#e9f4ef')
for x in (2.35,7.65):arrow(x,4.7,x,4);arrow(x,2.5,x,1.8)
fig.savefig(ROOT/'results/backward-flow.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('Generated three article figures from actual results.')
