"""Explanatory figures from existing checkpoints and evaluations; no retraining."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import torch
from .models import TASKS,broadcast_bound
from .navigation import plan,room_route


BLUE='#246487';TEAL='#148678';ORANGE='#cd822e';RED='#bd4d50';GRAY='#7e8b98'


def box(ax,xy,w,h,title,body,color=BLUE):
    x,y=xy
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor='#f0f6f8',edgecolor=color,lw=1.5))
    ax.text(x+w/2,y+h*.78,title,ha='center',va='center',fontsize=12,weight='bold',color=color)
    ax.text(x+w/2,y+h*.38,body,ha='center',va='center',fontsize=10,linespacing=1.7)


def arrow(ax,a,b,label='',color=GRAY):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.8,color=color))
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.035,label,ha='center',fontsize=9,color=color)


def make_figures(study,output,rows):
    study,output=Path(study),Path(output)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(output/name,dpi=165,bbox_inches='tight',facecolor='white');plt.close(fig)
    def load(task,method,seed):
        model=TASKS[task][0](method)
        model.load_state_dict(torch.load(study/f'{task}_{method}_{seed}'/'checkpoint.pt',weights_only=True,map_location='cpu')['model'])
        return model
    normal=[dict(task=r['task'],method=r['method'],seed=r['seed'],**e)
            for r in rows for e in r['evaluations'] if e['deployment']=='argmax' and e['intervention']=='normal']
    def values(task,method,key):return [r[key] for r in normal if r['task']==task and r['method']==method]

    # Saved navigation-check map and the exact public two-passage room geometry.
    # These are planner checks, not a claim that the finite learner trained on maps.
    layout=json.loads((study/'navigation_examples.json').read_text())[0]
    fig,(gx,rx)=plt.subplots(1,2,figsize=(13,7))
    def grid(ax,width,height):
        ax.set(xlim=(-.5,width-.5),ylim=(height-.5,-.5),aspect='equal',
               xticks=range(width),yticks=range(height),xlabel='x cell',ylabel='y cell')
        ax.set_xticks(np.arange(-.5,width,1),minor=True)
        ax.set_yticks(np.arange(-.5,height,1),minor=True)
        ax.grid(which='minor',color='#d6e0e6',lw=.8)
        ax.tick_params(which='minor',length=0)
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#a8b6c0')
    def cell(ax,p,color,edge='white'):
        ax.add_patch(plt.Rectangle((p[0]-.5,p[1]-.5),1,1,facecolor=color,edgecolor=edge,zorder=2))
    def receiver(ax,p):
        ax.scatter(*p,s=280,color=BLUE,zorder=6)
        ax.text(*p,'R',ha='center',va='center',weight='bold',color='white',zorder=7)
    def goal(ax,p,label):
        ax.add_patch(plt.Rectangle((p[0]-.39,p[1]-.39),.78,.78,facecolor='#d8eee7',edgecolor=TEAL,lw=1.5,zorder=4))
        ax.text(*p,label,ha='center',va='center',weight='bold',color='#18584f',zorder=5,fontsize=9)
    grid(gx,13,13)
    for p in layout['blocked']:cell(gx,p,'#596875')
    path=layout['paths'][2]
    gx.plot([p[0] for p in path],[p[1] for p in path],color=TEAL,lw=3,zorder=3)
    for i,p in enumerate(layout['goals']):goal(gx,p,f'G{i}')
    receiver(gx,layout['start'])
    gx.set_title('13 × 13: saved public-map validation example',fontsize=12,pad=13)
    gx.text(.5,-.14,'Actual saved layout 0; the illustrated decoded choice is G2.\nOnly the sender knows the true rewarded destination.',
            transform=gx.transAxes,ha='center',va='top',fontsize=10)
    grid(rx,7,7)
    for y in range(7):
        if y not in (1,5):cell(rx,(3,y),'#596875')
    for choice,color,style in ((0,TEAL,'-'),(1,ORANGE,'--')):
        path=room_route(choice)
        rx.plot([p[0] for p in path],[p[1] for p in path],color=color,ls=style,lw=3,zorder=3)
        gate=(3,1 if choice==0 else 5)
        rx.scatter(*gate,s=320,marker='s',facecolor='white',edgecolor=color,lw=2,zorder=5)
        rx.text(*gate,str(choice),ha='center',va='center',weight='bold',color=color,zorder=6)
    receiver(rx,(0,3));goal(rx,(6,3),'G')
    rx.set_title('7 × 7: public two-passage room',fontsize=12,pad=13)
    rx.text(.5,-.14,'Both candidate openings are public; their safety is private.\nThe receiver plans through passage 0 or 1 after decoding.',
            transform=rx.transAxes,ha='center',va='top',fontsize=10)
    handles=[Patch(facecolor='white',edgecolor='#b8c6cf',label='floor cell'),
             Patch(facecolor='#596875',label='wall / blocked cell'),
             Line2D([],[],marker='o',ls='',color=BLUE,label='R: receiver start'),
             Patch(facecolor='#d8eee7',edgecolor=TEAL,label='G: public destination'),
             Line2D([],[],color=TEAL,lw=3,label='receiver-planned route'),
             Line2D([],[],color=ORANGE,lw=3,ls='--',label='alternative passage route')]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=10,frameon=False,bbox_to_anchor=(.5,.005))
    fig.suptitle('Top-down navigation grids: cells replace a continuous MuJoCo floor',weight='bold',fontsize=16,y=.98)
    fig.subplots_adjust(bottom=.25,top=.88,wspace=.27)
    save(fig,'grid-layouts.png')

    # 1. Information flow: show exactly where private information stops.
    fig,ax=plt.subplots(figsize=(12,4.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,(.02,.55),.23,.32,'Sender','Private goal / safe side\nOwn observation + history')
    box(ax,(.39,.55),.22,.32,'Public channel','Emit a learned word\nor remain silent',TEAL)
    box(ax,(.75,.55),.23,.32,'Receiver','Own permitted observations\nMessage → memory → decision')
    arrow(ax,(.25,.71),(.38,.71),'word / silence');arrow(ax,(.62,.71),(.75,.71),'signal')
    box(ax,(.35,.06),.30,.25,'Receiver-local planner','Selected public waypoint\n+ public map → movement',TEAL)
    arrow(ax,(.85,.53),(.65,.2),'waypoint')
    ax.text(.14,.30,'Hidden sender fact has\nno direct receiver input',ha='center',color=RED,fontsize=11)
    ax.text(.85,.10,'Training may inspect truth\nto score the joint outcome.\nDeployment follows these arrows.',ha='center',fontsize=10)
    ax.set_title('What each agent knows, and how a message affects action',pad=18,weight='bold')
    save(fig,'information-flow.png')

    # 2. Actual learned four-goal code plus local path, not a made-up codebook.
    model=load('private_goal','learned',0)
    with torch.no_grad():
        signals=model.sender.argmax(-1).numpy();decoded=model.receiver.argmax(-1).numpy()
    goals=[(1,1),(11,1),(1,11),(11,11)];g=2;chosen=int(decoded[signals[g]])
    path=plan((6,6),goals[chosen])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.2]})
    ax.set(xlim=(-.5,12.5),ylim=(12.5,-.5),aspect='equal',xticks=range(13),yticks=range(13));ax.grid(alpha=.16)
    ax.tick_params(labelbottom=False,labelleft=False,length=0)
    for i,(x,y) in enumerate(goals):
        ax.scatter(x,y,s=220,marker='s',color='#e0e8ed',edgecolor=BLUE,zorder=3)
        ax.text(x,y,str(i),ha='center',va='center',weight='bold',zorder=4)
    ax.plot([p[0] for p in path],[p[1] for p in path],color=TEAL,lw=3,label='receiver-planned path')
    ax.scatter(6,6,s=150,color=BLUE,zorder=5,label='receiver start')
    ax.set_title('Same public map for every possible target')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.03),fontsize=9)
    used=sorted(set(int(s) for s in signals));matrix=np.zeros((4,len(used)))
    for i,s in enumerate(signals):matrix[i,used.index(int(s))]=1
    bx.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    bx.set(xticks=range(len(used)),xticklabels=['silence' if z==0 else f'word {z}' for z in used],
           yticks=range(4),yticklabels=[f'Private target {i}' for i in range(4)],xlabel='Opening signal')
    for i in range(4):
        for j in range(len(used)):
            bx.text(j,i,'1' if matrix[i,j] else '0',ha='center',va='center',color='white' if matrix[i,j] else GRAY)
    bx.set_title('Actual deterministic code, learned seed 0')
    label='silence' if signals[g]==0 else f'word {signals[g]}'
    fig.suptitle(f'Example: sender knows target {g} → {label} → receiver selects {chosen}',weight='bold',y=1.03)
    fig.tight_layout();save(fig,'private-goal-code.png')

    # 3. Trace a frozen learned policy through an illustrative change history.
    model=load('route_safety','learned',0)
    state=np.array([0,0,0,1,1,1,1,0,0]);cache=0;sent=[];remember=[]
    s=model.sender.detach().argmax(-1).numpy();d=model.receiver.detach().argmax(-1).numpy()
    for t in range(9):
        z=int(s[t,state[t],cache]);sent.append(z)
        if z:cache=z
        remember.append(int(d[cache]))
    fig,axes=plt.subplots(3,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[1,1,1.3]})
    axes[0].step(range(9),state,where='mid',color=BLUE,lw=2.5)
    axes[1].step(range(9),remember,where='mid',color=TEAL,lw=2.5)
    for ax,label in zip(axes[:2],('True safe passage\n(sender observes)','Remembered choice\n(receiver computes)')):
        ax.set(yticks=[0,1],yticklabels=['upper','lower'],ylim=(-.3,1.3));ax.set_ylabel(label,fontsize=10)
    for t,z in enumerate(sent):
        axes[2].scatter(t,1 if z else 0,s=70,color=ORANGE if z else '#bdc7ce',zorder=4)
        axes[2].text(t,1.16 if z else .15,f'w{z}' if z else '—',ha='center',fontsize=10)
    axes[2].set(yticks=[0,1],yticklabels=['silent','packet'],ylim=(-.25,1.65),xticks=range(9),xlabel='Logical observation tick')
    for ax in axes:
        for tick in (0,4,8):ax.axvline(tick,color=GRAY,ls=':',alpha=.55)
        ax.grid(axis='x',alpha=.12)
    axes[0].set_title('An illustrative frozen-policy trace: changes do not require a constant stream',weight='bold',pad=18)
    axes[2].text(0,1.5,'Crossing 1',ha='left',color=BLUE,fontsize=9)
    axes[2].text(4,1.5,'Crossing 2',ha='center',color=BLUE,fontsize=9)
    axes[2].text(8,1.5,'Crossing 3',ha='right',color=BLUE,fontsize=9)
    fig.tight_layout();save(fig,'route-memory-timeline.png')

    # 4. Economic threshold and actual learned conditional action choices.
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.7))
    x=np.linspace(0,.45,200)
    ax.plot(x,1-x,color=BLUE,lw=2.5,label='inspect: utility 1 − d')
    ax.axhline(.92,color=TEAL,lw=2.5,label='ask: utility 1 − 0.08')
    ax.axvline(.08,color=GRAY,ls=':')
    ax.axvspan(0,.08,color=BLUE,alpha=.07);ax.axvspan(.08,.45,color=TEAL,alpha=.07)
    ax.text(.04,.72,'Inspect',ha='center',color=BLUE);ax.text(.28,.72,'Ask',ha='center',color=TEAL)
    for cost in (.02,.12,.4):ax.plot(cost,max(1-cost,.92),'o',color=ORANGE,ms=7)
    ax.set(xlim=(0,.45),ylim=(.55,1.02),xlabel='Inspection cost d',ylabel='Utility when target is hidden',title='Why the best action changes at d = 0.08')
    ax.legend(fontsize=9,loc='lower left')
    names=['hidden, d=.02','known, d=.02','hidden, d=.12','known, d=.12','hidden, d=.40','known, d=.40']
    warm=[r for r in normal if r['task']=='scouting' and r['method']=='learned_warm']
    mat=[]
    for i in range(6):
        ask=np.mean([r[f'context_{i}_ask'] for r in warm]);scout=np.mean([r[f'context_{i}_scout'] for r in warm])
        mat.append([1-ask-scout,scout,ask])
    bx.imshow(mat,cmap='Greens',vmin=0,vmax=1,aspect='auto')
    bx.set(yticks=range(6),yticklabels=names,xticks=range(3),xticklabels=['act directly','inspect','ask'],title='Actual warm-policy decisions (five-seed mean)')
    for i in range(6):
        for j in range(3):bx.text(j,i,f'{mat[i][j]:.0%}',ha='center',va='center',color='white' if mat[i][j]>.5 else GRAY)
    fig.tight_layout();save(fig,'scouting-decision.png')

    # 5. Actual successful dialogue example and its bit accounting.
    model=load('complementary','dialogue_warm',2)
    q=3;bits=np.array([1,0,0,1,0,1,0,1])
    with torch.no_grad():
        request=int(model.query.argmax(-1)[q]);slot=int(model.lookup.argmax(-1)[request-1])
        reply=int(model.bit_encoder.argmax(-1)[int(bits[slot])]);answer=int(model.answer.argmax(-1)[q,reply])
    fig,ax=plt.subplots(figsize=(12,5.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,(.02,.55),.30,.32,'Receiver’s private knowledge',f'Needs entity {q}\nDoes not know its safe entrance')
    box(ax,(.68,.55),.30,.32,'Sender’s private knowledge','Hazard map X\n[1, 0, 0, 1, 0, 1, 0, 1]')
    arrow(ax,(.33,.78),(.67,.78),f'query word {request} (3 bits)',BLUE)
    arrow(ax,(.67,.58),(.33,.58),f'answer word {reply} (1 bit)',TEAL)
    ax.text(.5,.41,f'Learned lookup selects map slot {slot}; receiver recovers entrance {answer}.',ha='center',fontsize=11)
    ax.text(.5,.32,'One request-presence bit is also charged. This is a selected seed-2 example.',ha='center',color=GRAY,fontsize=10)
    labels=['Request flag','Query','Reply'];sizes=[1,3,1];colors=[GRAY,BLUE,TEAL]
    left=.14
    for label,size,color in zip(labels,sizes,colors):
        width=size*.14
        ax.add_patch(plt.Rectangle((left,.10),width,.10,facecolor=color,edgecolor='white'))
        ax.text(left+width/2,.15,f'{label}: {size}',ha='center',va='center',color='white',fontsize=10)
        left+=width
    ax.text(.5,.04,'At most 5 charged wire bits, compared with 8 for the full map',ha='center',weight='bold')
    ax.set_title('A question identifies which private fact matters',pad=18,weight='bold')
    save(fig,'dialogue-information-flow.png')

    # 6. Visualize the exact entropy converse, not another fitted result.
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.7))
    accuracy=np.linspace(.5,.999999,500);error=1-accuracy
    entropy=8*(-error*np.log2(error)-(1-error)*np.log2(1-error))
    bound=broadcast_bound()
    ax.plot(accuracy,entropy,color=BLUE,lw=2.5,label='8 h₂(error)')
    ax.axhline(3,color=ORANGE,ls='--',label='at least 3 bits remain unknown')
    ax.axvline(bound,color=RED,ls=':',label=f'accuracy ≤ {100*bound:.2f}%')
    ax.axvspan(bound,1,color=RED,alpha=.07)
    ax.set(xlim=(.5,1),ylim=(0,8.3),xlabel='One-way retrieval accuracy',ylabel='Conditional-entropy upper bound (bits)',title='A five-bit message cannot describe every map fact')
    ax.legend(fontsize=8,loc='lower left')
    methods=['silent','broadcast5','dialogue_warm']
    ys=[np.mean(values('complementary',m,'success')) for m in methods]
    errs=[np.std(values('complementary',m,'success'),ddof=1) for m in methods]
    bx.bar(range(3),ys,yerr=errs,capsize=4,color=[GRAY,BLUE,TEAL])
    bx.axhline(bound,color=RED,ls='--',label='one-way upper bound')
    bx.set(xticks=range(3),xticklabels=['silent','one-way 5 bits','queried dialogue'],ylim=(0,1.08),ylabel='Accuracy',title='A query changes the information problem')
    bx.legend(fontsize=9,loc='lower right')
    fig.tight_layout();save(fig,'entropy-bound.png')

    # 7. Causal deletion, evaluated pairs rather than post-hoc invented scores.
    tasks=list(TASKS);primary={'private_goal':'learned','route_safety':'learned','scouting':'learned_warm','complementary':'dialogue_warm'}
    fig,ax=plt.subplots(figsize=(11,4.5));xs=np.arange(4)
    for offset,iv,color,label in ((-.18,'normal',TEAL,'normal channel'),(.18,'blocked',RED,'delivery blocked')):
        samples=[]
        for task in tasks:
            samples.append([e['success'] for r in rows if r['task']==task and r['method']==primary[task]
                            for e in r['evaluations'] if e['deployment']=='argmax' and e['intervention']==iv])
        ax.bar(xs+offset,[np.mean(s) for s in samples],width=.34,yerr=[np.std(s,ddof=1) for s in samples],capsize=3,color=color,label=label)
    ax.set(xticks=xs,xticklabels=['Private goal','Changing safety','Scouting','Complementary'],ylim=(0,1.09),ylabel='Success probability',title='What happens when information is prevented from reaching a frozen partner?')
    ax.legend(loc='lower left');ax.grid(axis='y',alpha=.15);fig.tight_layout()
    save(fig,'message-interventions.png')

    # 8. Surface the failed seed responsible for the qualified dialogue result.
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,4.6));warm=[r for r in normal if r['task']=='complementary' and r['method']=='dialogue_warm']
    warm=sorted(warm,key=lambda r:r['seed'])
    ax.bar(range(5),[r['success'] for r in warm],color=BLUE)
    ax.set(xticks=range(5),xlabel='Optimization seed',ylabel='Accuracy',ylim=(0,1.08),title='A 95% mean includes imperfect runs')
    ax.axhline(.95,color=GRAY,ls=':',label='five-seed mean');ax.legend(loc='lower left',fontsize=9)
    bx.bar(range(5),[r['utility'] for r in warm],color=[RED if r['utility']<.84 else TEAL for r in warm])
    bx.axhline(.84,color=ORANGE,ls='--',label='full-map broadcast reference')
    bx.set(xticks=range(5),xlabel='Optimization seed',ylabel='Utility',ylim=(0,1),title='Seed 1 fails the required utility comparison')
    bx.legend(loc='lower left',fontsize=9);fig.tight_layout();save(fig,'dialogue-seed-reliability.png')

    return ['grid-layouts.png','information-flow.png','private-goal-code.png','route-memory-timeline.png','scouting-decision.png',
            'dialogue-information-flow.png','entropy-bound.png','message-interventions.png','dialogue-seed-reliability.png']
