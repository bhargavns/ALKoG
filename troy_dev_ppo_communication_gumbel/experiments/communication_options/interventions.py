"""Route interventions retain separate sender-predicted and delivered caches.

The sender does not receive delivery acknowledgments or read the receiver's
actual cache. Under reliable normal execution the two cache indices coincide.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .models import RouteSafety,probabilities


def route_intervention(model,hard,iv):
    if model.method=='oracle':return model(hard,iv)
    H,V,T=model.H,model.V,model.T;device=model.sender.device
    sender=probabilities(model.sender,hard);decoder=probabilities(model.receiver,hard)
    p=torch.zeros(2,H,H,device=device);p[:,0,0]=.5
    hs=torch.arange(H,device=device)[:,None,None].expand(H,H,V+1)
    hr=torch.arange(H,device=device)[None,:,None].expand(H,H,V+1)
    word=torch.arange(V+1,device=device)[None,None,:].expand(H,H,V+1)
    packets=p.sum()*0;gate_packets=packets;travel_packets=packets
    for t in range(T):
        s=sender[t]
        if model.method=='continuous' or (model.method=='opening' and t==0):
            s=torch.cat((s[:,:,:1]*0,probabilities(model.sender[t,:,:,1:],hard)),-1)
        elif model.method=='silent' or (model.method=='opening' and t>0):
            s=torch.zeros_like(s);s[:,:,0]=1
        elif model.method=='fixed_event':
            s=torch.zeros_like(s)
            for state in range(2):
                for h in range(H):
                    remembered=0 if h in (0,1) else 1
                    z=state+1 if t in model.gates and state!=remembered else 0
                    s[state,h,z]=1
            decoder=torch.zeros_like(decoder);decoder[:,0]=1;decoder[2]=torch.tensor([0.,1.],device=device)
        mass=p[:,:,:,None]*s[:,:,None,:]
        count=mass[:,:,:,1:].sum();packets=packets+count
        if t in model.gates:gate_packets=gate_packets+count
        else:travel_packets=travel_packets+count
        intended_null=torch.zeros_like(hs) if model.method=='memoryless' else hs
        hsn=torch.where(word==0,intended_null,word)
        actual=word
        if iv=='constant':actual=torch.where(word>0,1,0)
        if iv=='permute':actual=torch.where(word>0,word%V+1,0)
        if iv=='blocked' or (iv=='suppress_updates' and t>0):actual=torch.zeros_like(word)
        retained=torch.zeros_like(hr) if model.method=='memoryless' or iv=='reset_memory' else hr
        hrn=torch.where(actual==0,retained,actual)
        transition=F.one_hot(hsn*H+hrn,H*H).float()
        if iv=='drop25':transition=.75*transition+.25*F.one_hot(hsn*H+retained,H*H).float()
        p=(mass.reshape(2,-1)@transition.reshape(-1,H*H)).reshape(2,H,H)
        if t in model.gates:p=p*decoder.T[:,None,:]
        if t<T-1:p=(1-model.rho)*p+model.rho*p.flip(0)
    success=p.sum()
    return dict(success=success,packets=packets,utility=success-.02*packets,
                gate_packets=gate_packets,travel_packets=travel_packets)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    rows=json.loads((a.study/'results.json').read_text())
    with torch.no_grad():
        for row in rows:
            if row['task']!='route_safety':continue
            model=RouteSafety(row['method'])
            model.load_state_dict(torch.load(a.study/f"route_safety_{row['method']}_{row['seed']}"/'checkpoint.pt',weights_only=True,map_location='cpu')['model'])
            for e in row['evaluations']:
                hard=e['deployment']=='argmax'
                values=route_intervention(model,hard,e['intervention'])
                if e['intervention']=='normal':
                    for key in ('success','packets','utility'):
                        assert abs(float(values[key])-e[key])<2e-6
                e.update({k:float(v) for k,v in values.items()})
    (a.study/'validated_results.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
