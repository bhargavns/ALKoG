"""Inspect learned token grounding, silent retention and contextual movement."""
import argparse
import json
from pathlib import Path
import torch
from .study import Pair


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    results=json.loads((a.study/'results.json').read_text())
    output=[]
    for row in results:
        arm,seed=row['arm'],row['seed']
        memory='graph' if arm=='graph' else 'memoryless' if arm=='memoryless' else 'gru'
        pair=Pair(memory)
        pair.load_state_dict(torch.load(a.study/f'{arm}_{seed}'/'checkpoint.pt',map_location='cpu',
                                        weights_only=True)['model'])
        with torch.no_grad():
            token=torch.arange(4)
            h=pair.initial(4,'cpu')
            retention=[]
            for t in range(25):
                h,belief=pair.receive(h,token,torch.full((4,),float(t==0)))
                retention.append(belief.tolist())
            # Same token, distinct public positions around the hazard. Geometry
            # is independent of danger, so action differences require own context.
            contexts=torch.tensor([[.6,0.,.1,0.,.4,.3], [.4,0.,-.1,0.,.6,.3],
                                   [.6,.1,.1,.1,.4,.2], [.6,-.1,.1,-.1,.4,.4]])
            h,belief=pair.receive(pair.initial(4,'cpu'),token,torch.ones(4))
            actions=[]
            for b in belief:
                inp=torch.cat((contexts,b.expand(4,1)),-1)
                actions.append(pair.actor(inp).softmax(-1).tolist())
            output.append(dict(arm=arm,seed=seed,
                               sender_probabilities_by_risk=pair.encoder(torch.tensor([[0.],[1.]])).softmax(-1).tolist(),
                               belief_after_single_token_and_silence=retention,
                               action_probabilities_by_token_and_context=actions))
    a.output.write_text(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
