"""Achievability control: prescribed event timing, learned code and navigation.

This is not a learned gate. It asks whether sparse event communication works
with the available learned content/movement modules before blaming the task.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.distributions import Bernoulli
from . import study


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    original = study.schedule
    def schedule(arm, env, model, sample=True):
        if hasattr(env, 'schedule_override'):
            return original(arm, env, model, sample)
        send = ((env.t == 0) | ((env.risk != env.last) & env.relevant &
                               (env.pos[:, 0] < env.lion[:, 0]))).float()
        return send, Bernoulli(probs=torch.full_like(send, .5))
    study.schedule = schedule
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((a.study/'manifest.json').read_text())
    (a.output/'source.py').write_text(Path(__file__).read_text())
    (a.output/'manifest.json').write_text(json.dumps(dict(
        seeds=a.seeds or manifest['seeds'],device=a.device,base_study=str(a.study),
        iterations=manifest['iterations'],batch_size=manifest['batch_size'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    rows=[]
    for seed in (a.seeds or manifest['seeds']):
        torch.manual_seed(100+seed)
        model = study.Pair('graph').to(a.device)
        model.load_state_dict(torch.load(a.study/f'warmup_{seed}.pt', weights_only=True,
                                        map_location=a.device)['graph'])
        out=a.output/f'event_control_{seed}'; out.mkdir()
        # Reuse the fixed-schedule arm in the PPO objective, so no gradient is
        # incorrectly applied to the prescribed timing rule.
        study.train(model, 'periodic', seed, a.device, manifest['iterations'], manifest['batch_size'], out)
        evaluations=[]
        with torch.no_grad():
            for scenario in ('dynamic', 'static', 'rotated', 'late'):
                for iv in ('normal','suppress_updates','constant','shuffle','delay','drop','suppress_redundant'):
                    torch.manual_seed(987654)
                    b=study.rollout(model, 'periodic', study.Grid(1024,900001,a.device,scenario), .05,
                                    False,iv)
                    evaluations.append(dict(scenario=scenario,intervention=iv,**study.metrics(b)))
        rows.append(dict(arm='prescribed_event_control',seed=seed,evaluation=evaluations))
        (a.output/'results.json').write_text(json.dumps(rows,indent=2))
        print('EVENT CONTROL COMPLETED',seed,flush=True)


if __name__=='__main__':
    main()
