"""Aggregate every final seed without selecting checkpoints or discarding failures."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np

def summarize(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)) if len(a)>1 else 0.,min=float(a.min()),max=float(a.max()))

def analyze(study,output):
    study=Path(study);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    protocol=json.loads((study/'protocol.json').read_text());allrows=[];languages=[];configs=[];results={}
    for split in protocol['splits']:
        name=split['name'];results[name]={}
        for method in protocol['methods']:
            members=[];metrics=[]
            for seed in protocol['final_seeds']:
                folder=study/f'{name}_{method}_{seed}'
                data=json.loads((folder/'evaluation.json').read_text());c=json.loads((folder/'config.json').read_text());configs.append(c)
                for row in data['rows']:allrows.append(dict(split=name,method=method,seed=seed,**row))
                scalar={k:v for k,v in data['language'].items() if isinstance(v,(int,float))}
                languages.append(dict(split=name,method=method,seed=seed,**scalar))
                members.append(data);metrics.append(scalar)
            grouped={}
            for partition in ('all','train_meanings','heldout_meanings'):
                grouped[partition]={}
                for iv in ('normal','blocked','constant','shuffle','reverse','permute_vocabulary','oracle'):
                    r=[next(x for x in d['rows'] if x['partition']==partition and x['intervention']==iv) for d in members]
                    grouped[partition][iv]={k:summarize([x[k] for x in r]) for k in ('success','reward','food_correct','hazard','steps','optimal_route','safe_excess_steps')}
            lang={k:summarize([m[k] for m in metrics]) for k in metrics[0] if all(k in m for m in metrics)}
            gains=[];drops=[]
            for seed,d in zip(protocol['final_seeds'],members):
                baseline=json.loads((study/f'{name}_silent_task_{seed}'/'evaluation.json').read_text())
                get=lambda data,iv:next(r for r in data['rows'] if r['partition']=='train_meanings' and r['intervention']==iv)
                gains.append(get(d,'normal')['reward']-get(baseline,'normal')['reward'])
                drops.append(get(d,'normal')['success']-get(d,'shuffle')['success'])
            normal=grouped['train_meanings']['normal'];held=grouped['heldout_meanings']['normal']
            effective=(normal['success']['mean']>=.80 and normal['success']['min']>=.65 and np.mean(drops)>=.30 and min(gains)>0)
            reliable=(held['success']['mean']>=.95 and held['success']['min']>=.90 and grouped['all']['normal']['hazard']['max']==0)
            productive=all(lang.get(k,{'mean':0})['mean']>=threshold for k,threshold in [('semantic_exact_heldout',.90),('recombination_exact',.90),('single_token_edit_exact',.90),('positional_disentanglement',.80),('heldout_messages_unseen_in_training',.90)])
            results[name][method]=dict(task=grouped,language=lang,utility_gain_over_silent=summarize(gains),shuffle_success_drop=summarize(drops),
                                       criteria=dict(effective_communication=bool(effective),reliable_food_retrieval=bool(reliable),productive_compositional_code=bool(productive)),
                                       silent_food_ceiling=members[0]['silent_food_ceiling'])
    for filename,rows in [('task-results.csv',allrows),('language-results.csv',languages),('run-configurations.csv',configs)]:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (output/filename).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (output/'aggregate-results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({split:{method:{'id':round(d['task']['train_meanings']['normal']['success']['mean'],4),
                                  'ood':round(d['task']['heldout_meanings']['normal']['success']['mean'],4),
                                  'semantic_ood':round(d['language']['semantic_exact_heldout']['mean'],4),
                                  'recombination':round(d['language'].get('recombination_exact',{'mean':0})['mean'],4),
                                  **d['criteria']} for method,d in methods.items()} for split,methods in results.items()},indent=2))
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.study,a.output)

if __name__=='__main__':main()
