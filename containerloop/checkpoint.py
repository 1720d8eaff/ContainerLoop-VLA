"""Checkpoint-backed GR00T inference using NVIDIA's public policy API.

get_action already returns physical units; do not unnormalize its output again.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def assemble(rgb, state, instruction):
    if not rgb or not state or not instruction: raise ValueError('RGB, state and instruction required')
    for name,value in rgb.items():
        a=np.asarray(value)
        if a.ndim!=5 or a.shape[0]!=1 or a.shape[-1]!=3 or a.dtype!=np.uint8: raise ValueError(f'{name}: expected uint8 [1,T,H,W,3]')
    for name,value in state.items():
        a=np.asarray(value)
        if a.ndim!=3 or a.shape[0]!=1 or not np.isfinite(a).all(): raise ValueError(f'{name}: invalid state')
    return {'video':rgb,'state':{k:np.asarray(v,dtype=np.float32) for k,v in state.items()},'language':{'task':[[instruction]]}}


def load_checkpoint(path, embodiment, device='cpu'):
    from gr00t.policy import Gr00tPolicy
    return Gr00tPolicy(model_path=str(path),embodiment_tag=embodiment,device=device,strict=True)


def validate_actions(actions):
    if not actions: raise ValueError('empty action')
    for name,value in actions.items():
        a=np.asarray(value)
        if a.ndim!=3 or a.shape[0]!=1 or a.shape[1]==0 or not np.isfinite(a).all(): raise ValueError(f'{name}: invalid action chunk')
    return actions


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--embodiment',required=True);p.add_argument('--input',required=True);p.add_argument('--instruction',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as f:
        obs=assemble({k[6:]:f[k] for k in f.files if k.startswith('video.')},{k[6:]:f[k] for k in f.files if k.startswith('state.')},a.instruction)
    policy=load_checkpoint(a.checkpoint,a.embodiment,a.device)
    policy.reset();actions,info=policy.get_action(obs);validate_actions(actions)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(out,**actions)
    out.with_suffix('.json').write_text(json.dumps({'checkpoint':a.checkpoint,'embodiment':a.embodiment,'source':'checkpoint_inference','units':'physical units returned by Gr00tPolicy','shapes':{k:list(v.shape) for k,v in actions.items()}},indent=2))

if __name__=='__main__':main()
