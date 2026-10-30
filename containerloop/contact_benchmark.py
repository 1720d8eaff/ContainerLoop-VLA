"""CPU MuJoCo contact benchmark for a cylinder transfer using a Cartesian gripper.

This is a new simulated test fixture, not the G1-D or a historical experiment.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from .ablation import CONFIGS


def scene(mass=.08, friction=1.):
    return f'''<mujoco model="container_transfer">
    <option timestep="0.002" gravity="0 0 -9.81"/>
    <default><geom friction="{friction} .01 .001" solref=".01 1"/><joint damping="3"/></default>
    <worldbody>
      <geom name="table" type="plane" size="1 1 .02"/>
      <body name="container" pos="0 0 .041"><freejoint name="object"/>
        <geom type="cylinder" size=".025 .04" mass="{mass}" rgba=".2 .6 .8 1"/>
      </body>
      <body name="hand" pos="0 0 .25">
        <joint name="x" type="slide" axis="1 0 0" range="-.3 .3"/>
        <joint name="y" type="slide" axis="0 1 0" range="-.3 .3"/>
        <joint name="z" type="slide" axis="0 0 1" range="-.21 .1"/>
        <geom type="box" pos="0 0 .07" size=".06 .04 .01" mass=".2"/>
        <body name="left" pos="-.055 0 0"><joint name="close_left" type="slide" axis="1 0 0" range="0 .04"/>
          <geom name="left_finger" type="box" size=".008 .025 .043" mass=".05"/></body>
        <body name="right" pos=".055 0 0"><joint name="close_right" type="slide" axis="-1 0 0" range="0 .04"/>
          <geom name="right_finger" type="box" size=".008 .025 .043" mass=".05"/></body>
      </body>
    </worldbody>
    <actuator><position joint="x" kp="800" kv="50"/><position joint="y" kp="800" kv="50"/>
      <position joint="z" kp="1200" kv="70"/><position joint="close_left" kp="800" kv="10"/>
      <position joint="close_right" kp="800" kv="10"/></actuator></mujoco>'''


def episode(condition, flags, trace_path=None):
    xml=scene(condition['mass_kg'],condition['friction'])
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    oid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'object');adr=model.jnt_qposadr[oid]
    data.qpos[adr:adr+2]=condition['offset_xy_m']
    mujoco.mj_forward(model,data)
    command=np.zeros(5);max_height=0.;contact_frames=0;trace=[]
    started=time.perf_counter()
    for step in range(2400):
        t=step*.002;pos=data.qpos[adr:adr+3].copy()
        # World-space scripted setpoints, measured object feedback only if enabled.
        target=np.array([0.,0.,-.19,0.,0.])
        if flags['feedback'] and t<1.4: target[:2]=pos[:2]
        if t>=.8: target[3:]=.034
        if t>=1.4: target[2]=-.04
        if t>=2.2: target[0]=.16
        if t>=3.0: target[2]=-.19
        if t>=3.8: target[3:]=0
        if t>=4.2: target[2]=.0
        if flags['geometry']: target=np.clip(target,[-.2,-.2,-.20,0,0],[.2,.2,.05,.035,.035])
        command=.95*command+.05*target if flags['smoothing'] else target
        data.ctrl[:]=command
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all(): raise ValueError('nonfinite simulation')
        pos=data.qpos[adr:adr+3].copy()
        max_height=max(max_height,float(pos[2]));contact_frames+=int(data.ncon>1)
        if step%10==0: trace.append([float(data.time),*pos.tolist(),*data.ctrl.tolist(),int(data.ncon)])
    final=data.qpos[adr:adr+3].copy();drift=float(np.linalg.norm(final[:2]-[.16,0]))
    success=bool(max_height>.14 and drift<.045 and .025<final[2]<.07)
    if trace_path:
        p=Path(trace_path);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['sim_time_s','object_x_m','object_y_m','object_z_m','ctrl_x_m','ctrl_y_m','ctrl_z_m','finger_left_m','finger_right_m','contacts']);writer.writerows(trace)
    return {'success':success,'max_height_m':max_height,'drift_m':drift,'final_position_m':final.tolist(),
            'contact_frames':contact_frames,'steps':2400,'wall_time_s':time.perf_counter()-started,
            'source':'mujoco_simulation','policy':'scripted_cartesian_gripper','model_sha256':hashlib.sha256(xml.encode()).hexdigest()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--conditions',default='configs/contact_conditions.json');p.add_argument('--output',default='artifacts/contact');p.add_argument('--limit',type=int)
    args=p.parse_args();conditions=json.loads(Path(args.conditions).read_text());conditions=conditions[:args.limit] if args.limit else conditions
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);records=[]
    for c in conditions:
        for method,flags in CONFIGS.items():
            record=episode(c,flags,out/f"{c['id']}_{method}.csv");record.update(condition=c,method=method);records.append(record)
    (out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary={m:{'episodes':len([r for r in records if r['method']==m]),'successes':sum(r['success'] for r in records if r['method']==m)} for m in CONFIGS}
    (out/'summary.json').write_text(json.dumps({'source':'mujoco_simulation','mujoco_version':mujoco.__version__,'results':summary},indent=2))
    print(json.dumps(summary))

if __name__=='__main__':main()
