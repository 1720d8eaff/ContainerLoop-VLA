"""G1-D pouring simulation. Research prototype with approximate geometry."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,json,math,os,time
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from optim import cem

HERE=Path(__file__).resolve().parent
ARM=['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint']
METHODS={'direct':(False,False,False),'smoothing':(True,False,False),'geometry':(False,True,False),'feedback':(False,False,True),'full':(True,True,True)}

def run(job):
 c,method=job; smoothing,geometry,feedback=METHODS[method];m=mujoco.MjModel.from_xml_path(str(HERE/'g1d_pour_prototype.xml'));d=mujoco.MjData(m);ik=mujoco.MjData(m);pred=mujoco.MjData(m)
 def id(kind,name):
  value=mujoco.mj_name2id(m,kind,name)
  if value<0:raise ValueError('missing '+name)
  return value
 joints=[id(mujoco.mjtObj.mjOBJ_JOINT,n) for n in ARM];qa=m.jnt_qposadr[joints];limits=m.jnt_range[joints];aa=[id(mujoco.mjtObj.mjOBJ_ACTUATOR,'act_'+n) for n in ARM]
 fingers=[id(mujoco.mjtObj.mjOBJ_ACTUATOR,'act_'+n+'_finger_slide') for n in ('left','right')]
 finger_geoms={id(mujoco.mjtObj.mjOBJ_GEOM,n+'_pad') for n in ('left','right')}
 site=id(mujoco.mjtObj.mjOBJ_SITE,'grip_site');bc=id(mujoco.mjtObj.mjOBJ_BODY,'beaker');ball=id(mujoco.mjtObj.mjOBJ_BODY,'ball');flask=id(mujoco.mjtObj.mjOBJ_BODY,'flask');ball_geom=id(mujoco.mjtObj.mjOBJ_GEOM,'ball_geom')
 joint=id(mujoco.mjtObj.mjOBJ_JOINT,'beaker_free'); q0=int(m.jnt_qposadr[joint]);bj=id(mujoco.mjtObj.mjOBJ_JOINT,'ball_free');bq=int(m.jnt_qposadr[bj])
 d.qpos[q0:q0+2]+=c['offset_xy_m'];d.qpos[bq:bq+2]+=c['offset_xy_m'];d.qpos[bq:bq+2]+=c['ball_offset_xy_m']
 m.body_mass[bc]*=c['beaker_mass_scale'];m.body_inertia[bc]*=c['beaker_mass_scale']
 for i in range(m.ngeom):
  name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
  if name.startswith('beaker_') or name.endswith('_pad'):m.geom_friction[i,0]=c['grasp_friction']
 mujoco.mj_setConst(m,d)
 def solve(target,rot,start):
  def fun(q):
   ik.qpos[qa]=q;mujoco.mj_forward(m,ik);rr=Rotation.from_matrix(ik.site_xmat[site].reshape(3,3));return np.r_[8*(ik.site_xpos[site]-target),(rot*rr.inv()).as_rotvec(),.02*(q-start)]
  res=least_squares(fun,np.clip(start,limits[:,0]+1e-5,limits[:,1]-1e-5),bounds=(limits[:,0],limits[:,1]),max_nfev=70)
  return res.x,float(np.linalg.norm(fun(res.x)[:3])/8)
 desired=np.array([.38,.18,.78])+np.array([*c['offset_xy_m'],0]);q,_=solve(desired,Rotation.identity(),np.zeros(7));d.qpos[qa]=q;d.ctrl[aa]=q;mujoco.mj_forward(m,d)
 dt=float(m.opt.timestep);history=[];bilateral_steps=0;grasp_drift=None;max_beaker_z=float(d.xpos[bc,2]);above_mouth=False;previous_ball=None;total_steps=0;mouth_crossing=None
 spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,spec));calls=0;plan_cost=None
 def bilateral():
  touch=set()
  for contact in d.contact:
   pair={int(contact.geom1),int(contact.geom2)}
   for fg in finger_geoms & pair:
    other=int(contact.geom2) if int(contact.geom1)==fg else int(contact.geom1)
    if m.geom_bodyid[other]==bc:touch.add(fg)
  return len(touch)==2
 def step(qtarget,grip,n,label):
  nonlocal bilateral_steps,grasp_drift,max_beaker_z,above_mouth,previous_ball,total_steps,mouth_crossing
  for i in range(n):
   if smoothing: d.ctrl[aa]=.8*d.ctrl[aa]+.2*qtarget
   else:d.ctrl[aa]=qtarget
   d.ctrl[fingers]=grip;mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
   touching=bilateral();bilateral_steps=bilateral_steps+1 if touching else 0
   if grasp_drift is None and label=='grasp' and bilateral_steps>=6:
    grasp_drift=float(np.linalg.norm(d.site_xpos[site]-(d.xpos[bc]+np.array([0,0,.03]))))
   max_beaker_z=max(max_beaker_z,float(d.xpos[bc,2]));bp=d.xpos[ball].copy();fp=d.xpos[flask].copy()
   if previous_ball is not None and previous_ball[2] > fp[2]+.12 >= bp[2]:
    fraction=(previous_ball[2]-(fp[2]+.12))/(previous_ball[2]-bp[2]);cross=previous_ball+fraction*(bp-previous_ball)
    if np.linalg.norm(cross[:2]-fp[:2]) <= .022-.009:
     above_mouth=True;mouth_crossing={'time_s':float(d.time),'ball_center_xyz_m':cross.tolist()}
   previous_ball=bp.copy();total_steps+=1
   if True:history.append({'step':int(round(d.time/dt)),'stage':label,'time_s':float(d.time),'ball_xyz_m':bp.tolist(),'beaker_xyz_m':d.xpos[bc].tolist(),'grip_xyz_m':d.site_xpos[site].tolist(),'bilateral_contact':touching,'beaker_quat_wxyz':d.xquat[bc].tolist(),'ball_velocity_m_s':d.qvel[m.jnt_dofadr[bj]:m.jnt_dofadr[bj]+3].tolist(),'beaker_relative_gripper_m':(d.site_xmat[site].reshape(3,3).T @ (d.xpos[bc]-d.site_xpos[site])).tolist(),'n_contacts':int(d.ncon)})
  if not np.isfinite(d.qpos).all():raise RuntimeError('non-finite simulation')
 def move(label,target,rot,duration,grip):
  nonlocal q
  target=np.asarray(target,float);nq,error=solve(target,rot,q);start=q.copy();n=round(duration/dt)
  for t in range(n):step(start+(nq-start)*(t+1)/n,grip,1,label)
  q=nq
  return error
 step(q,0,300,'settle')
 obj=d.xpos[bc].copy();approach=obj+np.array([0,0,.035]);approach[:2]=obj[:2] if feedback else desired[:2]
 # CEM selects a 2-D grasp-center correction using physics rollouts from the current state.
 mujoco.mj_getState(m,d,state,spec)
 def objective(x):
  nonlocal calls
  calls+=1;delta=x[0];pgoal=approach+np.r_[delta,0];candidate,_=solve(pgoal,Rotation.identity(),q)
  mujoco.mj_setState(m,pred,state,spec);mujoco.mj_forward(m,pred);pred.ctrl[aa]=candidate;pred.ctrl[fingers]=0
  for _ in range(60):mujoco.mj_step(m,pred)
  mujoco.mj_forward(m,pred);err=np.linalg.norm(pred.site_xpos[site]-pgoal);movement=np.linalg.norm(pred.xpos[bc]-obj)
  return float(err**2+movement**2*(8 if geometry else 2)+np.dot(delta,delta)*(.4 if geometry else .1))
 sequence,plan_cost=cem(objective,1,2,[-.004,-.004],[.004,.004],population=4,elites=2,iterations=2,seed=c['seed']);approach[:2]+=sequence[0]
 move('approach',approach,Rotation.identity(),2.,0);step(q,.017,600,'grasp')
 move('lift',[obj[0],obj[1],.84],Rotation.identity(),2.,.017)
 # Full controller uses the observed post-lift state and evaluates future
 # complete transfer/tilt candidate trajectories in a separate simulator.
 pour_target=[.48,.12,.84];pour_planner=None
 if geometry and feedback and smoothing:
  mujoco.mj_getState(m,d,state,spec)
  original_q=q.copy();plan_records=[]
  def evaluate_pour(offset):
   pgoal=np.array([.48,.12,.84])+np.array([offset[0,0],offset[0,1],0.]);qprev=original_q.copy();waypoints=[]
   for angle,dur in [(0,2.),(30,.7),(60,.7),(90,.7),(120,.7),(150,.7)]:
    rotation=Rotation.identity() if angle==0 else Rotation.from_euler('x',angle,degrees=True)
    qnext,error=solve(pgoal,rotation,qprev);waypoints.append((qprev.copy(),qnext.copy(),round(dur/dt)));qprev=qnext
   mujoco.mj_setState(m,pred,state,spec);mujoco.mj_forward(m,pred);prior=None;crossed=False;min_mouth=10.
   fp=d.xpos[flask].copy();lastq=None
   for start_q,end_q,n in waypoints:
    for t in range(n):
     target=start_q+(end_q-start_q)*(t+1)/n
     pred.ctrl[aa]=.8*pred.ctrl[aa]+.2*target;pred.ctrl[fingers]=.017
     mujoco.mj_step(m,pred);mujoco.mj_forward(m,pred);bp=pred.xpos[ball].copy()
     if abs(bp[2]-(fp[2]+.12))<.015:min_mouth=min(min_mouth,float(np.linalg.norm(bp[:2]-fp[:2])))
     if prior is not None and prior[2]>fp[2]+.12>=bp[2]:
      u=(prior[2]-(fp[2]+.12))/(prior[2]-bp[2]);point=prior+u*(bp-prior)
      if np.linalg.norm(point[:2]-fp[:2])<=.013:crossed=True
     prior=bp
    lastq=end_q
   for _ in range(800):
    pred.ctrl[aa]=.8*pred.ctrl[aa]+.2*lastq;pred.ctrl[fingers]=.017
    mujoco.mj_step(m,pred);mujoco.mj_forward(m,pred);bp=pred.xpos[ball].copy()
    if abs(bp[2]-(fp[2]+.12))<.015:min_mouth=min(min_mouth,float(np.linalg.norm(bp[:2]-fp[:2])))
    if prior is not None and prior[2]>fp[2]+.12>=bp[2]:
     u=(prior[2]-(fp[2]+.12))/(prior[2]-bp[2]);point=prior+u*(bp-prior)
     if np.linalg.norm(point[:2]-fp[:2])<=.013:crossed=True
    prior=bp
   z=float(bp[2]-fp[2]);rad=.06-(.06-.022)*z/.12
   inside=bool(.009<=z<=.111 and np.linalg.norm(bp[:2]-fp[:2])<=max(0.,rad-.009))
   speed=float(np.linalg.norm(pred.qvel[m.jnt_dofadr[bj]:m.jnt_dofadr[bj]+3]))
   success=bool(crossed and inside and speed<.08)
   cost=(0. if success else 10.)+min(min_mouth,1.)+min(float(np.linalg.norm(bp-fp)),2.)*.01
   plan_records.append({'target_xyz_m':pgoal.tolist(),'predicted_success':success,'crossed_mouth':crossed,'inside':inside,'cost':cost})
   return cost
  seq,score=cem(evaluate_pour,1,2,[-.012,-.012],[.012,.012],population=4,elites=2,iterations=2,seed=c['seed']+10000)
  pour_target=(np.array([.48,.12,.84])+np.r_[seq[0],0]).tolist()
  pour_planner={'source':'simulator_ground_truth_forward_rollout','candidate_count':len(plan_records),'selected_cost':score,'candidates':plan_records}
 move('transfer',pour_target,Rotation.identity(),2.,.017)
 errors=[]
 for angle in [30,60,90,120,150]:errors.append(move('pour_'+str(angle),pour_target,Rotation.from_euler('x',angle,degrees=True),.7,.017))
 step(q,.017,800,'settle_pour')
 bp=d.xpos[ball].copy();fp=d.xpos[flask].copy();z=bp[2]-fp[2];rad=.06-(.06-.022)*z/.12
 inside=bool(.009<=z<=.111 and np.linalg.norm(bp[:2]-fp[:2]) <= max(0.,rad-.009))
 speed=float(np.linalg.norm(d.qvel[m.jnt_dofadr[bj]:m.jnt_dofadr[bj]+3]));success=bool(inside and speed<.08 and above_mouth)
 return {'episode_id':c['id']+'_'+method,'method':method,'condition':c,'success':success,'inside_flask_final':inside,'crossed_mouth_region':above_mouth,'mouth_crossing':mouth_crossing,'ball_final_xyz_m':bp.tolist(),'ball_final_speed_m_s':speed,'flask_center_xyz_m':fp.tolist(),'beaker_max_z_m':max_beaker_z,'grasp_drift_m':grasp_drift,'cem_evaluations':calls,'cem_selected_grasp_offset_xy_m':sequence[0].tolist(),'cem_cost':plan_cost,'ik_pour_position_errors_m':errors,'simulation_seconds':float(d.time),'pour_target_xyz_m':pour_target,'pour_planner':pour_planner,'trajectory':history,'model':'G1-D URDF kinematics/inertia; approximate gripper and vessel collisions; robot link meshes absent','method_flags':{'smoothing':smoothing,'geometry':geometry,'feedback':feedback}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=48);p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 (out/'traces').mkdir()
 conditions=[]
 for mass in [.875,1.125,1.375]:
  for friction in [1.0,1.4]:
   for off in [[.002,-.002],[-.002,.002]]:
    k=len(conditions);conditions.append({'id':f'v{k:02}', 'beaker_mass_scale':mass,'grasp_friction':friction,'offset_xy_m':off,'ball_offset_xy_m':[.001,-.001],'seed':1000+k})
 conditions=conditions[:a.limit];jobs=[(c,m) for c in conditions for m in METHODS];results=[];start=time.perf_counter()
 with (out/'episodes.jsonl').open('w',encoding='utf-8') as stream, concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  for result in pool.map(run,jobs):
   trace=result.pop('trajectory'); tracefile='traces/'+result['episode_id']+'.jsonl'; result['trace_file']=tracefile
   with (out/tracefile).open('w',encoding='utf-8') as tf:
    for frame in trace:tf.write(json.dumps(frame,allow_nan=False,separators=(',',':'))+'\n')
   result['trace_steps']=len(trace);results.append(result);stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush();print(result['episode_id'],result['success'],'ball',result['ball_final_xyz_m'],flush=True)
 stats={}
 for method in METHODS:
  group=[r for r in results if r['method']==method];valid=[r['grasp_drift_m'] for r in group if r['grasp_drift_m'] is not None]
  stats[method]={'episodes':len(group),'successes':sum(r['success'] for r in group),'success_rate':sum(r['success'] for r in group)/len(group),'grasp_measured_episodes':len(valid),'mean_grasp_drift_m_measured':float(np.mean(valid)) if valid else None}
 summary={'source':'new_mujoco_simulation','task':'G1-D simplified gripper pours ball from beaker into conical flask','model_limitations':'G1-D URDF kinematics and inertias; missing robot mesh and robot link collision geometry; approximate gripper and vessels; fixed-base with gravity compensation; not hardware or VLA','success_definition':'ball center crosses downward through flask mouth with radius clearance, finishes within inner cone at z 9–111 mm with velocity below 0.08 m/s','episodes':len(results),'conditions':len(conditions),'model_sha256':hashlib.sha256((HERE/'g1d_pour_prototype.xml').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'mujoco_version':mujoco.__version__,'numpy_version':np.__version__,'elapsed_wall_s':time.perf_counter()-start,'methods':'smoothing=joint target filtering; geometry=larger object disturbance penalty during CEM; feedback=uses observed beaker position at grasp planning; all use same IK and grasp CEM; full uses CEM to optimize the pour target with complete future physics rollouts from the post-lift simulator state' ,'results':stats}
 (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
