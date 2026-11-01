"""CEM action-sequence planning in a MuJoCo Cartesian-gripper proxy.

Run from the repository root: python -m containerloop.contact_benchmark
The metric is grasp centering error at confirmed bilateral contact, in metres.
No bilateral grasp -> null, never a fabricated zero. Historical results are not
reused. Ground-truth simulator state is used; this is not a VLA or G1-D model.
"""
import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import mujoco
import numpy as np
from .ablation import CONFIGS
from .optim import cem


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 3
    hold_steps: int = 20
    population: int = 24
    elites: int = 6
    iterations: int = 3
    max_steps: int = 2400
    contact_confirm_steps: int = 6

    def __post_init__(self):
        values = asdict(self)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in values.values()):
            raise ValueError('planner parameters must be positive integers')
        if self.population < 2 or self.elites > self.population:
            raise ValueError('require population >= 2 and elites <= population')


# Expected object center relative to the gripper body in this proxy scene.
GRASP_OFFSET = np.array([0., 0., -.015])
PLACE_XY = np.array([.16, 0.])
GLOBAL_LO = np.array([-.3, -.3, -.21, 0., 0.])
GLOBAL_HI = np.array([.3, .3, .1, .04, .04])
METRIC_DEFINITION = ('Euclidean 3D object-center minus gripper-center-plus-offset '
                     'at first sustained bilateral finger contact; metres; null if absent')


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



def contact_sides(model, data, object_geom, finger_geoms):
    touched = set()
    for contact in data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        if object_geom in pair:
            touched.update(pair.intersection(finger_geoms))
    return len(touched) == 2


def advance(model, data, target, smoothing, count):
    """Exactly the same actuator/filter dynamics for prediction and execution."""
    for _ in range(count):
        data.ctrl[:] = .95 * data.ctrl + .05 * target if smoothing else target
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def action_limits(flags):
    lo, hi = GLOBAL_LO.copy(), GLOBAL_HI.copy()
    if flags['geometry']:
        lo[:3] = [-.2, -.2, -.21]
        hi[:3] = [.2, .2, .05]
    return lo, hi


def stage_goal(stage, initial_xy, object_pos):
    # Task subgoals constrain the search; CEM selects the actuator sequence.
    xy = object_pos[:2] if stage in ('approach', 'close') else initial_xy
    xyz = np.array([*xy, .055])
    fingers = 0.
    if stage in ('close', 'lift', 'transfer', 'lower'):
        fingers = .034
    if stage in ('lift', 'transfer'):
        xyz[2] = .205
    if stage in ('transfer', 'lower', 'release', 'retreat', 'done'):
        xyz[:2] = PLACE_XY
    if stage in ('retreat', 'done'):
        xyz[2] = .25
    return np.array([*xyz[:2], xyz[2] - .25, fingers, fingers])


class CEMAgent:
    def __init__(self, model, flags, config, seed, object_adr, hand_body):
        self.model, self.flags, self.config = model, dict(flags), config
        self.seed, self.adr, self.hand = int(seed), object_adr, hand_body
        self.predicted = mujoco.MjData(model)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.snapshot = np.empty(mujoco.mj_stateSize(model, self.state_spec))
        self.queue, self.last_action, self.previous_stage = [], None, None
        self.plan_count, self.objective_calls = 0, 0
        self.diagnostics = []

    def act(self, data, stage, goal):
        new_stage = stage != self.previous_stage
        if new_stage or (self.flags['feedback'] and not self.queue):
            self.plan(data, stage, goal)
        # Without feedback, execute one planned sequence per observed stage,
        # then hold its final command; feedback replans after each first action.
        if self.queue:
            self.last_action = self.queue.pop(0)
        self.previous_stage = stage
        return self.last_action.copy()

    def plan(self, data, stage, goal):
        cfg = self.config
        mujoco.mj_getState(self.model, data, self.snapshot, self.state_spec)
        lo, hi = action_limits(self.flags)
        spread = np.array([.025, .025, .035, .004, .004])
        center = np.clip(goal, lo, hi)
        lower, upper = np.maximum(lo, center - spread), np.minimum(hi, center + spread)
        # Open fingers stay open while approaching or releasing.
        if stage in ('approach', 'release', 'retreat', 'done'):
            lower[3:], upper[3:] = 0., 1e-5
        baseline = np.tile(np.clip(goal, lower, upper), (cfg.horizon, 1))

        def objective(sequence):
            self.objective_calls += 1
            d = self.predicted
            mujoco.mj_setState(self.model, d, self.snapshot, self.state_spec)
            mujoco.mj_forward(self.model, d)
            cost = 0.
            previous = data.ctrl.copy()
            desired_tool = np.array([goal[0], goal[1], goal[2] + .25])
            for command in sequence:
                advance(self.model, d, command, self.flags['smoothing'], cfg.hold_steps)
                pos = d.qpos[self.adr:self.adr + 3]
                tool = d.xpos[self.hand]
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    return 1e12
                cost += 400. * float(np.sum((tool - desired_tool) ** 2))
                cost += 20. * float(np.sum((d.ctrl[3:] - goal[3:]) ** 2))
                cost += .2 * float(np.sum((command - previous) ** 2))
                if stage in ('lift', 'transfer', 'lower'):
                    object_goal = desired_tool + GRASP_OFFSET
                    cost += 600. * float(np.sum((pos - object_goal) ** 2))
                if self.flags['geometry'] and stage in ('close', 'lift', 'transfer', 'lower'):
                    cost += 200. * float(np.sum((pos - tool - GRASP_OFFSET) ** 2))
                previous = command
            return cost

        baseline_cost = objective(baseline)
        sequence, score = cem(objective, cfg.horizon, 5, lower, upper,
                              population=cfg.population, elites=cfg.elites,
                              iterations=cfg.iterations, seed=self.seed + self.plan_count)
        # A feasible incumbent prevents finite-sample CEM from degrading its seed.
        if baseline_cost < score:
            sequence, score = baseline, baseline_cost
        self.queue = [row.copy() for row in (sequence[:1] if self.flags['feedback'] else sequence)]
        self.diagnostics.append({'stage': stage, 'simulation_time_s': float(data.time),
                                 'baseline_cost': baseline_cost, 'chosen_cost': score,
                                 'first_action': sequence[0].tolist()})
        self.plan_count += 1


def episode(condition, flags, trace_path=None, *, planner_config=None):
    cfg = planner_config or PlannerConfig()
    if set(flags) != {'smoothing', 'geometry', 'feedback'} or any(type(v) is not bool for v in flags.values()):
        raise ValueError('flags must contain three boolean control switches')
    mass, friction = float(condition['mass_kg']), float(condition['friction'])
    xy = np.asarray(condition['offset_xy_m'], float)
    if not np.isfinite([mass, friction]).all() or mass <= 0 or friction < 0 or xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError('invalid condition')
    xml = scene(mass, friction)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'object')
    adr = int(model.jnt_qposadr[oid])
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    # The cylinder is the sole geom of the container body.
    container = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'container')
    object_geom = int(model.body_geomadr[container])
    fingers = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
               for n in ('left_finger', 'right_finger')}
    finger_adrs = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                   for n in ('close_left', 'close_right')]
    data.qpos[adr:adr + 2] = xy
    mujoco.mj_forward(model, data)
    initial_object = data.qpos[adr:adr + 3].copy()
    agent = CEMAgent(model, flags, cfg, condition.get('seed', 42), adr, hand)
    started = time.perf_counter()
    stage, stable, released_steps = 'approach', 0, 0
    max_height, contact_frames, trace = float(initial_object[2]), 0, []
    grasp = None
    target = np.zeros(5)
    initial_tool = data.xpos[hand].copy()
    trace.append([0, float(data.time), stage, *initial_object.tolist(), *initial_tool.tolist(),
                  *data.ctrl.tolist(), int(data.ncon), int(contact_sides(model, data, object_geom, fingers)),
                  0, float(np.linalg.norm(initial_object - initial_tool - GRASP_OFFSET))])
    for step in range(cfg.max_steps):
        if step % cfg.hold_steps == 0 and stage != 'done':
            goal = stage_goal(stage, xy, data.qpos[adr:adr + 3])
            target = agent.act(data, stage, goal)
        advance(model, data, target, flags['smoothing'], 1)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError('nonfinite simulation')
        pos = data.qpos[adr:adr + 3].copy()
        tool = data.xpos[hand].copy()
        grasp_target = tool + GRASP_OFFSET
        bilateral = contact_sides(model, data, object_geom, fingers)
        contact_frames += int(bilateral)
        max_height = max(max_height, float(pos[2]))
        stable = stable + 1 if bilateral and stage in ('close', 'lift', 'transfer', 'lower') else 0
        event = False
        if grasp is None and stage == 'close' and stable >= cfg.contact_confirm_steps:
            grasp = {'grasp_drift_m': float(np.linalg.norm(pos - grasp_target)),
                     'grasp_target_position_m': grasp_target.tolist(),
                     'grasp_actual_position_m': pos.tolist(),
                     'grasp_measurement_time_s': float(data.time),
                     'grasp_measurement_step': step + 1,
                     'grasp_position_change_m': float(np.linalg.norm(pos - initial_object))}
            event = True
        next_stage = stage
        if stage == 'approach' and np.linalg.norm(tool - np.array([*xy, .055])) < .012:
            next_stage = 'close'
        elif stage == 'close' and grasp is not None:
            next_stage = 'lift'
        elif stage == 'lift' and pos[2] > .155:
            next_stage = 'transfer'
        elif stage == 'transfer' and np.linalg.norm(pos[:2] - PLACE_XY) < .015:
            next_stage = 'lower'
        elif stage == 'lower' and pos[2] < .075 and np.linalg.norm(pos[:2] - PLACE_XY) < .04:
            next_stage = 'release'
        elif stage == 'release':
            released_steps = released_steps + 1 if max(data.qpos[finger_adrs]) < .004 and not bilateral else 0
            if released_steps >= 10:
                next_stage = 'retreat'
        elif stage == 'retreat' and tool[2] > .23 and np.linalg.norm(data.qvel[:3]) < .03:
            next_stage = 'done'
        trace.append([step + 1, float(data.time), stage, *pos.tolist(), *tool.tolist(),
                      *data.ctrl.tolist(), int(data.ncon), int(bilateral), int(event),
                      float(np.linalg.norm(pos - grasp_target))])
        if next_stage != stage:
            stage = next_stage
            # Replan immediately on the next simulator step after a stage change.
            if stage != 'done':
                goal = stage_goal(stage, xy, pos)
                target = agent.act(data, stage, goal)
        # Keep simulating after completion so every trial has the same duration.
        # The last optimized retreat command is held until max_steps.
    final = data.qpos[adr:adr + 3].copy()
    drift = float(np.linalg.norm(final[:2] - PLACE_XY))
    success = bool(stage == 'done' and grasp is not None and max_height > .14
                   and drift < .045 and .025 < final[2] < .07)
    if trace_path:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['step', 'sim_time_s', 'stage', 'object_x_m', 'object_y_m', 'object_z_m',
                             'tool_x_m', 'tool_y_m', 'tool_z_m', 'ctrl_x_m', 'ctrl_y_m', 'ctrl_z_m',
                             'finger_left_m', 'finger_right_m', 'contacts', 'bilateral_contact',
                             'grasp_measurement_event', 'grasp_center_error_m'])
            writer.writerows(trace)
    record = {'success': success, 'failure_reason': None if success else 'incomplete_or_invalid_' + stage,
              'max_height_m': max_height, 'drift_m': drift, 'placement_xy_error_m': drift,
              'final_position_m': final.tolist(), 'final_stage': stage,
              'contact_frames': contact_frames, 'steps': step + 1, 'wall_time_s': time.perf_counter() - started,
              'source': 'mujoco_simulation', 'policy': 'cem_action_optimizer',
              'model_sha256': hashlib.sha256(xml.encode()).hexdigest(),
              'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'mujoco_version': mujoco.__version__, 'method_flags': dict(flags),
              'planner_config': asdict(cfg), 'cem_plans': agent.plan_count,
              'cem_objective_evaluations': agent.objective_calls, 'cem_diagnostics': agent.diagnostics,
              'grasp_drift_definition': METRIC_DEFINITION, 'grasp_reference_offset_m': GRASP_OFFSET.tolist(),
              'grasp_measurement_event': 'first_stable_bilateral_contact' if grasp else None,
              'grasp_drift_m': None, 'grasp_target_position_m': None, 'grasp_actual_position_m': None,
              'grasp_measurement_time_s': None, 'grasp_measurement_step': None, 'grasp_position_change_m': None}
    if grasp:
        record.update(grasp)
    return record


def summarize(records):
    result = {}
    for method in CONFIGS:
        rows = [r for r in records if r['method'] == method]
        measured = [r['grasp_drift_m'] for r in rows if r['grasp_drift_m'] is not None]
        result[method] = {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
                          'success_rate': sum(r['success'] for r in rows) / len(rows) if rows else None,
                          'grasp_measured_episodes': len(measured), 'grasp_missing_episodes': len(rows) - len(measured),
                          'mean_grasp_drift_m_measured': float(np.mean(measured)) if measured else None,
                          'mean_placement_xy_error_m_all_attempts': float(np.mean([r['drift_m'] for r in rows])) if rows else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', default='configs/contact_conditions.json')
    parser.add_argument('--output', default='artifacts/contact_cem')
    parser.add_argument('--limit', type=int)
    for name, value in asdict(PlannerConfig()).items():
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=value)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    cfg = PlannerConfig(**{k: getattr(args, k) for k in asdict(PlannerConfig())})
    conditions = json.loads(Path(args.conditions).read_text(encoding='utf-8'))
    if args.limit:
        conditions = conditions[:args.limit]
    if not conditions or len({c['id'] for c in conditions}) != len(conditions):
        raise ValueError('conditions must have unique IDs and be nonempty')
    if any(Path(c['id']).name != c['id'] or c['id'] in ('.', '..') or '/' in c['id'] or '\\' in c['id'] for c in conditions):
        raise ValueError('condition IDs must be plain file names')
    out = Path(args.output)
    # A new directory avoids mixing the new benchmark with historical files.
    out.mkdir(parents=True, exist_ok=False)
    records = []
    with (out / 'episodes.jsonl').open('w', encoding='utf-8') as stream:
        for condition in conditions:
            for method, flags in CONFIGS.items():
                record = episode(condition, flags, out / f"{condition['id']}_{method}.csv", planner_config=cfg)
                record.update(episode_id=f"{condition['id']}_{method}", condition=condition, method=method)
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                print(f"{record['episode_id']}: success={record['success']} grasp_drift_m={record['grasp_drift_m']}", flush=True)
    summary = {'source': 'mujoco_simulation', 'policy': 'cem_action_optimizer',
               'mujoco_version': mujoco.__version__, 'episodes': len(records), 'conditions': len(conditions),
               'grasp_drift_definition': METRIC_DEFINITION,
               'planner_config': asdict(cfg), 'results': summarize(records)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
