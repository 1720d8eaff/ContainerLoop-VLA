"""Reproducible synthetic point-mass planning example, without hardware."""
import numpy as np
from .geometry import project_delta, relative, world_target
from .optim import cem


def main():
    object_pose = np.eye(4)
    object_pose[:3, 3] = [0.4, 0.1, 0.2]
    tool = np.eye(4)
    tool[:3, 3] = [0.2, 0., 0.3]
    goal_relative = np.eye(4)
    goal_relative[:3, 3] = [0., 0., 0.1]
    goal = world_target(object_pose, goal_relative)[:3, 3]
    start = tool[:3, 3]

    def cost(actions):
        positions = start + np.cumsum(actions, axis=0)
        return np.sum((positions[-1] - goal)**2) + 0.02 * np.sum(actions**2) + 0.05 * np.sum(np.diff(actions, axis=0)**2)

    actions, score = cem(cost, 8, 3, -0.05, 0.05, seed=7)
    first = project_delta(np.r_[actions[0], [0., 0., 0.]], 0.05, 0.1)
    print('initial object-relative position:', np.round(relative(object_pose, tool)[:3, 3], 4))
    print('target world position:', np.round(goal, 4))
    print('first projected delta:', np.round(first, 4))
    print('synthetic optimization cost:', round(score, 6))


if __name__ == '__main__':
    main()
