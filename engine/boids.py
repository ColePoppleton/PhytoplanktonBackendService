import numpy as np
from scipy.spatial import KDTree
from typing import Dict, Any


class PhytoplanktonSim:
    def __init__(self, count: int, bounds: Dict[str, float]):
        self.count = count
        self.bounds = bounds
        self.positions = np.random.uniform(
            [bounds['min_lon'], bounds['min_lat']],
            [bounds['max_lon'], bounds['max_lat']],
            (count, 2)
        )
        self.velocities = np.random.uniform(-0.01, 0.01, (count, 2))

    def step(self, env_data: Dict[str, np.ndarray], params: Dict[str, Any]):
        tree = KDTree(self.positions)

        self._apply_survival_logic(env_data, params)

        if len(self.positions) > 0:
            self._apply_boid_rules(tree, params)
            self._update_physics(params)

    def _apply_survival_logic(self, env, params):
        lats = ((self.positions[:, 1] - self.bounds['min_lat']) /
                (self.bounds['max_lat'] - self.bounds['min_lat']) * (env['temp'].shape[0] - 1)).astype(int)
        lons = ((self.positions[:, 0] - self.bounds['min_lon']) /
                (self.bounds['max_lon'] - self.bounds['min_lon']) * (env['temp'].shape[1] - 1)).astype(int)

        temp_vals = env['temp'][lats, lons]
        sal_vals = env['sal'][lats, lons]

        survivable_mask = (temp_vals >= params['temp_min']) & \
                          (temp_vals <= params['temp_max']) & \
                          (sal_vals >= params['sal_min']) & \
                          (sal_vals <= params['sal_max'])

        self.positions = self.positions[survivable_mask]
        self.velocities = self.velocities[survivable_mask]

    def _apply_boid_rules(self, tree, params):
        indices = tree.query_ball_point(self.positions, params['perception'])

        cohesion_force = np.zeros_like(self.positions)
        alignment_force = np.zeros_like(self.positions)
        separation_force = np.zeros_like(self.positions)

        for i, neighbors in enumerate(indices):
            neighbors = [n for n in neighbors if n != i]
            if not neighbors: continue

            neighbor_pos = self.positions[neighbors]
            neighbor_vel = self.velocities[neighbors]

            center_mass = np.mean(neighbor_pos, axis=0)
            cohesion_force[i] = (center_mass - self.positions[i]) * params['cohesion']

            avg_vel = np.mean(neighbor_vel, axis=0)
            alignment_force[i] = (avg_vel - self.velocities[i]) * params['alignment']

            diff = self.positions[i] - neighbor_pos
            dist = np.linalg.norm(diff, axis=1)
            too_close = dist < (params['perception'] * 0.5)
            if np.any(too_close):
                separation_force[i] = np.sum(diff[too_close], axis=0) * params['separation']

        self.velocities += (cohesion_force + alignment_force + separation_force)

    def _update_physics(self, params):
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        speed = np.where(speed == 0, 1, speed)
        self.velocities = (self.velocities / speed) * np.minimum(speed.flatten(), params['max_speed'])[:, np.newaxis]

        self.positions += self.velocities

        self.positions[:, 0] = np.mod(self.positions[:, 0] + 180, 360) - 180
        self.positions[:, 1] = np.clip(self.positions[:, 1], -90, 90)