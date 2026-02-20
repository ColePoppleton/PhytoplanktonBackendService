import numpy as np
from typing import Dict


class BoidParameterizer:
    def get_parameters(self, density_map: np.ndarray) -> Dict[str, float]:
        if density_map.ndim > 2:
            field = np.mean(density_map, axis=0)
        else:
            field = density_map
        mean_val = np.mean(field)
        std_val = np.std(field)
        grad_y, grad_x = np.gradient(field)
        avg_gradient_magnitude = np.mean(np.sqrt(grad_y ** 2 + grad_x ** 2))
        cohesion = 0.5 + (mean_val * 2.5)
        separation = 2.0 - (mean_val * 1.5)
        separation = max(0.2, separation)
        alignment = 1.0 + (avg_gradient_magnitude * 10.0)
        speed = 3.0 - (mean_val * 2.0)
        chaos = std_val * 2.0

        return {
            "cohesion": float(cohesion),
            "separation": float(separation),
            "alignment": float(alignment),
            "speed": float(speed),
            "chaos": float(chaos),
            "meta_avg_density": float(mean_val)
        }