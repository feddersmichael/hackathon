from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable
import torch


from tqdm import trange


def _sample_coil_config() -> CoilConfig:
    phase = torch.rand((8,)) * 2 * torch.pi
    amplitude = torch.rand((8,))
    return CoilConfig(phase=phase, amplitude=amplitude)


class DummyOptimizer(BaseOptimizer):
    """
    DummyOptimizer is a dummy optimizer that randomly samples coil configurations and returns the best one.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter

    def optimize(self, simulation: Simulation):
        best_coil_config = None
        best_cost = -torch.inf if self.direction == "maximize" else torch.inf
        
        pbar = trange(self.max_iter)
        for _ in pbar:
            coil_config = _sample_coil_config()
            simulation_data = simulation(coil_config)
            
            cost = self.cost_function(simulation_data)
            if (self.direction == "minimize" and cost < best_cost) or (self.direction == "maximize" and cost > best_cost):
                best_cost = cost
                best_coil_config = coil_config
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")
        
        return best_coil_config