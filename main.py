from src.costs.base import BaseCost
from src.optimizers import DummyOptimizer, BayesianOptimizer, BayesianOptimizerparallel, GDOptimizer
from src.data import Simulation, CoilConfig

import numpy as np

def run(simulation: Simulation, 
        cost_function: BaseCost,
        timeout: int = 100) -> CoilConfig:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    #optimizer = BayesianOptimizerparallel(cost_function=cost_function, max_iter=timeout)
    optimizer = GDOptimizer(cost_function=cost_function, max_iter=timeout)
    best_coil_config = optimizer.optimize(simulation)
    return best_coil_config