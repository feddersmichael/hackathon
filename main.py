from src.costs.base import BaseCost
from src.optimizers import DummyOptimizer, BayesianOptimizerparallel, GDOptimizer, GeneticOptimizer
from src.data import CoilConfigT, SimulationT

import torch

def run(simulation: SimulationT,
        cost_function: BaseCost,
        timeout: int = 100) -> CoilConfigT:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    optimizer_bay = BayesianOptimizerparallel(cost_function=cost_function, max_iter=timeout, timeout=2*60)
    optimizer_gra = GDOptimizer(cost_function=cost_function, max_iter=timeout, optim=torch.optim.Adam, learning_rate=0.02, timeout=3*60-10)
    optimizer_gen = GeneticOptimizer(cost_function=cost_function, population_size=20, generations=timeout, mutation_rate=0.5, crossover_rate=0.6, timeout=2*60)
    
    best_coil_config = optimizer_gen.optimize(simulation)
    best_coil_config = optimizer_gra.optimize(simulation, best_coil_config)
    return best_coil_config