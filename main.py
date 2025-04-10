from src.costs.base import BaseCost
from src.optimizers import DummyOptimizer, BayesianOptimizerparallel, GDOptimizer, GeneticOptimizer
from src.data import CoilConfig, Simulation

import torch

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
    # extract min max indices from subject
    mask = simulation.simulation_raw_data.subject
    true_indices = torch.where(mask)
    min_indices = torch.min(true_indices[0]), torch.min(true_indices[1]), torch.min(true_indices[2])
    max_indices = torch.max(true_indices[0]), torch.max(true_indices[1]), torch.max(true_indices[2])
    
    simulation.simulation_raw_data.field = simulation.simulation_raw_data.field[:,:,:,min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2],:]
    simulation.simulation_raw_data.coil = simulation.simulation_raw_data.coil[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2],:]
    simulation.simulation_raw_data.properties = simulation.simulation_raw_data.properties[:,min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]]
    
    #optimizer_bay = BayesianOptimizerparallel(cost_function=cost_function, max_iter=5000, timeout=2*60)
    optimizer_gra = GDOptimizer(cost_function=cost_function, max_iter=10000, optim=torch.optim.Adam, learning_rate=0.02, timeout=3*60-10)
    optimizer_gen = GeneticOptimizer(cost_function=cost_function, population_size=20, generations=10000, mutation_rate=0.5, crossover_rate=0.6, timeout=2*60)
    
    best_coil_config = optimizer_gen.optimize(simulation)
    best_coil_config = optimizer_gra.optimize(simulation, best_coil_config)
    return best_coil_config