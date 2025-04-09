from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

import numpy as np
import time
from skopt import gp_minimize
from skopt.space import Real
from joblib import Parallel, delayed  # For parallel execution

class BayesianOptimizer(BaseOptimizer):
    """
    MyOptimizer using Bayesian Optimization (via Gaussian Process).
    Instead of random search, it models the cost function and suggests better candidates iteratively.
    """
    def __init__(self, cost_function: BaseCost, max_iter: int = 100) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter

        # Define the search space: 8 phase values (0 to 2π) and 8 amplitude values (0 to 1)
        self.search_space = [Real(0, 2 * np.pi, name=f"phase_{i}") for i in range(8)] + \
                            [Real(0, 1, name=f"amplitude_{i}") for i in range(8)]
        
    def _objective_function(self, params):
        """Wraps simulation and cost evaluation for Bayesian optimization."""
        # Extract phases and amplitudes
        phase = np.array(params[:8])
        amplitude = np.array(params[8:])
        
        # Create coil configuration
        coil_config = CoilConfig(phase=phase, amplitude=amplitude)
        
        # Run simulation
        simulation_data = self.simulation(coil_config)
        
        # Compute cost
        return -self.cost_function(simulation_data)
        
    def optimize(self, simulation: Simulation):
        """Optimize coil configuration using Bayesian Optimization."""
        self.simulation = simulation  # Store simulation for use in objective function

        # Perform Bayesian Optimization
        result = gp_minimize(self._objective_function, self.search_space, n_calls=self.max_iter, random_state=42)
        
        # Extract best coil configuration
        best_params = result.x
        best_phase = np.array(best_params[:8])
        best_amplitude = np.array(best_params[8:])
        best_coil_config = CoilConfig(phase=best_phase, amplitude=best_amplitude)
        
        return best_coil_config


class BayesianOptimizerparallel(BaseOptimizer):
    """
    Bayesian Optimization with parallel cost function evaluations.
    """
    def __init__(self, cost_function: BaseCost, max_iter: int = 100, timeout: int = 5*60-10, num_workers: int = -1) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.num_workers = num_workers  # Number of parallel workers
        self.timeout = timeout  # Timeout in seconds

        # Define the search space: 8 phase values (0 to 2π) and 8 amplitude values (0 to 1)
        self.search_space = [Real(0, 2 * np.pi, name=f"phase_{i}") for i in range(8)] + \
                            [Real(0, 1, name=f"amplitude_{i}") for i in range(8)]
    
    def _evaluate_single(self, params):
        """Evaluate a single candidate solution."""
        phase = np.array(params[:8])
        amplitude = np.array(params[8:])
        coil_config = CoilConfig(phase=phase, amplitude=amplitude)
        simulation_data = self.simulation(coil_config)
        return -self.cost_function(simulation_data)  # Negative for maximization
    
    # def optimize(self, simulation: Simulation):
    #     """Perform Bayesian Optimization with parallelized evaluations."""
    #     self.simulation = simulation  # Store simulation for use in objective function

    #     # Run Bayesian optimization
    #     result = gp_minimize(
    #         self._evaluate_single,  # No batch processing needed
    #         self.search_space,
    #         n_calls=self.max_iter,
    #         n_restarts_optimizer=40,
    #         acq_optimizer='lbfgs',
    #         n_jobs=self.num_workers,  # Parallel execution
    #         random_state=42
    #     )
        
    #     # Extract best coil configuration
    #     best_params = result.x
    #     best_phase = np.array(best_params[:8])
    #     best_amplitude = np.array(best_params[8:])
    #     best_coil_config = CoilConfig(phase=best_phase, amplitude=best_amplitude)
        
    #     return best_coil_config

    def optimize(self, simulation: Simulation):
        """Perform Bayesian Optimization with a time limit."""
        self.simulation = simulation  # Store simulation for use in objective function
        start_time = time.time()

        best_x = None
        best_y = np.inf if self.direction == "minimize" else -np.inf

        def _callback(res):
            """Callback function to stop when timeout is reached."""
            nonlocal best_x, best_y
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout:
                print(f"⏳ Timeout reached ({self.timeout} sec). Stopping optimization.")
                return True  # Stop optimization
            
            # Store the best found result so far
            if res.fun < best_y:  # Minimization
                best_y = res.fun
                best_x = res.x

            return False  # Keep running until timeout or finish

        result = gp_minimize(
            self._evaluate_single,  # Objective function for Bayesian Optimization
            self.search_space,
            n_calls=self.max_iter,
            n_jobs=self.num_workers,  # Parallel execution
            random_state=None,
            callback=_callback  # Callback to check for timeout
        )
        
        # If optimization stopped early, return the best found so far
        best_params = best_x if best_x is not None else result.x
        best_phase = np.array(best_params[:8])
        best_amplitude = np.array(best_params[8:])
        best_coil_config = CoilConfig(phase=best_phase, amplitude=best_amplitude)
        
        return best_coil_config
