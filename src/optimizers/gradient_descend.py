import torch
import numpy as np
from tqdm import trange
from ..data.simulation import Simulation
from ..costs.base import BaseCost
from .base import BaseOptimizer

class GDOptimizer(BaseOptimizer):
    """
    GDOptimizer is a gradient descent optimizer that minimizes a cost function using PyTorch.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100,
                 learning_rate: float = 0.01) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def _initialize_coil_config(self):
        """Initialize the coil configuration as leaf tensors."""
        # Create independent leaf tensors with requires_grad set to True
        phase = torch.rand(8, requires_grad=True) * 2 * np.pi  # Phase values between 0 and 2π
        amplitude = torch.rand(8, requires_grad=True)  # Amplitude values between 0 and 1
        return phase, amplitude
        
    def optimize(self, simulation: Simulation):
        """Optimize coil configuration using gradient descent."""
        # Initialize coil configuration with leaf tensors
        phase, amplitude = self._initialize_coil_config()

        best_coil_config = None
        best_cost = np.inf if self.direction == "minimize" else -np.inf

        # Set the optimizer (use leaf tensors directly)
        optimizer = torch.optim.SGD([phase, amplitude], lr=self.learning_rate)

        pbar = trange(self.max_iter)
        for i in pbar:
            optimizer.zero_grad()  # Clear previous gradients

            # Create a CoilConfig-like structure using the tensors directly
            coil_config = {'phase': phase, 'amplitude': amplitude}

            # Run simulation with the current coil configuration
            simulation_data = simulation(coil_config)
            
            # Compute the cost function
            cost = self.cost_function(simulation_data)
            
            # Backpropagate to compute gradients
            cost.backward()
            
            # Update coil configuration parameters using gradient descent
            optimizer.step()

            # Check if this is the best configuration
            if (self.direction == "minimize" and cost < best_cost) or (self.direction == "maximize" and cost > best_cost):
                best_cost = cost
                best_coil_config = {'phase': phase.detach().numpy(), 'amplitude': amplitude.detach().numpy()}
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")
        
        return best_coil_config
