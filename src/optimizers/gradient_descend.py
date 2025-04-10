import torch
import numpy as np
import time
from tqdm import trange
from ..data.simulation import SimulationT, SimulationData, CoilConfigT
from ..costs.base import BaseCost
from .base import BaseOptimizer

class GDOptimizer(BaseOptimizer):
    """
    GDOptimizer is a gradient descent optimizer that minimizes a cost function using PyTorch.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 optim = torch.optim.SGD,
                 init_coil_config: CoilConfigT = None,
                 max_iter: int = 100,
                 timeout: int = 3*60-10,
                 learning_rate: float = 0.01) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.timeout = timeout
        self.optim = optim
        self.init_coil_config = init_coil_config

    def _initialize_coil_config(self):
        """Initialize the coil configuration as leaf tensors."""
        # Create independent leaf tensors with requires_grad set to True
        phase = torch.rand(8, requires_grad=True) * 2 * np.pi  # Phase values between 0 and 2π
        amplitude = torch.ones(8, requires_grad=True)  # Amplitude values between 0 and 1
        
        # Ensure they are leaf tensors (clone and detach if necessary)
        phase = phase.clone().detach().requires_grad_(True)
        amplitude = amplitude.clone().detach().requires_grad_(True)

        return phase, amplitude
        
    def optimize(self, simulation: SimulationT, init: CoilConfigT=None):
        """Optimize coil configuration using gradient descent."""
        # Initialize coil configuration with leaf tensors
        start_time = time.time()

        if init is not None:
            phase = init.phase.clone().detach().requires_grad_(True)
            amplitude = init.amplitude.clone().detach().requires_grad_(True)
        else:
            phase, amplitude = self._initialize_coil_config()

        best_coil_config = None
        last_cost = -torch.inf
        best_cost = -torch.inf

        # Set the optimizer (use leaf tensors directly)
        optimizer = self.optim([phase, amplitude], lr=self.learning_rate, maximize=True)

        pbar = trange(self.max_iter)
        j = 0

        for i in pbar:
            optimizer.zero_grad()  # Clear previous gradients

            # Create a CoilConfig-like structure using the tensors directly
            coil_config = CoilConfigT(phase=phase, amplitude=amplitude)

            # Run simulation with the current coil configuration
            simulation_data = simulation(coil_config)
            
            # Compute the cost function
            cost = self.cost_function(simulation_data)
            
            phase.grad = None
            amplitude.grad = None
            
            # Backpropagate to compute gradients
            cost.backward()
            
            # Update coil configuration parameters using gradient descent
            optimizer.step()
        
            # Check if this is the best configuration
            if cost > best_cost:
                best_cost = cost
                best_coil_config = CoilConfigT(phase=phase.detach(), amplitude=amplitude.detach())
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")

            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout:
                print(f"⏳ Timeout reached ({self.timeout} sec). Stopping optimization.")
                break
        
            # check if result is stable
            # if i > 0 and torch.abs(cost - last_cost) < 1e-2:
            #     j += 1
            #     if j > 30:
            #         print("New Inits")
            #         phase, amplitude = self._initialize_coil_config()
            #         optimizer = self.optim([phase, amplitude], lr=self.learning_rate, maximize=True)
            #         j = 0
            # else:
            #     j = 0
            # last_cost = cost
        return best_coil_config
