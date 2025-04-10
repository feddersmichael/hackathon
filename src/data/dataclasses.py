from dataclasses import dataclass, field
import numpy.typing as npt
import numpy as np
import torch

@dataclass
class CoilConfig:
    """
    Stores the coil configuration data i.e. the phase and amplitude of each coil.
    """
    phase: torch.Tensor = field(default_factory=lambda: torch.zeros(8, dtype=torch.float64))
    amplitude: torch.Tensor = field(default_factory=lambda: torch.ones(8, dtype=torch.float64))
    
    def _post_init_(self):
        # No need to convert phase and amplitude to tensors as they're already tensors.
        assert self.phase.shape == self.amplitude.shape, "Phase and amplitude must have the same shape."
        assert self.phase.shape == (8,), "Phase and amplitude must have shape (8,)."

@dataclass
class SimulationData:
    """
    Stores the simulation data for a specific coil configuration.
    """
    simulation_name: str
    properties: npt.NDArray[np.float64]
    field: npt.NDArray[np.float64]
    subject: npt.NDArray[np.bool_]
    coil_config: CoilConfig
    
@dataclass
class SimulationRawData:
    """
    Stores the raw simulation data. Each coil contribution is stored separately along an additional dimension.
    """
    simulation_name: str
    properties: npt.NDArray[np.float64]
    field: npt.NDArray[np.float64]
    subject: npt.NDArray[np.bool_]
    coil: npt.NDArray[np.float64]