from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator

import torch


class B1HomogeneityCost(BaseCost):
    def __init__(self) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        b1_field = self.b1_calculator(simulation_data)
        #subject = simulation_data.subject
        
        b1_field_abs = torch.abs(b1_field)
        b1_field_subject_voxels = b1_field_abs #[subject]
        return torch.mean(b1_field_subject_voxels)/torch.std(b1_field_subject_voxels, correction=0)