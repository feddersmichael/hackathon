from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator, SARCalculator

import torch


class B1SARCost(BaseCost):
    def __init__(self, lamb: float) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()
        self.sar_calculator = SARCalculator()
        self.lamb = lamb

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        b1_field = self.b1_calculator(simulation_data)
        sar = self.sar_calculator(simulation_data)
        subject = simulation_data.subject
        
        b1_field_abs = torch.abs(b1_field)
        b1_field_subject_voxels = b1_field_abs[subject]
        return (torch.mean(b1_field_subject_voxels)/torch.std(b1_field_subject_voxels)) - self.lamb * torch.max(sar[subject])