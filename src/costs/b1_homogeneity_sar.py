from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator, SARCalculator

import torch


class B1HomogeneitySARCost(BaseCost):
    def __init__(self,
                 weight: float = 100) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()
        self.sar_calculator = SARCalculator()
        self.weight = weight

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        b1_field = self.b1_calculator(simulation_data)
        sar = self.sar_calculator(simulation_data)
        
        #subject = simulation_data.subject

        b1_field_abs = torch.abs(b1_field)
        b1_field_subject_voxels = b1_field_abs #[subject]

        sar_subject_voxels = sar #[subject]

        one_over_cov = torch.mean(b1_field_subject_voxels)/torch.std(b1_field_subject_voxels)
        peak_sar = torch.max(sar_subject_voxels)
        peak_sar_sqrt = torch.sqrt(peak_sar)
        min_b1 = torch.min(b1_field_subject_voxels)
        return one_over_cov + self.weight * min_b1 / peak_sar_sqrt
