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
        b1_field_abs = torch.sqrt(b1_field[0, :, :, :].square() + b1_field[1, :, :, :].square())
        # b1_field_subject_voxels = b1_field_abs #[subject]
        amt_el = b1_field_abs.numel()
        b1_abs_sum = torch.sum(b1_field_abs)
        b1_mean = b1_abs_sum / amt_el
        return b1_abs_sum / torch.sqrt(amt_el * torch.sum((b1_field_abs - b1_mean.view(1, 1, 1)).square()))