import torch
import h5py
import os
import einops

from typing import Tuple

from torch import Tensor

from .dataclasses import SimulationRawData, SimulationData, CoilConfig


class Simulation:
    def __init__(self, 
                 path: str,
                 coil_path: str = "data/antenna/antenna.h5",
                 access_E: bool = True):
        self.path = path
        self.coil_path = coil_path
        self.access_E = access_E
        
        self.simulation_raw_data = self._load_raw_simulation_data()
        
    def _load_raw_simulation_data(self) -> SimulationRawData:
        # Load raw simulation data from path
        
        def read_field() -> Tensor:
            with (h5py.File(self.path) as f):
                re_hfield = torch.tensor(f["hfield"]["re"][0:2, :, :, :, :], dtype=torch.float32)
                im_hfield = torch.tensor(f["hfield"]["im"][0:2, :, :, :, :], dtype=torch.float32)
                hfield_ampl = torch.sqrt(re_hfield**2 + im_hfield**2)
                hfield_phase = torch.atan2(im_hfield, re_hfield)
                if self.access_E:
                    re_efield = torch.tensor(f["efield"]["re"][:], dtype=torch.float32)
                    im_efield = torch.tensor(f["efield"]["im"][:], dtype=torch.float32)
                    efield_ampl = torch.sqrt(re_efield**2 + im_efield**2)
                    efield_phase = torch.atan2(im_efield, re_efield)
                else:
                    efield_ampl = torch.zeros_like(re_hfield)
                    efield_phase = torch.zeros_like(im_hfield)
            field = torch.stack([torch.stack([efield_ampl, efield_phase], dim=0), torch.stack([hfield_ampl, hfield_phase], dim=0)],dim=0)
            return field

        def read_physical_properties() -> torch.Tensor:
            with h5py.File(self.path) as f:
                physical_properties = torch.tensor(f["input"][:], dtype=torch.float32)
            return physical_properties
        
        def read_subject_mask() -> torch.Tensor:
            with h5py.File(self.path) as f:
                subject = torch.tensor(f["subject"][:], dtype=torch.bool)
            subject = torch.max(subject, dim=-1).values
            return subject

        def read_coil_mask() -> torch.Tensor:
            with h5py.File(self.coil_path) as f:
                coil = torch.tensor(f["masks"][:], dtype=torch.bool)
            return coil
        
        def read_simulation_name() -> str:
            return os.path.basename(self.path)[:-3]

        simulation_raw_data = SimulationRawData(
            simulation_name=read_simulation_name(),
            properties=read_physical_properties(),
            field=read_field(),
            subject=read_subject_mask(),
            coil=read_coil_mask()
        )
        
        return simulation_raw_data

    def _shift_field(self, field: torch.Tensor, phase: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        """
        Shift the field calculating field_shifted = field * amplitude (e ^ (phase * 1j)) and summing over all coils.
        """
        dtype = field.dtype  # Ensure consistency with field's dtype
        if self.access_E:
            phase_shift = field[1, 1, :, :, :, :, :] + phase.view(1, 1, 1, 1, 8)
        else:
            phase_shift = field[1, 1, :, :, :, :, :] + phase.view(1, 1, 1, 1, 8)
            phase_cos = torch.cos(phase_shift) * field[1, 0, :, :, :, :, :]
            phase_sin = torch.sin(phase_shift) * field[1, 0, :, :, :, :, :]
            # phase_shift = torch.stack([torch.cos(phase_shift), torch.sin(phase_shift)], dim=0)
            # phase_shift = phase_shift * field[1, 0, :, :, :, :, :].unsqueeze(0)
            # phase_shift = torch.stack([phase_shift[0, 0, :, :, :, :] - phase_shift[1, 1, :, :, :, :],
            #                                   phase_shift[1, 0, :, :, :, :] + phase_shift[0, 1, :, :, :, :]], dim=0)
            phase_real = torch.sum((phase_cos[0, :, :, :, :] - phase_sin[1, :, :, :, :]) * amplitude.view(1, 1, 1, 8), dim=3)
            phase_im = torch.sum((phase_cos[1, :, :, :, :] + phase_sin[0, :, :, :, :]) * amplitude.view(1, 1, 1, 8), dim=3)
            field_shift = torch.stack([phase_real, phase_im], dim=0)
            # field_shift = torch.sum(amplitude.view(1, 1, 1, 1, 8) * phase_shift, dim=4)

        return field_shift


    def phase_shift(self, coil_config: CoilConfig) -> SimulationData:
        field_shifted = self._shift_field(self.simulation_raw_data.field, coil_config.phase, coil_config.amplitude)
        
        simulation_data = SimulationData(
            simulation_name=self.simulation_raw_data.simulation_name,
            properties=self.simulation_raw_data.properties,
            field=field_shifted,
            subject=self.simulation_raw_data.subject,
            coil_config=coil_config
        )
        return simulation_data
    
    def __call__(self, coil_config: CoilConfig) -> SimulationData:
        return self.phase_shift(coil_config)
