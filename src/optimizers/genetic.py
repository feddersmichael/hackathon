from typing import List
import torch
from tqdm import trange
import random
import time

from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer
from concurrent.futures import ThreadPoolExecutor


def _sample_coil_config() -> CoilConfig:
    phase = torch.rand(8) * (2 * torch.pi)
    amplitude = torch.rand(8)
    return CoilConfig(phase=phase, amplitude=amplitude)


class GeneticOptimizer(BaseOptimizer):
    """
    GeneticOptimizer uses a genetic algorithm to optimize coil configurations.
    """
    def __init__(self, cost_function: BaseCost, timeout: int = 2*60, population_size: int = 300, generations: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7) -> None:
        super().__init__(cost_function)
        self.timeout = timeout
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _initialize_population(self) -> List[CoilConfig]:
        return [_sample_coil_config() for _ in range(self.population_size)]

    def _evaluate_population(self, simulation: Simulation, population: List[CoilConfig]):
        def evaluate(coil):
            return coil, self.cost_function(simulation(coil))

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(evaluate, population))
        return results


    def _select_parents(self, evaluated_population: List[tuple]) -> List[CoilConfig]:
        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        return [pair[0] for pair in evaluated_population[:self.population_size // 2]]

    def _crossover(self, parent1: CoilConfig, parent2: CoilConfig) -> CoilConfig:
        if torch.rand(1).item() > self.crossover_rate:
            return parent1
        mask = torch.rand(8) < 0.5
        child_phase = torch.where(mask, parent1.phase, parent2.phase)
        child_amplitude = torch.where(mask, parent1.amplitude, parent2.amplitude)
        return CoilConfig(phase=child_phase, amplitude=child_amplitude)

    def _mutate(self, coil: CoilConfig) -> CoilConfig:
        if torch.rand(1).item() < self.mutation_rate:
            coil.phase += torch.normal(mean=0, std=0.01, size=coil.phase.shape)
            coil.phase = coil.phase % (2 * torch.pi)
        if torch.rand(1).item() < self.mutation_rate:
            coil.amplitude += torch.normal(mean=0, std=0.01, size=coil.amplitude.shape)
            coil.amplitude = torch.clamp(coil.amplitude, 0, 1)
        return coil

    def optimize(self, simulation: Simulation):
        start_time = time.time()
        population = self._initialize_population()
        pbar = trange(self.generations)
        best_cost = -torch.inf
        best_param = None
        for _ in pbar:
            evaluated_population = self._evaluate_population(simulation, population)
            parents = self._select_parents(evaluated_population)
            offspring = [self._crossover(random.choice(parents), random.choice(parents)) for _ in range(self.population_size - len(parents))]
            population = parents + offspring
            population = [self._mutate(coil) for coil in population]
            max_cost = max(evaluated_population, key=lambda x: x[1])[1]
            if max_cost > best_cost:
                best_param = max(evaluated_population, key=lambda x: x[1])[0]
                best_cost = max_cost
            pbar.set_postfix_str(f"Best cost {best_cost:.2f}")
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.timeout:
                print(f"⏳ Timeout reached ({self.timeout} sec). Stopping optimization.")
                break

        return best_param