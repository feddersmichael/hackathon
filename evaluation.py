from main import run

from src.costs import B1HomogeneityCost, B1HomogeneitySARCost
from src.data import Simulation, CoilConfig
from src.utils import evaluate_coil_config

import numpy as np
import json
import time


if __name__ == "__main__":
    # Load simulation data
    start_time = time.time()
    print('Starting Optimization ...')
    # simulation = Simulation("data/simulations/children_0_tubes_2.h5")
    # simulation = Simulation("data/simulations/children_0_tubes_5.h5")
    # simulation = Simulation("data/simulations/children_1_tubes_2.h5")
    # simulation = Simulation("data/simulations/children_1_tubes_6.h5")
    # simulation = Simulation("data/simulations/children_2_tubes_3.h5")
    # simulation = Simulation("data/simulations/children_2_tubes_7.h5")
    simulation = Simulation("data/simulations/children_3_tubes_3.h5")
    # simulation = Simulation("data/simulations/children_3_tubes_10.h5")
    
    # Define cost function
    cost_function = B1HomogeneitySARCost()

    cc = CoilConfig()
    result = evaluate_coil_config(cc, simulation, cost_function)
    print(f"Default Cost: {result["default_coil_config_cost"]}")
    
    # Run optimization
    best_coil_config = run(simulation=simulation, cost_function=cost_function)
    
    # Evaluate best coil configuration
    result = evaluate_coil_config(best_coil_config, simulation, cost_function)

    # Save results to JSON file
    with open("results/results_12.json", "w") as f:
        json.dump(result, f, indent=4)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    print("Results saved to results/results.json")
    print("Finished ...")
