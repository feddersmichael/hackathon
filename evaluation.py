from main import run

from src.costs import B1HomogeneityCost, B1SARCost
from src.data import Simulation
from src.utils import evaluate_coil_config

import numpy as np
import json
import time

if __name__ == "__main__":
    # Load simulation data
    start_time = time.time()    
    print(start_time)
    print('Starting Optimization ...')
    simulation = Simulation("data/simulations/children_0_tubes_2_id_19969.h5")
    
    # Define cost function
    cost_function = B1SARCost(lamb = 0.5)
    
    # Run optimization
    best_coil_config = run(simulation=simulation, cost_function=cost_function, timeout=500)
    
    # Evaluate best coil configuration
    result = evaluate_coil_config(best_coil_config, simulation, cost_function)

    # Save results to JSON file
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    print("Results saved to results.json")
    print("Finished ...")
