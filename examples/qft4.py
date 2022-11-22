import numpy as np
from galopy.circuit_search import *

if __name__ == "__main__":
    # Initialize parameters
    min_probability = 1.0
    n_population = 1600
    n_offsprings = 400
    n_mutated = 1600
    n_elite = 200
    n_generations = 2000

    # Gate represented as a matrix
    matrix = np.array([[0.5,   0.5,  0.5,   0.5],
                       [0.5,  0.5j, -0.5, -0.5j],
                       [0.5,  -0.5,  0.5,  -0.5],
                       [0.5, -0.5j, -0.5,  0.5j]])
    # State modes:
    # (3)----------
    # (2)----------
    # (1)----------
    # (0)----------
    basic_states = np.array([[3],
                             [2],
                             [1],
                             [0]])
    # Create an instance of search
    search = CircuitSearch('cuda', matrix, input_basic_states=basic_states, depth=4,
                           n_ancilla_modes=0, n_ancilla_photons=0)
    # Launch the search!
    search.run(min_probability, n_generations, n_population, n_offsprings, n_mutated, n_elite)
