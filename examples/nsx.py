from galopy.nsx_search import *

if __name__ == "__main__":
    # Initialize parameters
    min_probability = 1. / 4.
    n_population = 1600
    n_offsprings = 400
    n_mutated = 1600
    n_elite = 200
    n_generations = 2000

    # Create an instance of NS_x search
    search = NSxSearch('cuda', depth=3, n_ancilla_modes=2, n_ancilla_photons=1, n_success_measurements=1)
    # Launch the search!
    # Save result to the file result.csv
    search.run(min_probability, n_generations, n_population, n_offsprings, n_mutated, n_elite, result_file="result.csv")
