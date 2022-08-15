# galopy
Implementation of genetic algorithm for linear optics quantum gates search

0. I use PyGAD (pip install pygad)
1. Set the search constants to feet your taste directly in galo.py (see below).
2. python galo.py
3. Observe the result!

#  Initialization here!

depth - the number of linear optics gates in the circuit (gate is one of [I, BS, PS]) 

depth = 8

ancillas - how many ancilla modes to add

ancillas = 4

other constants - read the PyGAD guide : https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html

num_generations = 2000

num_parents_mating = 10

sol_per_pop = 20

num_genes = modes * depth + ancillas * 2

init_range_low = 0

init_range_high = 90

parent_selection_type = "tournament"

keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"

mutation_percent_genes = 30