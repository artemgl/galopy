# galopy
Implementation of genetic algorithm for linear optics quantum gates search

0. I use PyGAD (pip install pygad)
1. Set the search constants to feet your taste directly in galo.py (see below).
2. python galo.py
3. Observe the result!

# Basic example
Search CZ gate with 3 beam splitters, 2 ancilla modes and 0 ancilla photons
```python
import numpy as np
from galopy.circuit_search import *

# Initialize parameters
min_probability = 1. / 9.
n_population = 2000
n_offsprings = 400
n_mutated = 2000
n_elite = 800
n_generations = 200

# Gate represented as a matrix
matrix = np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., -1.]])
# State modes:
# (3)----------
# (2)----------
# (1)----------
# (0)----------
basic_states = np.array([[0, 2],
                         [0, 3],
                         [1, 2],
                         [1, 3]])
# Create an instance of search
search = CircuitSearch('cuda', matrix, input_basic_states=basic_states, depth=3,
                       n_ancilla_modes=2, n_ancilla_photons=0)
# Launch the search!
search.run(min_probability, n_generations, n_population, n_offsprings, n_mutated, n_elite)
```
Output:
```
Circuit:
         Element   Angle Modes
0  Phase shifter  216.97     1
1  Beam splitter   53.86  4, 1
2  Phase shifter  299.16     1
3  Phase shifter  126.44     3
4  Beam splitter   54.93  5, 3
5  Phase shifter  230.04     3
6  Phase shifter  127.93     0
7  Beam splitter   53.63  2, 0
8  Phase shifter  355.89     0
Fidelity:  0.9991157650947571
Probability:  0.11715999245643616
Processed 156 generations in 11.00 seconds
```

#  Initialization here!

depth - the number of linear optics gates in the circuit (gate is one of [I, BS, PS]) 

depth = 8

ancillas - how many ancilla modes to add

ancillas = 4

other constants - read the PyGAD guide : https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html

