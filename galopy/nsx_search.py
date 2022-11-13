from galopy.genetic_algorithm import *
import numpy as np


class NSxSearch(GeneticAlgorithm):
    def __init__(self, device: str, depth=1, n_ancilla_modes=0, n_ancilla_photons=0, max_success_measurements=1):
        basic_states = np.array([[0, 0],
                                 [0, 1],
                                 [1, 1]], dtype='complex_')
        matrix = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., -1.]], dtype='complex_')
        super().__init__(device, basic_states, matrix, depth=depth, n_ancilla_modes=n_ancilla_modes,
                         n_ancilla_photons=n_ancilla_photons, max_success_measurements=max_success_measurements)

        # Work modes for NSx
        self.n_work_modes -= 1