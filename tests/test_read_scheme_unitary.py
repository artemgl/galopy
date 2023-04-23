import unittest
from tests.utils import *
import random
import numpy as np


class ReadSchemeUnitary(unittest.TestCase):

    _max_depth = 10
    _max_modes = 15
    _max_population = 20
    _max_test = 100

    def test(self):
        random.seed()
        for i in range(self._max_test):
            depth = random.randint(1, self._max_depth)

            n_state_photons = 1
            n_ancilla_photons = 0

            n_state_modes = random.randint(1, self._max_modes)
            n_ancilla_modes = random.randint(0, self._max_modes - n_state_modes)
            n_modes = n_state_modes + n_ancilla_modes

            n_population = random.randint(1, self._max_population)

            population = RandomPopulation(n_individuals=n_population, depth=depth, n_modes=n_modes,
                                          n_ancilla_modes=n_ancilla_modes, n_state_photons=n_state_photons,
                                          n_ancilla_photons=n_ancilla_photons, n_success_measurements=1, device='cpu')

            actuals = population._read_scheme_unitary()

            for j in range(n_population):
                actual = actuals[j]
                expected = construct_circuit_matrix(population[j], n_modes, depth)
                expected = np.array(expected)
                self.assertEqual(expected.shape, actual.shape)
                self.assertTrue((abs(actual - expected) < 0.0001).all(),
                                msg="Actual is\n{actual}\nExpected\n{expected}".format(actual=actual, expected=expected))


if __name__ == '__main__':
    unittest.main()
