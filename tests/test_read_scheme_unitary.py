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
            n_parents = random.randint(1, self._max_population)

            basic_states = np.random.randint(0, self._max_modes, size=(1, 1))
            n_modes = basic_states.max() + 1

            search = GeneticAlgorithm('cpu', basic_states, np.array([[1.]]), depth=depth)
            population = search._GeneticAlgorithm__gen_random_population(n_parents)
            actuals = search._GeneticAlgorithm__read_scheme_unitary(population)

            for j in range(n_parents):
                actual = actuals[j]
                expected = construct_circuit_matrix(population[j], n_modes, depth)
                expected = np.array(expected)
                self.assertEqual(expected.shape, actual.shape)
                self.assertTrue((abs(actual - expected) < 0.0001).all(),
                                msg="Actual is\n{actual}\nExpected\n{expected}".format(actual=actual, expected=expected))


if __name__ == '__main__':
    unittest.main()
