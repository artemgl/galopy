import unittest
from galopy.genetic_algorithm import *
from tests.utils import *
import random


# TODO: Move out of here ?
class TestGeneticAlgorithm(GeneticAlgorithm):

    def _fill_state(self, population, state_vector):
        pass

    def _calculate_fidelity_and_probability(self, state_vector):
        pass


class ReadSchemeUnitary(unittest.TestCase):

    _max_depth = 10
    _max_modes = 15
    _max_population = 20
    _max_test = 100

    def test(self):
        random.seed()
        for i in range(self._max_test):
            depth = random.randint(1, self._max_depth)
            n_modes = random.randint(2, self._max_modes)
            n_parents = random.randint(1, self._max_population)

            search = TestGeneticAlgorithm('cpu', depth=depth, n_state_modes=0,
                                          n_ancilla_modes=n_modes, n_state_photons=0,
                                          n_ancilla_photons=0, max_success_measurements=1)
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
