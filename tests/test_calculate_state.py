import unittest
from math import sqrt
from galopy.genetic_algorithm import *


class TestGeneticAlgorithm(GeneticAlgorithm):

    def _fill_state(self, population, state_vector):
        ancilla_photons = population[:, 5 * self._depth + 1:5 * self._depth + 1 + self._n_ancilla_photons].clone()

        indices = torch.tensor([[i] * self._n_ancilla_photons for i in range(state_vector.shape[0])],
                               device=self._device).t()
        ancilla_idx = [ancilla_photons[:, i].long() for i in range(self._n_ancilla_photons)]
        idx = [indices] + ancilla_idx + [self._n_modes - 1]
        state_vector[idx] = 1.

        return state_vector

    def _calculate_fidelity_and_probability(self, state_vector):
        pass


class CalculateState(unittest.TestCase):
    def test(self):
        # 3-mode QFT and NSx
        population = torch.tensor(
            [[30000, 12000, 0,
              21000, 0, 0,
              4500, 3526, 4500,
              1, 0,
              2, 1,
              2, 0,
              0,
              0, 1],
             [0, 0, 18000,
              0, 18000, 18000,
              2250, 6553, 2250,
              1, 0,
              1, 2,
              1, 0,
              0,
              0, 1]], requires_grad=False)

        search = TestGeneticAlgorithm('cpu', depth=3, n_state_modes=1, n_ancilla_modes=2, n_state_photons=1,
                                      n_ancilla_photons=2, max_success_measurements=1)
        p = search._GeneticAlgorithm__build_permutation_matrix()
        n, n_inv = search._GeneticAlgorithm__build_normalization_matrix(p)
        vector = search._GeneticAlgorithm__calculate_state(population, p, n, n_inv)

        self.assertTrue(abs(vector[0][0, 0, 0][0] - sqrt(2.) / 3.) < 0.0001)
        self.assertTrue(abs(vector[0][1, 1, 1][0] - sqrt(2.) / 3.) < 0.0001)
        self.assertTrue(abs(vector[0][2, 2, 2][0] - sqrt(2.) / 3.) < 0.0001)
        self.assertTrue(abs(vector[0][0, 1, 2][0] + 1 / sqrt(3.)) < 0.0001)

        self.assertTrue(abs(0.625 * sqrt(-48. + 36. * sqrt(2.)) - 0.75 * sqrt(-24. + 18. * sqrt(2)) - vector[1][0, 0, 0][0]) < 0.0001)
        self.assertTrue(abs(0.5 * sqrt(-4. + 3. * sqrt(2.)) - 1.25 * sqrt(sqrt(2.)) + 1.5 / sqrt(sqrt(2.)) - vector[1][0, 0, 1][0]) < 0.0001)
        self.assertTrue(abs(5.25 * sqrt(2.) + 0.5 * sqrt(3. - 4. / sqrt(2.)) - sqrt(6. - 4. * sqrt(2.)) - 7.5 - vector[1][0, 0, 2][0]) < 0.0001)
        self.assertTrue(abs(-0.5 * sqrt(sqrt(2.)) - 0.25 * sqrt(-4. + 3. * sqrt(2.)) + 0.25 * sqrt(-8. + 6. * sqrt(2.)) - vector[1][0, 1, 1][0]) < 0.0001)
        self.assertTrue(abs(0.5 + sqrt(-sqrt(2.) + 1.5) - sqrt(-2. * sqrt(2.) + 3.) - vector[1][0, 1, 2][0]) < 0.0001)
        # self.assertTrue(abs( - vector[1][0, 2, 2][0]) < 0.0001)



if __name__ == '__main__':
    unittest.main()
