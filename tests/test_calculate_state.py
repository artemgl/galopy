import unittest
from math import sqrt
from galopy.genetic_algorithm import *
import random
from tests.utils import *


class TestGeneticAlgorithm(GeneticAlgorithm):

    def _fill_state(self, population, state_vector):
        ancilla_photons = population[:, 5 * self._depth + 1:5 * self._depth + 1 + self._n_ancilla_photons].clone()

        indices = torch.tensor([[i] * self._n_ancilla_photons for i in range(state_vector.shape[0])],
                               device=self._device).t()
        ancilla_idx = [ancilla_photons[:, i].long() for i in range(self._n_ancilla_photons)]
        idx = [indices] + ancilla_idx
        state_vector[idx] = 1.

        return state_vector

    def _calculate_fidelity_and_probability(self, state_vector):
        pass


class CalculateState(unittest.TestCase):

    _max_depth = 10
    _max_modes = 5
    _max_photons = 7
    _max_population = 5
    _max_test = 20

    def test(self):
        random.seed()
        for i in range(self._max_test):
            depth = random.randint(1, self._max_depth)
            n_modes = random.randint(2, self._max_modes)
            n_parents = random.randint(1, self._max_population)
            n_photons = random.randint(1, self._max_photons)

            search = TestGeneticAlgorithm('cpu', depth=depth, n_state_modes=0,
                                          n_ancilla_modes=n_modes, n_state_photons=0,
                                          n_ancilla_photons=n_photons, max_success_measurements=1)
            population = search._GeneticAlgorithm__gen_random_population(n_parents)

            p = search._GeneticAlgorithm__build_permutation_matrix()
            n, n_inv = search._GeneticAlgorithm__build_normalization_matrix(p)
            actuals = search._GeneticAlgorithm__calculate_state(population, p, n, n_inv)

            for j in range(n_parents):
                actual = actuals[j].coalesce()
                actual_indices = actual.indices()[:-1].t()
                actual_values = actual.values()

                initial_index = population[j, 5 * depth + 1:5 * depth + 1 + n_photons].numpy().tolist()
                photon_counts_per_modes = photons_to_modes(initial_index, n_modes)
                normalization_coeff = 1.
                for count in photon_counts_per_modes.values():
                    normalization_coeff *= factorial(count)
                normalization_coeff = sqrt(normalization_coeff)

                input = use_input_ancillas(initial_index) + "/ " + str(normalization_coeff)
                expected = run_circuit(input, construct_circuit_matrix(population[j], n_modes, depth), n_modes)

                for k in range(actual_indices.shape[0]):
                    index = actual_indices[k].numpy().tolist()
                    actual_value = actual_values[k]

                    photon_counts_per_modes = photons_to_modes(index, n_modes)
                    normalization_coeff = 1.
                    for count in photon_counts_per_modes.values():
                        normalization_coeff *= factorial(count)
                    normalization_coeff = sqrt(normalization_coeff)

                    expected_value = normalization_coeff * use_ancillas(expected, photon_counts_per_modes)
                    expected_value = complex(expected_value)

                    self.assertTrue(abs(expected_value - actual_value) < 0.0001,
                                    msg="Actual is\n{actual}\nExpected\n{expected}\nPopulation is\n{pop}"
                                    .format(actual=actual_value, expected=expected_value, pop=population))


if __name__ == '__main__':
    unittest.main()
