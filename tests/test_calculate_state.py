import unittest
from math import sqrt
from tests.utils import *


class CalculateState(unittest.TestCase):

    _max_depth = 10
    _max_modes = 5
    _max_states = 4
    _max_photons = 7
    _max_population = 5
    _max_test = 20

    def test(self):
        random.seed()
        for i in range(self._max_test):
            search = gen_random_search('cpu', self._max_depth, self._max_states, self._max_modes, self._max_photons)

            depth = search.depth
            n_modes = search.n_modes
            n_ancilla_photons = search.n_ancilla_photons
            basic_states = search.basic_states
            n_states = basic_states.shape[0]

            n_parents = random.randint(1, self._max_population)
            population = search._GeneticAlgorithm__gen_random_population(n_parents)

            p = search._GeneticAlgorithm__build_permutation_matrix()
            n, n_inv = search._GeneticAlgorithm__build_normalization_matrix(p)
            actuals = search._GeneticAlgorithm__calculate_state(population, p, n, n_inv)

            for j in range(n_parents):
                common_index = population[j, 5 * depth + 1:5 * depth + 1 + n_ancilla_photons].numpy().tolist()
                for n in range(n_states):
                    actual = actuals[j, n].coalesce()
                    actual_indices = actual.indices()[:-1].t()
                    actual_values = actual.values()

                    initial_index = common_index + basic_states[n].tolist()
                    photon_counts_per_modes = photons_to_modes(initial_index, n_modes)
                    normalization_coeff = 1.
                    for count in photon_counts_per_modes.values():
                        normalization_coeff *= factorial(count)
                    normalization_coeff = sqrt(normalization_coeff)

                    input = use_input(photon_counts_per_modes) + " / " + str(normalization_coeff)
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
