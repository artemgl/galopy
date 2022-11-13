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
            # depth = random.randint(1, self._max_depth)
            # n_modes = random.randint(2, self._max_modes)
            # n_photons = random.randint(1, self._max_photons)

            search = gen_random_search('cpu', self._max_depth, self._max_states, self._max_modes, self._max_photons)
            # search.n_photons = 2
            # search.depth = 1
            # search.n_state_modes = 2
            # search.n_modes = 2
            # search.n_ancilla_photons = 0
            # search.n_ancilla_modes = 0
            # search.n_state_photons = 2
            # search.n_basic_states = 1
            # search.n_work_modes = 2
            # search.basic_states = torch.tensor(np.array([[1, 1]]))
            # search.matrix = torch.tensor(np.array([[1.]]))

            depth = search.depth
            n_modes = search.n_modes
            n_ancilla_photons = search.n_ancilla_photons
            basic_states = search.basic_states
            n_states = basic_states.shape[0]
            # print(search.n_photons)
            # print(search.depth)
            # print(search.n_state_modes)
            # print(search.n_modes)
            # print(search.n_ancilla_photons)
            # print(search.n_ancilla_modes)
            # print(search.n_state_photons)
            # print(search.n_basic_states)
            # print(search.n_work_modes)
            # print(search.basic_states)
            # print(search.matrix)

            # n_states = basic_states.shape[0]

            # search = GeneticAlgorithm('cpu',
            #                           depth=depth,
            #                           n_ancilla_modes=n_modes,
            #                           n_ancilla_photons=n_photons)

            n_parents = random.randint(1, self._max_population)
            population = search._GeneticAlgorithm__gen_random_population(n_parents)
            # population = torch.tensor([[20595, 25489,  4904,     1,     0,     0]], dtype=torch.int32)
            # n_parents = 1

            # print(population)

            p = search._GeneticAlgorithm__build_permutation_matrix()
            n, n_inv = search._GeneticAlgorithm__build_normalization_matrix(p)
            # print(search._GeneticAlgorithm__construct_state(population))
            # print(search._GeneticAlgorithm__read_scheme_unitary(population))
            actuals = search._GeneticAlgorithm__calculate_state(population, p, n, n_inv)

            # print(actuals)

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
