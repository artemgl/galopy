import unittest
from tests.utils import *


class ConstructState(unittest.TestCase):

    _max_depth = 10
    _max_states = 5
    _max_modes = 10
    _max_photons = 6
    _max_population = 20
    _max_test = 100

    def test(self):
        random.seed()
        for i in range(self._max_test):
            search = gen_random_search('cpu', self._max_depth, self._max_states, self._max_modes, self._max_photons)
            depth = search.depth
            n_ancilla_photons = search.n_ancilla_photons
            n_modes = search.n_modes
            basic_states = search.input_basic_states
            n_states = basic_states.shape[0]

            n_parents = random.randint(1, self._max_population)
            population = search._CircuitSearch__gen_random_population(n_parents)
            actuals = search._CircuitSearch__construct_state(population)

            for j in range(n_parents):
                common_index = population[j, 5 * depth:5 * depth + n_ancilla_photons].numpy().tolist()

                for k in range(n_states):
                    actual = actuals[j, k].coalesce()
                    actual_indices = actual.indices()[:-1].t()
                    actual_values = actual.values()

                    for n in range(actual_indices.shape[0]):
                        actual_index = actual_indices[n].numpy().tolist()
                        actual_value = actual_values[n]

                        index = common_index + basic_states[k].tolist()
                        input = use_input(photons_to_modes(index, n_modes))

                        output = run_circuit(input, np.identity(n_modes), n_modes)

                        photon_counts_per_modes = photons_to_modes(actual_index, n_modes)
                        expected_value = use_ancillas(output, photon_counts_per_modes)
                        expected_value = complex(expected_value)

                        self.assertTrue(abs(expected_value - actual_value) < 0.0001,
                                        msg="Actual is\n{actual}\nExpected\n{expected}\nPopulation is\n{pop}"
                                        .format(actual=actual_value, expected=expected_value, pop=population))


if __name__ == '__main__':
    unittest.main()
