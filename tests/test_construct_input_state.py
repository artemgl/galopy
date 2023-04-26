import unittest
import random
from galopy import *
from tests.utils import *


class ConstructInputState(unittest.TestCase):

    _max_depth = 10
    _max_states = 5
    _max_modes = 8
    _max_photons = 5
    _max_population = 20
    _max_test = 100

    def test(self):
        random.seed()
        for i in range(self._max_test):
            depth = random.randint(1, self._max_depth)

            n_photons = random.randint(1, self._max_photons)
            n_state_photons = random.randint(1, n_photons)
            n_ancilla_photons = n_photons - n_state_photons

            n_state_modes = random.randint(1, self._max_modes)
            n_states = random.randint(1, min(n_state_modes ** n_state_photons, self._max_states))

            basic_states = gen_random_states(n_state_modes, n_state_photons, n_states)
            basic_states = torch.tensor(basic_states)

            n_ancilla_modes = random.randint(0, self._max_modes - n_state_modes)
            n_modes = n_state_modes + n_ancilla_modes

            if n_ancilla_modes == 0 and n_ancilla_photons > 0:
                n_photons -= n_ancilla_photons
                n_ancilla_photons = 0

            n_population = random.randint(1, self._max_population)

            population = RandomPopulation(n_individuals=n_population, depth=depth, n_modes=n_modes,
                                          n_ancilla_modes=n_ancilla_modes, n_state_photons=n_state_photons,
                                          n_ancilla_photons=n_ancilla_photons, n_success_measurements=1, device='cpu')

            actuals = population._construct_input_state(basic_states)

            for j in range(n_population):
                circuit = population[j]
                common_index = circuit.initial_ancilla_state.tolist()

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
