from math import sqrt, factorial
import unittest
import random
from galopy import *
from tests.utils import *


class ConstructOutputState(unittest.TestCase):

    _max_depth = 10
    _max_modes = 5
    _max_states = 4
    _max_photons = 7
    _max_population = 5
    _max_test = 5

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

            input_state_vector = population._construct_input_state(basic_states)
            actuals = population._construct_output_state(basic_states.shape[0], input_state_vector.to_dense())

            for j in range(n_population):
                circuit = population[j]
                common_index = circuit.initial_ancilla_state.tolist()
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
                    expected = run_circuit(input, construct_circuit_matrix(circuit, n_modes, depth), n_modes)

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
                                        msg="Actual is\n{actual}\nExpected\n{expected}\nCircuit is\n{bs_angles}\n"
                                            "{ps_angles}\n{topology}\n{initial_ancilla_states}\n{basic_states}"
                                        .format(actual=actual_value, expected=expected_value,
                                                bs_angles=circuit.bs_angles, ps_angles=circuit.ps_angles,
                                                topology=circuit.topology,
                                                initial_ancilla_states=circuit.initial_ancilla_state,
                                                basic_states=basic_states))


if __name__ == '__main__':
    unittest.main()
