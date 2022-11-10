from genetic_algorithm import *


class NSxSearch(GeneticAlgorithm):
    def _fill_state(self, population, state_vector):
        """Assign the initial state in Dirac form."""
        # Read initial state for ancilla photons
        # TODO: Optimize
        ancilla_photons = population[:, 5 * self._depth + 1:5 * self._depth + 1 + self._n_ancilla_photons].clone()

        # Fill the state for NSx search
        # TODO: indices move to outer scope ?
        indices = torch.tensor([[i] * self._n_ancilla_photons for i in range(state_vector.shape[0])],
                               device=self._device).t()
        ancilla_idx = [ancilla_photons[:, i].long() for i in range(self._n_ancilla_photons)]
        print(ancilla_idx)
        idx = [indices] + ancilla_idx + [self._n_work_modes - 1, self._n_work_modes - 1]
        state_vector[idx] = 1.
        idx = [indices] + ancilla_idx + [self._n_work_modes - 2, self._n_work_modes - 1]
        state_vector[idx] = 1.
        idx = [indices] + ancilla_idx + [self._n_work_modes - 2, self._n_work_modes - 2]
        state_vector[idx] = 1.

        return state_vector

    def _calculate_fidelity_and_probability(self, state_vector):
        pass
