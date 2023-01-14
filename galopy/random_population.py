import torch
from math import tau
from galopy.population_old import Population


class RandomPopulation(Population):
    def __init__(self, perm_mtx, norm_mtx, inv_norm_mtx, device='cpu',
                 n_individuals=1, depth=1, n_modes=2, n_ancilla_photons=0,
                 n_ancilla_modes=0, n_success_measurements=0):
        """Generate random population."""
        self._permutation_matrix = perm_mtx
        self._normalization_matrix = norm_mtx
        self._inverted_normalization_matrix = inv_norm_mtx
        self.device = device

        self.n_individuals = n_individuals
        self.depth = depth
        self.n_modes = n_modes
        self.n_work_modes = self.n_modes
        self.n_ancilla_photons = n_ancilla_photons
        self.n_ancilla_modes = n_ancilla_modes
        self.n_success_measurements = n_success_measurements

        self._construct_data()
        self._normalize_data()

        self._precompute_extra()

    def _construct_data(self):
        self._bs_angles = torch.rand(self.n_individuals, self.depth, 2, device=self.device) * tau
        self._ps_angles = torch.rand(self.n_individuals, self.n_work_modes - 1, device=self.device) * tau

        self._topologies = torch.randint(0, self.n_work_modes, (self.n_individuals, self.depth, 2),
                                         device=self.device, dtype=torch.int8)

        if self.n_ancilla_modes > 0:
            self._initial_ancilla_states = torch.randint(0, self.n_ancilla_modes,
                                                         (self.n_individuals, self.n_ancilla_photons),
                                                         device=self.device, dtype=torch.int8)
            self._measurements = torch.randint(0, self.n_ancilla_modes,
                                               (self.n_individuals, self.n_success_measurements, self.n_ancilla_photons),
                                               device=self.device, dtype=torch.int8)
        else:
            self._initial_ancilla_states = torch.tensor([])
            self._measurements = torch.tensor([])
