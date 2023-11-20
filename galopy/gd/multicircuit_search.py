from galopy.gd.circuit_search import *
import numpy as np


class MultiCircuitSearch(CircuitSearch):
    def __init__(self, matrices, input_basic_states, output_basic_states=None, n_ancilla_modes=0, topology=None,
                 ancilla_state=None, measurements=None, device='cpu'):
        """
        Gradient descent searching a circuit.
        Parameters:
            matrices: Numpy array of matrices representing the desired transforms.

            input_basic_states: Basic states on which transform is performed.

            output_basic_states: Basic states which are counted as output.

            n_ancilla_modes: Number of modes in which ancilla photons are.

            topology: Topology (an arrangement of beam splitters) to be used.

            ancilla_state: Initial state of ancilla photons.

            measurements: The list of expected measurements in ancilla modes.

            device: The device on which you want to store data and perform calculations (e.g. 'cuda').
        """
        # TODO: Fail if len(matrices) == 0
        # TODO: Fail if matrices and measurements are not compatible
        if measurements is None:
            measurements = []
        super().__init__(matrices[0], input_basic_states, output_basic_states, n_ancilla_modes, topology, ancilla_state,
                         np.concatenate(measurements, axis=0), device)

        self.matrices = matrices
        self.measurements_list = measurements

    def __calculate_fidelity_and_probability(self, transforms):
        """Given transforms, get fidelity and probability for each one."""
        fidelities = []
        probabilities = []
        for i in range(len(self.measurements_list)):
            target_matrix = self.matrices[i]
            n = self.measurements_list[i].shape[0]
            transforms_slice = transforms[:n, ...]
            transforms = transforms[n:, ...]

            # Probabilities
            dot = torch.abs(transforms_slice.mul(transforms_slice.conj()))  # TODO: Optimize ?
            probability = dot.sum((1, 2)) / self.n_input_basic_states
            probability = probability.reshape(-1)

            # Fidelities
            # Formula is taken from the article:
            # https://www.researchgate.net/publication/222547674_Fidelity_of_quantum_operations
            m = target_matrix.t().conj() \
                .reshape(1, self.n_input_basic_states, self.n_output_basic_states).matmul(transforms_slice)

            a = torch.abs(m.matmul(m.transpose(-1, -2).conj()))  # TODO: Optimize ?
            a = a.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
            a = a.reshape(-1)

            b = m.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
            b = torch.abs(b.mul(b.conj()))  # TODO: Optimize ?
            b = b.reshape(-1)

            fidelity = (a + b) / self.n_input_basic_states / (self.n_input_basic_states + 1)

            probabilities.append(probability)
            fidelities.append(fidelity)

        return fidelities, probabilities

    def forward(self):
        fidelities, probabilities = self._get_fidelity_and_probability()
        fidelities = [fidelity.sum() for fidelity in fidelities]
        probabilities = [probability.sum() for probability in probabilities]

        p = probabilities[0] / (1 - probabilities[1])

        return fidelities[0] / ((1 - fidelities[1]) * p), p
