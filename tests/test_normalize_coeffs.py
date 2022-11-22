import unittest
from galopy.circuit_search import *
import numpy as np


# TODO: add more tests !!! Zero depth is not allowed
class NormalizeCoeffs(unittest.TestCase):
    def test(self):

        initial = torch.tensor([[ -4698, -28673,  17114,  -2234, -35668, -47181,  44352],
                                [-27332,  43438, -19370, -26779, -16354,   4237,  28544],
                                [  5540,   8659,  31154, -40704,  40692, -13689, -14071]])

        expected = torch.tensor([[31302, 7327, 8114, 2, 0,  1,  0],
                                 [ 8668, 7438, 7630, 1, 2,  1,  0],
                                 [ 5540, 8659, 4154, 0, 1,  1,  1]])

        search = CircuitSearch('cpu', np.array([[1., 0.], [0., 1.]]), np.array([[0], [1]]),
                                  n_ancilla_modes=2, n_ancilla_photons=1)
        actual = search._CircuitSearch__normalize_coeffs(initial)

        self.assertEqual(expected.shape, actual.shape)
        self.assertSequenceEqual(expected.numpy().tolist(), actual.numpy().tolist())


if __name__ == '__main__':
    unittest.main()
