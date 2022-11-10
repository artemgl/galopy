import unittest
from galopy.nsx_search import *


# TODO: add more tests
class NormalizeCoeffs(unittest.TestCase):
    def test(self):

        initial = torch.tensor([[ -4698, -28673,  17114,  -2234, -35668, -47181,  40098,  44352],
                                [-27332,  43438, -19370, -26779, -16354,  37736,   4237,  28544],
                                [  5540,   8659,  31154, -40704,  40692, -13689,  40202, -14071]], requires_grad=False)

        expected = torch.tensor([[31302, 7327, 8114, 1, 2, 0,  0,  0],
                                 [ 8668, 7438, 7630, 2, 0, 0,  0,  1],
                                 [ 5540, 8659, 4154, 0, 1, 0,  0,  1]], requires_grad=False)

        search = NSxSearch('cpu', depth=1, n_ancilla_modes=2, n_ancilla_photons=2, max_success_measurements=1)
        actual = search._GeneticAlgorithm__normalize_coeffs(initial)

        self.assertEqual(expected.shape, actual.shape)
        self.assertSequenceEqual(expected.numpy().tolist(), actual.numpy().tolist())


if __name__ == '__main__':
    unittest.main()
