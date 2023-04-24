import unittest
from galopy.circuit_search import *


class BuildPermutationMatrix(unittest.TestCase):
    def test1(self):
        expected = torch.tensor([[1., 0., 0., 0.],
                                 [0., 1., 1., 0.],
                                 [0., 0., 0., 0.],
                                 [0., 0., 0., 1.]])

        population = RandomPopulation(n_individuals=1, depth=1, n_modes=2, n_ancilla_modes=1, n_state_photons=2,
                                      n_ancilla_photons=0, n_success_measurements=1, device='cpu')
        actual = population._permutation_matrix

        self.assertEqual(expected.shape, actual.shape)
        self.assertSequenceEqual(expected.numpy().tolist(), actual.to_dense().numpy().tolist())

    def test2(self):
        expected = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 1., 0., 1., 0., 0., 0., 0., 0.],
                                 [0., 0., 1., 0., 0., 0., 1., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 1., 0., 1., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)

        population = RandomPopulation(n_individuals=1, depth=1, n_modes=3, n_ancilla_modes=0, n_state_photons=2,
                                      n_ancilla_photons=0, n_success_measurements=1, device='cpu')
        actual = population._permutation_matrix

        self.assertEqual(expected.shape, actual.shape)
        self.assertSequenceEqual(expected.numpy().tolist(), actual.to_dense().numpy().tolist())

    def test3(self):
        expected = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 1., 1., 0., 1., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 1., 0., 1., 1., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 1.]], requires_grad=False)

        population = RandomPopulation(n_individuals=1, depth=1, n_modes=2, n_ancilla_modes=0, n_state_photons=3,
                                      n_ancilla_photons=0, n_success_measurements=1, device='cpu')
        actual = population._permutation_matrix

        self.assertEqual(expected.shape, actual.shape)
        self.assertSequenceEqual(expected.numpy().tolist(), actual.to_dense().numpy().tolist())


if __name__ == '__main__':
    unittest.main()
