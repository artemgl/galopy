import unittest
from galopy.genetic_algorithm import *
import numpy as np


class BuildPermutationMatrix(unittest.TestCase):
    def test1(self):
        expected = torch.tensor([[1., 0., 0., 0.],
                                 [0., 1., 1., 0.],
                                 [0., 0., 0., 0.],
                                 [0., 0., 0., 1.]], requires_grad=False)

        search = GeneticAlgorithm('cpu', np.array([[1.]]), np.array([[0]]), n_ancilla_modes=1, n_ancilla_photons=1)
        actual = search._GeneticAlgorithm__build_permutation_matrix()

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

        search = GeneticAlgorithm('cpu', np.array([[1.]]), np.array([[0]]), n_ancilla_modes=2, n_ancilla_photons=1)
        actual = search._GeneticAlgorithm__build_permutation_matrix()

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

        search = GeneticAlgorithm('cpu', np.array([[1.]]), np.array([[0]]), n_ancilla_modes=1, n_ancilla_photons=2)
        actual = search._GeneticAlgorithm__build_permutation_matrix()

        self.assertEqual(expected.shape, actual.shape)
        self.assertSequenceEqual(expected.numpy().tolist(), actual.to_dense().numpy().tolist())


if __name__ == '__main__':
    unittest.main()
