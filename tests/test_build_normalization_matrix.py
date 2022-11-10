import unittest
from galopy.nsx_search import *
from math import sqrt


class BuildNormalizationMatrix(unittest.TestCase):
    def test1(self):
        expected = torch.tensor([[1. / sqrt(2.), 0., 0.,            0.],
                                 [           0., 1., 0.,            0.],
                                 [           0., 0., 0.,            0.],
                                 [           0., 0., 0., 1. / sqrt(2.)]], requires_grad=False)

        expected_inv = torch.tensor([[sqrt(2.), 0., 0.,        0.],
                                     [      0., 1., 0.,        0.],
                                     [      0., 0., 0.,        0.],
                                     [      0., 0., 0., sqrt(2.)]], requires_grad=False)

        search = NSxSearch('cpu', depth=1, n_ancilla_modes=0, n_ancilla_photons=0, max_success_measurements=1)
        actual, actual_inv = search._GeneticAlgorithm__build_normalization_matrix(
            search._GeneticAlgorithm__build_permutation_matrix())

        self.assertEqual(expected.shape, actual.shape)
        self.assertSequenceEqual(expected.numpy().tolist(), actual.to_dense().numpy().tolist())

        self.assertEqual(expected_inv.shape, actual_inv.shape)
        self.assertSequenceEqual(expected_inv.numpy().tolist(), actual_inv.to_dense().numpy().tolist())

    def test2(self):
        expected = torch.tensor([[1. / sqrt(2.), 0., 0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0., 1., 0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0., 0., 1., 0.,            0., 0., 0., 0.,            0.],
                                 [           0., 0., 0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0., 0., 0., 0., 1. / sqrt(2.), 0., 0., 0.,            0.],
                                 [           0., 0., 0., 0.,            0., 1., 0., 0.,            0.],
                                 [           0., 0., 0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0., 0., 0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0., 0., 0., 0.,            0., 0., 0., 0., 1. / sqrt(2.)]],
                                requires_grad=False)

        expected_inv = torch.tensor([[sqrt(2.), 0., 0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0., 1., 0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0., 0., 1., 0.,       0., 0., 0., 0.,       0.],
                                     [      0., 0., 0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0., 0., 0., 0., sqrt(2.), 0., 0., 0.,       0.],
                                     [      0., 0., 0., 0.,       0., 1., 0., 0.,       0.],
                                     [      0., 0., 0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0., 0., 0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0., 0., 0., 0.,       0., 0., 0., 0., sqrt(2.)]], requires_grad=False)

        search = NSxSearch('cpu', depth=1, n_ancilla_modes=1, n_ancilla_photons=0, max_success_measurements=1)
        actual, actual_inv = search._GeneticAlgorithm__build_normalization_matrix(
            search._GeneticAlgorithm__build_permutation_matrix())

        self.assertEqual(expected.shape, actual.shape)
        self.assertSequenceEqual(expected.numpy().tolist(), actual.to_dense().numpy().tolist())

        self.assertEqual(expected_inv.shape, actual_inv.shape)
        self.assertSequenceEqual(expected_inv.numpy().tolist(), actual_inv.to_dense().numpy().tolist())

    def test3(self):
        expected = torch.tensor([[1. / sqrt(6.),            0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0., 1. / sqrt(2.), 0.,            0., 0., 0., 0.,            0.],
                                 [           0.,            0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0.,            0., 0., 1. / sqrt(2.), 0., 0., 0.,            0.],
                                 [           0.,            0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0.,            0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0.,            0., 0.,            0., 0., 0., 0.,            0.],
                                 [           0.,            0., 0.,            0., 0., 0., 0., 1. / sqrt(6.)]],
                                requires_grad=False)

        expected_inv = torch.tensor([[sqrt(6.),       0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0., sqrt(2.), 0.,       0., 0., 0., 0.,       0.],
                                     [      0.,       0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0.,       0., 0., sqrt(2.), 0., 0., 0.,       0.],
                                     [      0.,       0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0.,       0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0.,       0., 0.,       0., 0., 0., 0.,       0.],
                                     [      0.,       0., 0.,       0., 0., 0., 0., sqrt(6.)]], requires_grad=False)

        search = NSxSearch('cpu', depth=1, n_ancilla_modes=0, n_ancilla_photons=1, max_success_measurements=1)
        actual, actual_inv = search._GeneticAlgorithm__build_normalization_matrix(
            search._GeneticAlgorithm__build_permutation_matrix())

        self.assertEqual(expected.shape, actual.shape)
        self.assertSequenceEqual(expected.numpy().tolist(), actual.to_dense().numpy().tolist())

        self.assertEqual(expected_inv.shape, actual_inv.shape)
        self.assertSequenceEqual(expected_inv.numpy().tolist(), actual_inv.to_dense().numpy().tolist())


if __name__ == '__main__':
    unittest.main()
