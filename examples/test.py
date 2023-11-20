import numpy as np
import torch
# from galopy.gd import CircuitSearch
# import galopy.gd.topology as tl

if __name__ == '__main__':
    # n_meas = 3
    # n_ibs = 4
    # n_sm = 2
    # n_sp = 3
    # v = torch.randint(low=0, high=10, size=(n_meas, n_sm ** n_sp, n_ibs))
    # size = [n_meas] + [n_sm] * n_sp + [n_ibs]
    # v = v.reshape(*size)
    # print(v)
    # u = torch.randint(low=0, high=10, size=(n_meas, n_sm, n_sm))
    # print(u)
    # size = [n_meas] + [1] * (n_sp - 1) + [n_sm] * 2
    # u = u.reshape(*size)
    #
    # # v.transpose_(0, 1)
    # v = u.matmul(v)
    # # v.transpose_(0, 1)
    # print(v)

    a = np.array([[0, 0],
                  [0, 1]])
    b = np.array([[1, 1]])
    c = np.array([[1, 2],
                  [0, 2],
                  [2, 2]])
    x = np.concatenate([a, b, c], axis=0)
    print(x)

    #
    # t = torch.tensor([[ 0.0000+0.0000e+00j,  0.0000+0.0000e+00j,  0.0104-1.6476e-03j, 0.0029+8.5463e-04j],
    #      [ 0.0000+0.0000e+00j,  0.0000+0.0000e+00j,  0.0000+0.0000e+00j, 0.0100-4.2406e-04j],
    #      [ 0.0000+0.0000e+00j,  0.0000+0.0000e+00j,  0.0573-2.0081e-02j, 0.0052+2.9798e-04j],
    #      [ 0.0000+0.0000e+00j,  0.0000+0.0000e+00j,  0.0000+0.0000e+00j, -0.0628+1.2380e-02j]])
    # t = t.conj().t().matmul(t).trace()
    # print(abs(t / 4))
    # u = torch.tensor([[1., 0., 0., 0.],
    #                           [0., 1., 0., 0.],
    #                           [0., 0., 1., 0.],
    #                           [0., 0., 0., -1.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.]], dtype=torch.complex64)
    # m = u.t().conj().matmul(t)
    # print(m)


    # size = [2] + [3] * 2 + [4]
    # u = torch.randint(low=0, high=10, size=size)
    #
    # print(u)

    # b = torch.tensor([5, 6, 7], device='cuda')

    # print(a.max().data.cpu().numpy())
    # print(a.shape[1])
