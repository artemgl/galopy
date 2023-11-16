import numpy as np
import torch
# from galopy.gd import CircuitSearch
# import galopy.gd.topology as tl

if __name__ == '__main__':
    n_meas = 3
    n_ibs = 4
    n_sm = 2
    n_sp = 3
    v = torch.randint(low=0, high=10, size=(n_meas, n_sm ** n_sp, n_ibs))
    size = [n_meas] + [n_sm] * n_sp + [n_ibs]
    v = v.reshape(*size)
    print(v)
    u = torch.randint(low=0, high=10, size=(n_meas, n_sm, n_sm))
    print(u)
    size = [n_meas] + [1] * (n_sp - 1) + [n_sm] * 2
    u = u.reshape(*size)

    # v.transpose_(0, 1)
    v = u.matmul(v)
    # v.transpose_(0, 1)
    print(v)

    #
    # a = torch.tensor([[1, 2, 3],
    #                   [4, 5, 6]])
    # b = torch.tensor([5, 6, 7], device='cuda')
    #
    # # print(a.max().data.cpu().numpy())
    # print(a[0, 1])
