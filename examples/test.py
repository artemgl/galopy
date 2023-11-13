import numpy as np
import torch
# from galopy.gd import CircuitSearch
# import galopy.gd.topology as tl

if __name__ == '__main__':
    a = torch.tensor([1, 2, 3], device='cuda')
    b = torch.tensor([5, 6, 7], device='cuda')

    print(a.max().data.cpu().numpy())
