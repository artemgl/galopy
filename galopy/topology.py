import torch
import numpy as np
from math import ceil, log


class Topology:
    def __init__(self, n_modes, device='cuda'):
        self.n_modes = n_modes
        self.device = device
        self.modes = self._gen()

    def _gen(self):
        return torch.tensor([[]], device=self.device)

    # TODO: optimize
    def gen_unitary(self, bs_angles, ps_angles):
        sin_s = torch.sin(bs_angles[..., 0]).reshape(-1)
        cos_s = torch.cos(bs_angles[..., 0]).reshape(-1)
        exp_beta_s = torch.exp(1.j * bs_angles[..., 1]).reshape(-1)
        exp_gamma_s = torch.exp(1.j * ps_angles).reshape(-1)

        transform = torch.eye(self.n_modes, dtype=torch.complex64, device=self.device)
        counter = 0
        for x, y in self.modes:
            local_transform = torch.eye(self.n_modes, dtype=torch.complex64, device=self.device)
            local_transform[x, x] = cos_s[counter]
            local_transform[y, x] = -exp_beta_s[counter] * sin_s[counter]
            local_transform[x, y] = exp_beta_s[counter].conj() * sin_s[counter]
            local_transform[y, y] = cos_s[counter]
            counter += 1
            transform = local_transform.matmul(transform)
        return torch.diag(exp_gamma_s).matmul(transform)


# x x   x x     x     x     x
# x |x x| |x    |x    |x x  |
# x x| x| ||x   ||x x || |x |
# x  x  x |||x x||| |x|| ||x|
# x x   x x||| x||| x||| x|||
# x |x x|  x||  x||  x||  x||
# x x| x|   x|   x|   x|   x|
# x  x  x    x    x    x    x
class Parallel(Topology):
    def __init__(self, n_modes):
        super().__init__(n_modes)

    def _gen(self):
        res = []
        blocks = [self.n_modes]
        for i in range(ceil(log(self.n_modes, 2))):
            # Слой
            blocks = [x for sublist in [[b // 2, b - (b // 2)] for b in blocks] for x in sublist]
            block = []

            # Индекс, с которого начинаются преобразования в общей матрице
            start = 0
            for j in range(len(blocks) // 2):
                # Параллельный блок в слое

                # Количество мод в левой половине
                left = blocks[2 * j]
                # Количество мод в правой половине
                right = blocks[2 * j + 1]
                for k in range(right):
                    # Параллельный шаг в блоке
                    for m in range(left):
                        # Конкретная мода в левой половине
                        x = start + m
                        y = start + left + (m + k) % right
                        block.append([x, y])
                start += left + right
            res = block + res
        return np.array(res)


#       x      x     x    x   x  x x
#      x|     x|    x|   x|  x| x| x
#     x||    x||   x||  x|| x|| xx
#    x|||   x|||  x||| x||| xxx
#   x||||  x|||| x|||| xxxx
#  x||||| x||||| xxxxx
# x|||||| xxxxxx
# xxxxxxx
class Stable(Topology):
    def __init__(self, n_modes):
        super().__init__(n_modes)

    def _gen(self):
        res = []
        for i in range(self.n_modes - 1):
            for j in range(i + 1, self.n_modes):
                res.append([i, j])
        return np.array(res)


#       x
#      xxx
#     xxxxx
#    xxxxxxx
#   xxxxxxxxx
#  xxxxxxxxxxx
# xxxxxxxxxxxxx
# x x x x x x x
class Pyramidal(Topology):
    def __init__(self, n_modes):
        super().__init__(n_modes)

    def _gen(self):
        res = []
        for i in range(self.n_modes - 1):
            for j in range(i + 1):
                res.append([i - j, i - j + 1])
        return np.array(res)
