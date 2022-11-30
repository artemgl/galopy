import torch


class LoNet(torch.nn.Module):
    def __init__(self, n_modes, device='cpu'):
        super().__init__()
        self.n_modes = n_modes
        i = self.n_modes * (self.n_modes - 1) // 2
        self.alphas = torch.nn.Linear(i, 1, bias=False, device=device)
        self.betas = torch.nn.Linear(i, 1, bias=False, device=device)
        self.gammas = torch.nn.Linear(n_modes, 1, bias=False, device=device)
        self.device = device
        # self._alphas_mask = torch.tensor([[True] * i + [False] * (n_modes - 1 - i) for i in range(n_modes)],
        #                                  dtype=torch.bool)

        # self.angles = torch.rand(n_modes, n_modes, device=device) * 2. * pi'

    def forward(self):
        sin_s = torch.sin(self.alphas.weight).reshape(-1)
        cos_s = torch.cos(self.alphas.weight).reshape(-1)
        exp_beta_s = torch.exp(1.j * self.betas.weight).reshape(-1)
        exp_gamma_s = torch.exp(1.j * self.gammas.weight).reshape(-1)

        transform = torch.eye(self.n_modes, dtype=torch.complex64, device=self.device)

        for i in range(self.n_modes - 1):
            for j in range(i + 1):
                local_transform = torch.eye(self.n_modes, dtype=torch.complex64, device=self.device)
                local_transform[i - j, i - j] = cos_s[i * (i + 1) // 2 + j]
                local_transform[i - j + 1, i - j] =\
                    -exp_beta_s[i * (i + 1) // 2 + j] * sin_s[i * (i + 1) // 2 + j]
                local_transform[i - j, i - j + 1] =\
                    exp_beta_s[i * (i + 1) // 2 + j].conj() * sin_s[i * (i + 1) // 2 + j]
                local_transform[i - j + 1, i - j + 1] = cos_s[i * (i + 1) // 2 + j]

                transform = local_transform.matmul(transform)

        transform = torch.diag(exp_gamma_s).matmul(transform)

        return transform
