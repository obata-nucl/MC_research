import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list[int], output_dim: int):
        super(NN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        eps = F.softplus(out[:, 0])
        kappa = -F.softplus(out[:, 1])
        chi_n = -F.softplus(out[:, 2])
        return torch.stack([eps, kappa, chi_n], dim=1)



def IBM2_PES(params, n_pi, n_nu, beta_f):
    beta_b = beta_f*5.0

    beta2 = beta_b**2
    A = 1/(1 + beta2)
    SQRT_2_OVER_7 = math.sqrt(2.0/7.0)

    term_1 = params[:, 0]*(n_pi + n_nu)*beta2*A
    term_2 = n_pi*n_nu*params[:, 1]*beta2*A*A*(
        4 - 2*SQRT_2_OVER_7*(params[:, 2] - 0.5)*beta_b - params[:, 2]*beta2/7
    )

    return term_1 + term_2