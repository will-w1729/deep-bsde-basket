# Credit: adapted from https://github.com/YifanJiang233/Deep_BSDE_solver/tree/master

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

dtype = torch.float64 # modify here for different type e.g. torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_gbm_basket_equation(x_0, r, q, vol, corr, K, weights, T, d, batch_size):
    """
    Return an fbsde object for use with multi-asset correlated GBM, european basket call option and 
    exact step calculation in forwards SDE for higher accuracy option price
    """

    def to_tensor(a):
        return torch.as_tensor(a, device=device, dtype=dtype)

    x_0 = to_tensor(x_0)
    q = to_tensor(q)
    vol = to_tensor(vol)
    weights = to_tensor(weights)
    K = to_tensor(K)
    r = to_tensor(r)

    # We store the cholesky decomposition of the correlation matrix between the stocks
    L_np = np.linalg.cholesky(corr)
    L = torch.as_tensor(L_np, device=device, dtype=dtype)

    def x_forward_step(x, delta_t, dW):
        """Computes exact step of GBM stock price over time delta_t"""
        dW = dW.squeeze(-1)
        drift = (r - q - 0.5 * vol**2) * delta_t
        diffusion = vol * dW
        x = x * torch.exp(drift + diffusion)
        return x

    # forming functions of the FBSDE systems from stock parameters
    q_expanded = q.unsqueeze(0).expand(batch_size, d)
    def b(t, x, y):
        return (r - q_expanded) * x.reshape(batch_size, d)

    vol_expanded = vol.unsqueeze(0).expand(batch_size, d)
    def sigma(t, x):
        return torch.diag_embed(vol_expanded * x).reshape(batch_size, d, d)

    def f(t, x, y, z):
        return r * y.reshape(batch_size, 1)

    def g(x):
        basket = x @ weights
        payoff = (basket - K).clamp_min(0)
        return payoff.reshape(batch_size, 1)

    equation = fbsde(x_0, b, sigma, f, g, T, x_0.shape[0], 1, x_0.shape[0], L, x_forward_step)
    return equation


class fbsde():
    """
    Object storing functions and parameters of the FBSDE system
    Additionally stores:
    cholesky decomposition of the correlation matrix between stocks
    option to simulate GBM forwards SDE with exact step or Euler-Maruyama
    """
    def __init__(self, x_0, b, sigma, f, g, T, dim_x, dim_y, dim_d, cholesky, x_forward_step=None):
        self.x_0 = x_0.to(device=device, dtype=dtype)
        self.b = b
        self.sigma = sigma
        self.f = f
        self.g = g
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d 
        self.cholesky = cholesky.to(device=device, dtype=dtype)
        self.x_forward_step = x_forward_step


class Model_GBM(nn.Module):
    """
    Contains multilayer feed forward neural network with three layers approximating Z_t.
    Generates correlated Brownian paths to simulate forwards and backwards SDE over discrete time steps to terminal time.
    A GBM forwards SDE (modelling stock prices) may be simulated exactly instead of using Euler-Maruyama.
    """
    def __init__(self, equation, dim_h):
        super().__init__()
        # Initialising NN layers and estimated option price parameter (y_0)
        self.linear1 = nn.Linear(equation.dim_x + 1, dim_h, dtype=dtype)
        self.linear2 = nn.Linear(dim_h, dim_h, dtype=dtype)
        self.linear3 = nn.Linear(dim_h, equation.dim_y * equation.dim_d, dtype=dtype)
        self.y_0 = nn.Parameter(torch.rand(equation.dim_y, device=device, dtype=dtype))
        self.equation = equation

    def forward(self, batch_size, N):
        """
        batch_size: Number of SDE simulations computed for each iteration
        N: Number of steps in discretisation of our SDEs
        """
        def phi(x):
            """
            Neural Network approximating Z_t
            """
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            return self.linear3(x).reshape(-1, self.equation.dim_y, self.equation.dim_d)

        delta_t = self.equation.T / N # time step

        # Generating correlated random Brownian paths from independent Brownian paths
        W_indep = torch.randn(batch_size, self.equation.dim_d, N, device=device, dtype=dtype) * np.sqrt(delta_t)
        W = torch.einsum('bij,ik->bkj', W_indep, self.equation.cholesky.T) # correlated Brownian paths

        x = self.equation.x_0 + torch.zeros(W.size(0), self.equation.dim_x, device=device, dtype=dtype)
        y = self.y_0 + torch.zeros(W.size(0), self.equation.dim_y, device=device, dtype=dtype)

        for i in range(N):
            t_val = torch.as_tensor(delta_t * i, device=device, dtype=dtype)
            u = torch.cat((x, torch.ones(x.size(0), 1, device=device, dtype=dtype) * t_val), 1)
            z = phi(u)
            w = W[:, :, i].unsqueeze(-1)
            if self.equation.x_forward_step:
                # Exact GBM forward step for x
                x = self.equation.x_forward_step(x, delta_t, w)
            else:
                # E-M forward steps for x and y
                x = x + self.equation.b(t_val, x, y) * delta_t + torch.matmul(self.equation.sigma(t_val, x), w).reshape(-1, self.equation.dim_x)
            y = y - self.equation.f(t_val, x, y, z) * delta_t + torch.matmul(z, w).reshape(-1, self.equation.dim_y)

        return x, y


class BSDEsolver():
    """
    Trains a Model_GBM model using Adam Optimiser.
    Pointwise loss is the MSE between Actual payoff and estimated payoff from the BSDE for a given stock path 
    """
    def __init__(self, equation, dim_h):
        self.model = Model_GBM(equation, dim_h).to(device=device, dtype=dtype)
        self.equation = equation

    def train(self, batch_size, N, itr):
        criterion = torch.nn.MSELoss().to(device=device, dtype=dtype)
        optimizer = torch.optim.Adam(self.model.parameters())

        loss_data, y0_data = [], []

        for _ in range(itr):
            x, y = self.model(batch_size, N)
            loss = criterion(self.equation.g(x), y)
            loss_data.append(float(loss))
            y0_data.append(float(self.model.y_0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss_data, y0_data
