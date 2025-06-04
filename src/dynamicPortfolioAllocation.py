
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d 

def pi_star_single_asset(mu: float, var: float, r_t: float, N: int = 1000000) -> float:
    """
    Compute the optimal portfolio weight pi_t* when investing in a single risky asset.

    Parameters
    ----------
    mu : float
        Expected return of the risky asset.
    var : float
        Variance of the risky asset's return.
    r_t : float
        Risk-free rate at time t.
    N : int
        Number of Monte Carlo samples.

    Returns
    -------
    float
        Optimal portfolio weight pi_t* in the risky asset.
    """
    sigma = np.sqrt(var)
    rets = np.random.normal(mu, sigma, size=N)
    delta = rets - r_t

    def neg_expected_log(pi: float) -> float:
        if not 0 < pi < 1:
            return np.inf
        arg = 1 + r_t + pi * delta
        if np.any(arg <= 0):
            return np.inf
        return -np.mean(np.log(arg))

    result = minimize_scalar(neg_expected_log, bounds=(1e-8, 1 - 1e-8), method='bounded')
    if not result.success:
        raise RuntimeError("Scalar optimization failed.")
    
    return result.x, result.fun

def value_func_exact(v0, t, T, mean_ret, var, r_f, N=int(1e7)):
    _, g = pi_star_single_asset(mean_ret, var, r_f, N=N)
    return np.log(v0) + (T-1)*g


def pi_star(mu: np.ndarray, cov: np.ndarray, r_t: float, N: int = 10000) -> np.ndarray:
    """
    Compute the optimal portfolio weights pi_t* using convex optimization.

    Parameters
    ----------
    mu : np.ndarray
        Expected return vector of shape (K,).
    cov : np.ndarray
        Covariance matrix of shape (K, K).
    r_t : float
        Risk-free rate at time t.
    N : int
        Number of Monte Carlo samples to draw from the return distribution.

    Returns
    -------
    np.ndarray
        Optimal weight vector pi_t of shape (K + 1,).
    """
    K = mu.shape[0]
    rets = np.random.multivariate_normal(mu, cov, size=N)  
    rets = np.hstack([np.full((rets.shape[0], 1), r_t), rets])

    pi = cp.Variable(K+1)
    log_terms = cp.log(1 + rets @ pi)
    objective = cp.Maximize(cp.sum(log_terms) / N)
    constraints = [pi >= 0, cp.sum(pi) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS) 

    if pi.value is None:
        raise RuntimeError("Optimization failed.")

    return pi.value, np.mean(log_terms.value)  

def crra_utility(v, gamma):
    """
    Compute the CRRA utility function.
    Parameters
    ----------
    v : np.ndarray
        Wealth levels.
    gamma : float
        Risk aversion coefficient.
    Returns
    -------
    np.ndarray
        Utility values.
    """ 
    if gamma == 1:
        return np.log(v)
    else:
        return v**(1 - gamma) / (1 - gamma)

def simulate_returns(mu, sigma, N):
    """
    Simulate returns from a normal distribution.
    Parameters
    ----------
    mu : float
        Mean return of the risky asset.
    sigma : float
        Standard deviation of the risky asset.
    N : int
        Number of samples to draw.
    Returns
    -------
    np.ndarray
        Simulated returns.
    """ 

    return np.random.normal(loc=mu, scale=sigma, size=N)

def generate_wealth_grids(T, v_size, v_min, v_max, r_f, mu, sigma):
    """
    Compute wealth grids for each time step.

    Parameters
    ----------
    T : int
        Number of time steps.
    v_size : int
        Number of wealth levels in the grid.
    r_f : float
        Risk-free rate.
    mu : float  
        Mean return of the risky asset.
    sigma : float
        Standard deviation of the risky asset.
    v_min : minimum initial wealth.
    v_max : maximum initial wealth.

    Returns
    -------
    ndarray
        Wealth grids for each time step.
    """
    v_grids = []
    amplify_factor = 1 + r_f + mu + 3*sigma
    shrink_factor = 1 + r_f + mu - 3*sigma
    for _ in range(T + 1):
        v_grid_t = np.linspace(v_min, v_max, v_size)
        v_grids.append(v_grid_t)
        v_max *= amplify_factor 
        v_min = max(0.0, v_min*shrink_factor) 
    return v_grids

def compute_cost_to_go(v_grid, V_approx, pi_grid, r_f, mu, sigma, N, v_min, v_max):
    """
    Compute V_t(v) for all v in v_grid using Monte Carlo simulation and interpolation.

    Parameters
    ----------
    v_min : float
        Minimum wealth.
    v_max : float
        Maximum wealth.
    v_grid : np.ndarray
        Wealth grid of shape (v_size,).
    V_approx : callable
        Interpolating function for the value function.
    pi_grid : np.ndarray
        Portfolio weights grid of shape (pi_size,).
    r_f : float
        Risk-free rate.
    mu : float
        Expected return of the risky asset.
    sigma : float
        Standard deviation of the risky asset.
    N : int
        Number of Monte Carlo samples.
    Returns
    -------
    np.ndarray
        Value function V_t(v) for all v in v_grid and the best portfolio weight pi_t*.
    """ 
    # Initialize the value function matrix
    EV_matrix = np.zeros((len(pi_grid), len(v_grid)))    
    
    # Simulate returns
    R_samples = simulate_returns(mu, sigma, N)

    for j, pi in enumerate(pi_grid):
        v_prime = np.outer(v_grid, 1 + pi * R_samples + (1 - pi) * r_f)
        v_prime = np.clip(v_prime, v_min, v_max)
        EV_matrix[j, :] = np.mean(V_approx(v_prime), axis=1)

    best_val = np.max(EV_matrix, axis=0)
    best_pi_indices = np.argmax(EV_matrix, axis=0)
    best_pi = pi_grid[best_pi_indices]

    return best_val, best_pi

def pi_star_single_asset_adp(T=10,  v_min=1e-2, v_max=5.0, v_size=100, pi_grid=None, N=1000, r_f=0.01, mu=0.05, sigma=0.2, gamma=1.0):                                     
    """
    Compute the optimal portfolio weight pi_t* when investing in a single risky asset.

    Parameters
    ----------
    T : float
        Time horizon.
    v_min : float
        Minimum wealth. 
    v_max : float
        Maximum wealth.
    v_size : int
        Number of wealth grid points.
    pi_grid : array_like, optional
        Control grid for portfolio weights. If None, a default grid is created.
    N : int
        Number of Monte Carlo samples.
    r_f : float
        Risk-free rate.
    mu : float
        Expected return of the risky asset.
    sigma : float
        Volatility of the risky asset.
    gamma : float
        Risk aversion coefficient.

    Returns
    -------
    ndarray
        Policy function for the optimal portfolio weight pi_t*.
    ndarray
        Value function V_0(v) for all v in the wealth grid at time t=0.
    ndarray
        Wealth grid for time t=0.
    """

    if pi_grid is None:
        pi_grid = np.linspace(0.0, 1.0, 21)

    v_grids = generate_wealth_grids(T, v_size, v_min, v_max, r_f, mu, sigma)
    V_approx = lambda v: crra_utility(v, gamma)
    policy = np.zeros((T, v_size))

    for t in reversed(range(T)):
        v_grid = v_grids[t]
        v_max_next = v_grids[t + 1][-1]
        best_val, best_pi = compute_cost_to_go(v_grid, V_approx, pi_grid, r_f, mu, sigma, N, v_min, v_max_next)
        V = best_val.copy()
        policy[t, :] = best_pi
        V_approx = interp1d(v_grid, V, kind='cubic', fill_value="extrapolate")
    return policy, V_approx, v_grids[0]

