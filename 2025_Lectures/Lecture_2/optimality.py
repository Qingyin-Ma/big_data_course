import numpy as np
from numba import njit, float64, prange
from numba.experimental import jitclass
from interpolation import interp
from scipy.special import roots_hermite
import matplotlib.pyplot as plt

# Gauss-Hermite quadrature
gh_points, gh_weights = roots_hermite(n=7)  # Gauss-Hermite sample points/weights
gh_points = np.sqrt(2) * gh_points          # normalized Gauss-Hermite points
gh_weights = gh_weights / np.sqrt(np.pi)    # normalized Gauss-Hermite weights

ifp_data = [
    ('γ', float64),           # coefficient of relative risk aversion
    ('β', float64),           # discount factor
    ('P', float64[:,:]),      # transition probability matrix for Z_t
    ('z_vals', float64[:]),   # state values for Z_t
    ('a_r', float64),         # scale parameter for R_t
    ('b_r', float64),         # additive parameter for R_t
    ('a_y', float64),         # scale parameter for Y_t
    ('b_y', float64),         # additive parameter for Y_t
    ('s_grid', float64[:]),   # grid points for saving
    ('gh_pts', float64[:]),   # normalized Gauss-Hermite sample points
    ('gh_wgts', float64[:])   # normalized Gauss-Hermite weights
]

@jitclass(ifp_data)
class IFP:
    """
    A class that stores primitives for the income fluctuation problem.
    """
    def __init__(self,
                 γ=1.5,
                 β=0.96,
                 P=np.array([[0.9,0.1],
                             [0.1,0.9]]),
                 z_vals=np.array([0,1.]),
                 a_r=0.1,
                 b_r=0.0,
                 a_y=0.2,
                 b_y=0.5,
                 gh_pts=gh_points,
                 gh_wgts=gh_weights,
                 grid_min=0.,
                 grid_med=20,
                 grid_max=1000,
                 grid_size=10000):
        self.P, self.z_vals, self.γ, self.β = P, z_vals, γ, β
        self.a_r, self.b_r, self.a_y, self.b_y = a_r, b_r, a_y, b_y
        self.gh_pts, self.gh_wgts = gh_pts, gh_wgts
        
        ## Create evenly-spaced grid for saving
        #self.s_grid = np.linspace(grid_min, grid_max, grid_size)
        
        # Create exponential grid for saving
        sp = (grid_med-grid_min)**2 / (grid_min+grid_max-2*grid_med) - grid_min
        grid = np.linspace(np.log(grid_min+sp), np.log(grid_max+sp), grid_size)
        grid = np.exp(grid) - sp
        grid[0], grid[-1] = grid_min, grid_max
        self.s_grid = grid
        
        # Test stability assuming {R_t} is IID and adopts the lognormal
        # specification given below. The test is then β E R_t < 1.
        ER = np.exp(b_r + a_r**2 / 2)
        assert β * ER < 1, "Stability condition failed."
        
    # Marginal utility function
    def u_prime(self, c):
        return c**(-self.γ)
    
    # Inverse marginal utility function
    def u_prime_inv(self, c):
        return c**(-1/self.γ)
    
    def R(self, z, ζ):
        return np.exp(self.a_r*ζ + self.b_r)
    
    def Y(self, z, η):
        return np.exp(self.a_y*η + z*self.b_y)

    
@njit(parallel=True)
def T(a_in, c_in, ifp):
    """
    The Coleman operator that updates the candidate consumption function 
    and the asset grid points via the endogenous grid method of Carroll (2006). 
    """
    u_prime, u_prime_inv = ifp.u_prime, ifp.u_prime_inv
    R, Y, P, z_vals, β = ifp.R, ifp.Y, ifp.P, ifp.z_vals, ifp.β
    s_grid, gh_pts, gh_wgts = ifp.s_grid, ifp.gh_pts, ifp.gh_wgts
    M, N, K = len(P), len(s_grid), len(gh_pts)
    
    # Create candidate consumption function 
    def c_func(a, n): # linear interpolation
        if a <= a_in[-1,n]:
            res = interp(a_in[:,n], c_in[:,n], a)
        else: # linear extrapolation
            num = c_in[-1,n] - c_in[-2,n]  
            den = a_in[-1,n] - a_in[-2,n]
            slope = num / den
            res = c_in[-1,n] + slope * (a - a_in[-1,n])
        return res
    
    # Create empty spaces 
    c_out = np.empty_like(c_in)  # to store updated consumption
    a_out = np.empty_like(a_in)  # to store endogenous asset grid
    
    # Compute updated consumption
    for n in prange(N):      # enumerate current saving
        s = s_grid[n]        # current saving
        for m in prange(M):  # enumerate current Z
            z = z_vals[m]    # current Z
            Ez = 0           # start computing expectation
            for m_next in prange(M):      # integration w.r.t. next-period Z
                z_hat = z_vals[m_next]    # next-period Z
                for kR in prange(K):      # integration w.r.t. next-period ζ
                    R_hat = R(z_hat, gh_pts[kR])  # next-period R
                    for kY in prange(K):  # integration w.r.t. next-period η
                        Y_hat = Y(z_hat, gh_pts[kY])  # next-period Y
                        a_hat = R_hat * s + Y_hat     # next-period a
                        integ = R_hat*u_prime(c_func(a_hat, m_next))  # integrand
                        Ez += integ * P[m,m_next] * gh_wgts[kR] * gh_wgts[kY]
            c_out[n,m] = u_prime_inv(β * Ez)  # updated consumption at current state
            a_out[n,m] = s + c_out[n,m]       # endogenous asset grid point 
    
    # Fix consumption-asset pair at (0,0) to improve interpolation
    c_out[0,:], a_out[0,:] = 0, 0
    
    return a_out, c_out


def initialize(ifp):
    "Initial guess of the optimal policy."
    a_init = np.zeros((len(ifp.s_grid), len(ifp.P)))
    a_init += ifp.s_grid.reshape(-1,1)
    c_init = a_init.copy()
    return a_init, c_init


@njit
def solve_model_time_iter(model, oper, a_init, c_init, tol=1e-4, 
                          max_iter=10000, verbose=True, print_skip=50):
    """
    Time iteration using the endogenous grid method of Carroll (2006).
    ---------
    Returns :
    ---------
    a_new : the endogenous asset grid points
    c_new : the optimal consumption level
    k     : steps for the time iteration to terminite
    """
    k, err = 0, tol + 1
    
    while err>tol and k<max_iter:
        a_new, c_new = oper(a_init, c_init, model)
        err = np.max(np.abs(c_new - c_init))  # using absolute distance
        #err = np.max(np.abs(c_new[1:,:,:]/c_init[1:,:,:] - 1))  # using relative distance
        a_init, c_init = np.copy(a_new), np.copy(c_new)
        k += 1
        
        if verbose and k%print_skip==0:
            print("Error at iteration", k, "is", err)
    
    if k == max_iter:
        print("Failed to converge!")
    
    if verbose and k < max_iter:
        print("\nConverged in", k, "iterations.")
        
    return a_new, c_new, k