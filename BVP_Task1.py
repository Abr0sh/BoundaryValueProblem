#%%
import numpy as np
from scipy.linalg import toeplitz
from boundaryValueProblem import TwoPointBoundaryValueProblem as bvp
import matplotlib.pyplot as plt

#%%  Set up the BVP
N = 10
L = 10
dx = L/(N+1)
x =(dx)*(np.linspace(1,N,N))
# Defining a test problem
def f(x):
    return -(np.pi/L)**2*np.sin(np.pi*x/L)
def exact_y(x):
    return np.sin(np.pi*x/L)

#%% Solving and plotting 
# ddy/dx^2 = -(np.pi/L)**2*sin(pi * x/L) has solution sin(pi x/L)
BVP = bvp(f)
BVP.set_boundary_conditions(0,0,0,0,'D-D')
feval = f(x)
x, y= BVP.solve_linear_BVP(L,N,feval)
plt.plot(x,y,'o',label = 'Numerical solution')
plt.plot(x,exact_y(x),label = 'Exact solution')
plt.legend()

# %% Calculating the global error
def norm_rms(vec,dx):
    return np.linalg.norm(vec) * np.sqrt(dx)
def step_grid(L,exp_i,exp_f,):
    K = np.linspace(exp_i,exp_f,exp_f-exp_i+1)
    N_grid = [2**k for k in K]
    return np.array([L/(n+1) for n in N_grid]), N_grid
dx_grid,N_grid = step_grid(L,1,12)
error = []
for n in N_grid:
    dx = L/(n+1)
    n = int(n)
    x =(dx)*(np.linspace(1,int(n),int(n)))
    feval = f(x)
    x,y= BVP.solve_linear_BVP(L,n,feval)
    exact = exact_y(x)
    error.append(norm_rms(exact-y,dx))
plt.loglog(dx_grid,error,'x',label= 'gloabal error in dx')
plt.loglog(dx_grid,0.03*dx_grid**2,label='O(dx^2)')
plt.legend()
# %%

