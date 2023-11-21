import numpy as np
from scipy.linalg import toeplitz
class TwoPointBoundaryValueProblem:


    def __init__(self,f):
        # Initiate the 2p BVP by assigning the right hand function in d^2 y/dx^2 = f(x,y)
        self.f = f
        
    def set_boundary_conditions(self,a,b,c,d,type):
        # Set up the boundary conditions for this BVP
        # for Dirichlet conditions use y(x_0) = a and y(x_N+1) = b and c = d = 0 and set type = 'D-D'
        # Dirichlet-Neumann conditions use y(x_0) = a, b= c = 0, dy(x_N)/dx = d and set type = 'D-H'
        # 
        self.a ,self.b,self.c,self.d = a, b, c, d
        self.type = type
    
    def solve_linear_BVP(self,L,N,fvec):

        # here we assume f() is only a function of x and self.fvec is f evaluated at the grid points x = [x1 x2 x3 . . . ]
        if self.type == 'D-D': # Dirichlet boundary conditions
            self.dx = L/(N+1)
            self.x = (self.dx)*(np.linspace(1,N,N))
            T = self.central_difference_approx_matrix()
            BC = np.zeros(len(self.x))*(1/self.dx**2)
            BC[0],BC[-1] = self.a/self.dx**2, self.b/self.dx**2
            self.BC = BC
            self.T = T
            y = np.matmul(np.linalg.inv(self.T),-self.BC + fvec)
            y = np.append(np.array([self.a]),np.append(y,np.array([self.b])))
            x = np.append(np.array([0]),np.append(self.x,np.array([L])))
            return x, y
        if self.type == 'D-N':
            raise NotImplementedError
        if self.type == 'N-D':
            raise NotImplementedError


    def central_difference_approx_matrix(self):
        # this function returns a matrix approximating the second derivative of y using the central difference approximation
        N = len(self.x) 
        r = np.zeros(N)
        r[:2] = [-2,1]
        c = np.zeros(N)
        c[:2] = [-2,1]
        # Create the Toeplitz matrix
        T_dx = (1/self.dx**2)*toeplitz(c=c, r=r)
        return T_dx
    def f_evaluate(self):
        return self.f(self.x)