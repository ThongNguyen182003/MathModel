import numpy as np
from typing import Callable

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 12)

Vector = np.array
Matrix = np.array
Function = Callable[[float, Vector], Vector]

def jacobian_x(f: Function, t: float, x: Vector, eps=1E-5) -> Matrix:
    """
    Return the numerical approximate of the Jacobian matrix of the
    2-dimensional real-vector-valued function f(t, x) with respect to x

    t: real number
    x: 2-dimensional real vector
    eps: infinitesimal constant of the approximation
    """
    n = len(x)
    J = np.zeros((n,n))
    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        J[:, i] = (f(t, x_eps) - f(t, x)) / eps
    return J

def solve_system(A: Matrix, b: Vector) -> Vector:
    """
    Solution x to the system of linear equations Ax = b

    x: 2-dimensional real vector
    A: 2x2 matrix with real entries
    b: 2-dimensional real vector

    If the system does not have a solution, x will be the vector 0.
    """
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.zeros(len(b))
    return x

def implicit_euler(f: Function, t: float, x: Vector, h: float) -> Vector:
    """
    The approximate of the exact solution to the ODE dx/dt = f(t, x) at
    time t + h, x is the value of the solution at time t, h is the time step

    t: real number
    h: real number
    x: 2-dimensional real vector
    f: 2-dimensional real-vector-valued function
    """

    # Compute the Jacobian matrix of f with respect to x
    jac = jacobian_x(f, t, x)

    # Compute the matrix A and vector b for solving the linear system Ax = b
    A = np.eye(2) - h * jac
    b = x + h * f(t, x)

    # Solve the linear system Ax = b using the function solve_system
    y = solve_system(A, b)

    return y

# example
def example1(t: float, x: Vector) -> Vector:
    return np.array([2*x[0]+4*x[1], -2*x[0]-2*x[1]])

# Define the Lotka-Volterra system of equations
def lotka_volterra(t: float, x: Vector) -> Vector:
    a, b, c, d = 2, 1, 1.5, 1
    dx = a*x[0] - b*x[0]*x[1]
    dy = c*x[0]*x[1] - d*x[1]
    return np.array([dx, dy])

# Set the initial conditions
# x0 = np.array([1, 1]) # lotka_volterra initial
x0 = np.array([-4, 2]) # example1 initial

# set time step
t0, tn = 0, 10
h = 0.01

# Create arrays to store the solution and time values
ts = np.arange(t0, tn, h)
xs = np.zeros((len(ts), 2))
xs[0] = x0

# Loop over time steps and approximate the solution using implicit Euler
for i in range(1, len(ts)):
    xs[i] = implicit_euler(example1, ts[i-1], xs[i-1], h)

# Plot the solution
plt.plot(ts, xs[:, 0], label='x1')
plt.plot(ts, xs[:, 1], label='x2')
plt.xlabel('Time')
plt.ylabel('x')
plt.legend()
plt.show()
