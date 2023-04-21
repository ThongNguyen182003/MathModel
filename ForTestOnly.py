import numpy as np
from typing import Callable

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 12)

Vector = np.array
Matrix = np.array
Function = Callable[[float, Vector], Vector]

def explicit_euler(f: Function, t: float, x: Vector, h: float) -> Vector:
    """
    The approximate of the exact solution to the ODE dx/dt = f(t, x) at
    time t + h, x is the value of the solution at time t, h is the time step

    t: real number
    h: real number
    x: 2-dimensional real vector
    f: 2-dimensional real-vector-valued function 
    """
    return x + h * f(t, x)

#The following functions are used to write the implicit Euler algorithm
def jacobian_x(f: Function, t: float, x: Vector, eps=1E-5) -> Matrix:
  """
  Return the numerical approximate of the Jacobian matrix of the
  2-dimensional real-vector-valued function f(t, x) with respect to x

  t: real number
  x: 2-dimensional real vector
  eps: infinitesimal constant of the approximation
  """
  # begin FTO
  n = len(x)
  J = np.zeros((n,n))
  for i in range(n):
    h = eps*np.abs(x[i])
    if h == 0:
      h = eps
    e_i = np.zeros(n)
    e_i[i] = 1.0
    J[:,i] = (f(t, x + h*e_i) - f(t, x - h*e_i))/(2*h)
  return J
  # end FTO
  pass

def solve_system(A: Matrix, b: Vector) -> Vector:
  """
  Solution x to the system of linear equations Ax = b

  x: 2-dimensional real vector
  A: 2x2 matrix with real entries
  b: 2-dimensional real vector

  If the system does not have a solution, x will be the vector 0.
  """
  # begin FTO
  try:
    x = np.linalg.solve(A, b)
  except np.linalg.LinAlgError:
    x = np.zeros(len(b))
  return x
  # end FTO
  pass

#Implicit Euler Algorithm
def implicit_euler(f: Function, t: float, x: Vector, h: float) -> Vector:
  """
  The approximate of the exact solution to the ODE dx/dt = f(t, x) at
  time t + h, x is the value of the solution at time t, h is the time step

  t: real number
  h: real number
  x: 2-dimensional real vector
  f: 2-dimensional real-vector-valued function 
  """
  # begin FTO
  y = np.random.rand(len(x))
  for _ in range(30):
    A = np.eye(len(x)) - h*jacobian_x(f, t + h, y)
    b = y - h*f(t + h, y) - x
    y += solve_system(A, b)
  return y
  # end FTO
  y = np.random.rand(2)
  for _ in range(30): #You can change this number of Newton steps!
    A = None
    b = None
    y += solve_system(A, b)
  return y

# Define the system of two nonlinear differential equations
def f(t: float, v: Vector) -> Vector:
    x, y = v
    return np.array([-3*x+3*y, -2*x+y])

# Set initial conditions
t0 = 0
x0 = np.array([-4, 2])
y0 = np.array([-4,2])

# Set time step and number of iterations
h = 0.01
num_iterations = 1000

# Initialize arrays to store the results
t = np.zeros(num_iterations)
x = np.zeros((num_iterations, 2))
y = np.zeros((num_iterations, 2))

# Run the explicit Euler method to estimate the solution
for i in range(num_iterations):
    x[i] = x0
    y[i] = y0
    t[i] = t0
    x0 = explicit_euler(f, t0, x0, h)
    y0 = explicit_euler(f, t0, y0, h)
    t0 += h

# Plot the solution
plt.plot(t, x[:, 0], label="x(t)")
plt.plot(t, x[:, 1], label="y(t)")
plt.xlabel("t")
plt.ylabel("x(t), y(t)")
plt.legend()
plt.show()

plt.plot(t, y[:, 0], label="xx(t)")
plt.plot(t, y[:, 1], label="yy(t)")
plt.xlabel("t")
plt.ylabel("x(t), y(t)")
plt.legend()
plt.show()