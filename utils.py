import numpy as np
from numpy.linalg import inv, norm

def relu(x):
    return x * (x > 0)
    
def differential(function, x, d=1e-6):
    y = function(x)
    ret = []
    for k in range(x.shape[0]):
        dk = np.zeros(x.shape)
        dk[k] = d
        x_plus_dk = x + dk
        y_plus_dy = function(x_plus_dk)
        ret.append((y_plus_dy - y) / d)
    return np.array(ret)

def second_differential(function, x, d=1e-6):
    y = function(x)
    ret = []
    for k in range(x.shape[0]):
        dk = np.zeros(x.shape)
        dk[k] = d
        x_plus_dk = x + dk
        x_minus_dk = x - dk
        second_diff = (function(x_plus_dk) - 2*y + function(x_minus_dk)) / (d**2)
        ret.append(second_diff)
    return np.array(ret)

def differential_dxdy(function, x, d=1e-6):
    """
    This function will not be used in solving the n-dim function
    This is only a prototype of partial derivative d2f/dxdy
    """
    assert x.shape[0] == 2, "This function only supports input dimension of 2"
    pp = function(x+np.array([d, d]))
    pm = function(x+np.array([d, -d]))
    mp = function(x+np.array([-d, d]))
    mm = function(x+np.array([-d, -d]))
    return (pp-pm-mp+mm) / (4 * d**2)

def normal_equation(X, y):
    w = inv(X.T.dot(X)).dot(X.T).dot(y)
    return w
    
def least_square(A, b):
    # Calculate x given Ax=b when (A.T @ A) is singular
    return A.T @ inv(A @ A.T) @ b