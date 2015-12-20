# -*- coding: utf-8 -*-
__author__ = 'Jiapeng Hong'

import numpy as np

def simple_quadratic_function(x):
    '''
    compute value of f(x1,x2) = x1^2 + 3 * x1 * x2, and numeric gradient of x1,x2
    :param x: a point with dim=2
    :return: (value, grad) tuple
    '''
    value   =  x[0] ** 2 + 3 * x[0] * x[1]
    grad    = []
    grad.append(2 * x[0] + 3 * x[1]) # gradient of x[0]
    grad.append(3 * x[0])
    return (value,grad)

def check_gradient():
    x           = [4,10]
    value, grad = simple_quadratic_function(x)
    num_grad    = compute_gradient(simple_quadratic_function, x)
    print(num_grad, grad)
    print("The above two columns you get should be very similar.\n" \
          "(Left-Your Numerical Gradient, Right-Analytical Gradient)\n")

    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print(diff)
    print("Norm of the difference between numerical and analytical num_grad "
          "(should be < 1e-9)\n")

def compute_gradient(J,theta):
    eps         = 1e-5
    cost_p      = np.zeros([theta.size])
    cost_m      = np.zeros([theta.size])
    for i in range(len(cost_p)):
        base            = np.zeros([theta.size])
        base[i]         = eps
        cost_p[i],_     = J(theta + base)
        cost_m[i],_     = J(theta - base)
    grad                = (cost_p - cost_m) / (2 * eps)
    return grad
