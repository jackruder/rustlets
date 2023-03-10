from re import T
import numpy as np
from matplotlib import pyplot as plt

def centraldiff3(f_prev, f_next, x_prev, x_next):
    """
    Approximate the derivative of f(x) at using
    a three point central difference scheme.

    Inputs:
        f_prev: f(x - h)
        f_next: f(x + h)
        x_prev: x-h
        x_next: x+h
    """
    return (f_next - f_prev)/(x_next - x_prev)

def centraldiff5(f_arr, i, h):
    """
    Approximate the derivative of f(x) at t=x[i] using
    a five point central difference scheme. Requires evenly spaced values of x for accuracy

    Inputs:
        f_arr: array of f values of length n
        i:     the index indicating t, it must hold that 1<i<n-2
        h:     the constant difference between any two adjacent x values
    """
    if not (i > 2 and i < len(f_arr)-2):
        return 0 # TODO better error handling

    return ((f_arr[i-2] - 8 * f_arr[i-1] + 8 * f_arr[i+1] - f_arr[i+2])/(12 * h))

def forwarddiff(f, f_next, x, x_next):
    """
    Approximate the derivative of f(x) using a forward difference scheme.

    Inputs:
    """
    return (f_next - f) / (x_next - x)

def backwarddiff(f_prev, f, x_prev, x):
    """
    Approximate the derivative of f(x) at t=x[i] using a backward difference scheme.    Inputs:
    """
    return (f - f_prev)/(x - x_prev)

def approximate_diff(f_arr,x_arr):
    if len(f_arr) != len(x_arr):
        return [] # TODO handle better

    diffs = np.empty(len(x_arr), dtype=np.complex64)
    for i,x in enumerate(x_arr):
        if i == 0:
            diffs[i] = forwarddiff(f_arr[0],f_arr[1],x_arr[0],x_arr[1])
        elif i == len(x_arr)-1:
            diffs[i] = backwarddiff(f_arr[i-1],f_arr[i],x_arr[i-1],x_arr[i])
        else:
            diffs[i] = forwarddiff(f_arr[i-1],f_arr[i+1],x_arr[i-1],x_arr[i+1])
    return diffs

def trapezoid_int(f_arr, x_arr):
    """
    integrate f from x[0] to x[N], assuming that x[n] - x[n-1] = c
    """

    s = 0 + 0j
    for i in range(len(f_arr)-1):
        h = x_arr[i+1] - x_arr[i]
        s += h * (f_arr[i] + f_arr[i+1])/2

    return s


if __name__ == '__main__':
    x = np.arange(-3, 3, 0.01)
    y = x**3

    dy = approximate_diff(y,x)

    print(trapezoid_int(y,x)) # expected 41.667 

    plt.plot(x,y,label='f')
    plt.plot(x,dy,label='df/dx')
    plt.legend()
    plt.show()




