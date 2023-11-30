import numpy as np

def sin_trapezoid():
    """
    Generates a 100x1 toy image (signal) of a sinusoid followed by a blocky trapezoid.
    """

    M = 100
    h = 1
    j = np.linspace(0,M,M+1)
    x = h*j

    toy_sig = (np.sin(x/np.pi) + np.cos(x/np.pi))*(x <= 37)
    toy_sig += ((2*x-1)/100)*((x > 60) & (x < 80)) 
    toy_sig += 0*(x > 80)
    toy_sig += 3

    return toy_sig