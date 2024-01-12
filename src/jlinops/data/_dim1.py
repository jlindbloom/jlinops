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

def piecewise_constant_1d_test_problem():
    """Generates a piecewise constant 1D test vector. From Beck 12.4.3.
    """
    n = 1000
    result = np.zeros(n)
    result[:250] = 1
    result[250:500] = 3
    result[750:] = 2
    return result



def mixed_test_problem():
    """Generates a 1D test vector with mixed behavior.
    """

    n = 1000
    grid = np.linspace(0, 1, n)
    result = np.zeros(n)
    result[:400] = grid[:400] + 20*np.sin((80)*np.pi*grid[:400])
    result[400:700] = grid[400:700] + 50.0 + 20*np.sin((80)*np.pi*grid[400:700])
    result[700:] = grid[700:] + 120.0 + 20*np.sin((80)*np.pi*grid[700:])

    return result


def comp_emp_bayes_t1d_test_problem():
    """Generates a 1D test vector.
    """

    def func(tvec):
        output = np.ones_like(tvec)
        for i, t in enumerate(tvec):
            if (t >= -2.8) and (t <= -2.1):
                output[i] =  2
            elif (t > -1.6) and (t <= -1.3):
                output[i] = 1.5
            elif t > 0:
                output[i] = 1 + 1.5*np.exp(  - ( (t - 0.5*np.pi)/(2/3) )**2)
            else:
                output[i] = 1
        return output
    
    return func(np.linspace(-np.pi, np.pi, 200))