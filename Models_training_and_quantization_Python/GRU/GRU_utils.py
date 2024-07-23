import numpy as np
import csv
import scipy.stats as stats
from scipy import integrate, special as sp
from scipy.optimize import fminbound

# Function to create a dataset
def create_dataset(P1, sgm, M, adelta, g, amin):
    # Generate random binary signals based on probability P1
    sns = np.array([1 if i <= P1 else 0 for i in np.random.rand(M)])
    # Generate amplitude values for signals
    A = amin + np.multiply(adelta, np.random.rand(np.sum(sns)))
    A = np.resize(A, (len(A), 1)).T

    # Create dataset of signals with added noise
    dataset_segnali = np.dot(sgm, np.random.randn(len(g), M))
    dataset_segnali = np.vstack([dataset_segnali, sns]).T

    z = 0
    for n, i in enumerate(dataset_segnali):
        if list(i)[-1] == 1:
            dataset_segnali[n][:-1] = i[:-1] + (np.dot(g, A))[:, z]
            z += 1

    return dataset_segnali

# Q-function definition
def qfunc(arg):
    return 0.5 - 0.5 * sp.erf(arg / 1.414)

# Function to calculate mean and confidence interval
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    std = np.std(a)
    return m, std

# Function to calculate parameters Ta and Pe
def calcolaParametri(g, Sigma, amin, amax, P1, sigma_noise):
    adelta = amax - amin  # Range of amplitude values
    P2 = 1 - P1  # Probability of 0 signal
    S = np.dot(g.T, (np.linalg.solve(Sigma, g)))  # Signal to noise ratio matrix

    # Function to calculate mean value of amplitude
    ma = lambda x=None: np.dot(x, S)
    # Threshold function for amplitude
    Ta = lambda x=None: np.dot(np.dot(0.5, x), S) + np.log(P2 / P1) / x
    # Probability error function
    f = lambda a=None, alpha=None: np.dot(P2, qfunc(Ta(alpha) / np.sqrt(S))) - np.dot(P1, qfunc((Ta(alpha) - ma(a)) / np.sqrt(S)))

    # Integral of the probability error function
    int_f = lambda alpha=None: integrate.quad(lambda x=None: f(x, alpha), amin, amax)[0] / adelta
    # Minimize the integral to find the optimal threshold
    min_x = fminbound(int_f, sigma_noise, amax)
    # Calculate probability of error
    Pe = P1 + int_f(min_x)
    # Final threshold value
    Ta = Ta(min_x)

    return Ta, Pe
