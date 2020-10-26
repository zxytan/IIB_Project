from GPyOpt.methods import BayesianOptimization
import numpy as np
from numpy.random import randn
import time
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(24)

results = pd.read_excel('gpyopt_test_results.xlsx')

def one_dim(x):
    #1-D
    #optimum at x=0.96609, f(x)=-1.48907
    return -(1.4-3*x)*np.sin(18*x)+randn()*sigma
bounds_one_dim = [{'name':'x', 'type':'continuous', 'domain':(0, 1.2)}]


def booth(X):
    #2-D
    #optimum at X=[1,3] f(X)=0
    X = np.atleast_2d(X)
    x = X[:, 0]
    y = X[:, 1]
    return((x+2*y-7)**2+(2*x+y-5)**2)+randn()*sigma
bounds_booth = [{'name': 'x', 'type': 'continuous', 'domain': (-10,10)}]

def sphere(X):
    #n-D
    #optimum at X = [0,...,0], f(X)=0
    X = np.atleast_2d(X)
    return(np.sum(np.power(X,2), 1))+randn()*sigma
bounds_sphere = [{'name': 'x', 'type': 'continuous', 'domain': (-1,1)}]

def rastrigin(X):
    #n-D
    #optimum at X=[0,...,0], f(X)=0
    X = np.atleast_2d(X)
    return(10*len(X[0])+np.sum(np.power(X,2)-10*np.cos(2*np.pi*X),1))+randn()*sigma
bounds_rastrigin = [{'name': 'x', 'type': 'continuous', 'domain': (-5.12,5.12)}]

"""
#iterate and test with different hyperparameters
n = 20
function = rastrigin
domain = bounds_rastrigin*n
actual_x = [[0]*n]
actual_f = 0
max_time = 600


for sigma in [0, 0.1, 1, 2, 4]:
    for max_iter in [100, 200, 500]:
        for acq_type in ['EI', 'MPI', 'LCB']:
            myOpt = BayesianOptimization(function,
                                        domain=domain,
                                        acquisition_type=acq_type,
                                        exact_feval=True)
            myOpt.run_optimization(max_iter = max_iter, max_time = max_time, eps=1e-6)
            result = {"n": n,"function":"rastrigin", "sigma": sigma, "max_it": max_iter, "max_time": max_time, "acquisition func": acq_type, 
                    "eucl_dist to true x_opt": float(np.linalg.norm(myOpt.x_opt-actual_x)), 
                    "diff to true f(x_opt)": float(np.abs(function(myOpt.x_opt)-actual_f)), "actual_it": myOpt.num_acquisitions,
                    "actual_time": myOpt.cum_time}

            results = results.append(result, ignore_index=True)
            print(myOpt.x_opt, result)
results.to_excel("gpyopt_test_results.xlsx", index=False)
"""
"""
demonstrate with 1D - acquisition, noise effect, convergence
sigma=0
xdata = np.linspace(0, 1.2, 100)
ydata = [one_dim(x) for x in xdata]
plt.plot(xdata, ydata)
plt.show()
sigma = 0.1
myOpt = BayesianOptimization(one_dim, domain=bounds_one_dim, exact_feval=True)
myOpt.run_optimization(max_iter=50, max_time=60, eps=1e-6)
myOpt.plot_acquisition()
myOpt.plot_convergence()
print(myOpt.x_opt, myOpt.fx_opt)
"""
sigma = 0
n=12
myOpt = BayesianOptimization(rastrigin, domain=bounds_rastrigin*n, exact_feval=True)
myOpt.run_optimization(max_iter=500, eps=1e-6)
myOpt.plot_convergence()
print(myOpt.x_opt, myOpt.fx_opt)
