from GPyOpt.methods import BayesianOptimization
import numpy as np
from numpy.random import randn
import time
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(24)

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

results = pd.read_excel('gpyopt_test_results.xlsx')

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

# test sparse GP
results = pd.DataFrame(None)
for function in [one_dim, booth, sphere, rastrigin]:
    if function == one_dim:
        ns = [1]
        domain = bounds_one_dim
        actual_x = 0.96609
        actual_f = -1.48907
        func_name = "one_dim"
        iters = [10,20,50,100]
        
    elif function == booth:
        ns = [2]
        domain = bounds_booth
        actual_x = [1,3]
        actual_f = 0
        func_name = "booth"
        iters = [10,20,50,100]
        
    elif function == sphere:
        ns = [5,12]
        domain = bounds_sphere
        func_name = "sphere"
        iters = [10,20,50,100]
        
    elif function == rastrigin:
        ns = [5,12]
        domain = bounds_rastrigin
        func_name = "rastrigin"
        iters = [10,20,50,100,200,500]

    for n in ns:
        domain = domain*n
        if function == sphere or function == rastrigin:
            actual_x = [[0]*n]
            actual_f = 0
            
        for sigma in [0, 0.1, 1]:
            for max_iter in iters:
                
                myOpt = BayesianOptimization(function,
                                            domain=domain,
                                            acquisition_type="EI",
                                            model_type='sparseGP',
                                            exact_feval=True)
                myOpt.run_optimization(max_iter = max_iter, max_time = 600, eps=1e-6)
                result = {"n": n,"function":func_name, "sigma": sigma, "max_it": max_iter, "max_time": 600, "acquisition func": "EI", 
                        "eucl_dist to true x_opt": float(np.linalg.norm(myOpt.x_opt-actual_x)), 
                        "diff to true f(x_opt)": float(np.abs(function(myOpt.x_opt)-actual_f)), "actual_it": myOpt.num_acquisitions,
                        "actual_time": myOpt.cum_time}
        
                results = results.append(result, ignore_index=True)
                print(myOpt.x_opt, result)
            results.to_excel("gpyopt_temp_results.xlsx", index=False)           
