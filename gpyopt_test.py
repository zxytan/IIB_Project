from GPyOpt.methods import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt

#2-Dimensional Example


def f(X):
    x = X[:,0]
    y = X[:,1]
    return((x+2*y-7)**2+(2*x+y-5)**2)

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-10,5)},
{'name':'var_2', 'type':'continuous', 'domain':(0, 10)}]

myOpt = BayesianOptimization(f, domain=bounds)

myOpt.run_optimization(max_iter = 15, max_time = 60, eps=10e-06)

print(myOpt.x_opt, myOpt.fx_opt)

myOpt.plot_acquisition()

myOpt.plot_convergence()


#n-dim example

n=10

def f_high(X):
    return(np.sum(np.power(X,2), 1))

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)}]*n

my_n_Opt = BayesianOptimization(f_high, domain=bounds)

my_n_Opt.run_optimization(max_iter = 15, max_time = 60, eps=10e-06)

print(my_n_Opt.x_opt, my_n_Opt.fx_opt)

my_n_Opt.plot_convergence()