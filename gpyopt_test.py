from GPyOpt.methods import BayesianOptimization
import numpy as np
import time

#2-Dimensional Example


def f(X):
    #Booth function
    #optimum at X=[1,3] f(X)=0
    x = X[:,0]
    y = X[:,1]
    return((x+2*y-7)**2+(2*x+y-5)**2)

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-10,5)},
{'name':'var_2', 'type':'continuous', 'domain':(0, 10)}]

myOpt = BayesianOptimization(f, domain=bounds)
tic = time.perf_counter()
myOpt.run_optimization(max_iter = 15, max_time = 60, eps=10e-06)
toc = time.perf_counter()
print(toc-tic)
print(myOpt.x_opt, myOpt.fx_opt)

myOpt.plot_acquisition()

myOpt.plot_convergence()


#n-dim example

n=10

def f_high(X):
    #Sphere function
    #optimum at X = [0,...,0], f(X)=0
    return(np.sum(np.power(X,2), 1))

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)}]*n

my_n_Opt = BayesianOptimization(f_high, domain=bounds)

tic = time.perf_counter()
my_n_Opt.run_optimization(max_iter = 15, max_time = 60, eps=10e-06)
toc = time.perf_counter()
print(toc-tic)

print(my_n_Opt.x_opt, my_n_Opt.fx_opt)

my_n_Opt.plot_convergence()

#n-dim example with local minima

def f_high_rough(X):
    #Rastrigin function
    #optimum at X=[0,...,0], f(X)=0
    return(10*n+np.sum(np.power(X,2)-10*np.cos(2*np.pi*X),1))

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-5.12,5.12)}]*n

my_n_rough_Opt = BayesianOptimization(f_high, domain=bounds)

tic = time.perf_counter()
my_n_rough_Opt.run_optimization(max_iter = 40, max_time = 60, eps=10e-06)
toc = time.perf_counter()
print(toc-tic)

print(my_n_rough_Opt.x_opt, my_n_rough_Opt.fx_opt)

my_n_rough_Opt.plot_convergence()
