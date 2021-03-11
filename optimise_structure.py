from structure import structure
from GPyOpt.methods import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
from GPTutorial.code.gaussian_process import Multifidelity_GP
import csv
import time

class optimise:
    def __init__(self, n, hf_model, lf_model, hf_max_iter, lf_max_iter, mf_max_iter, EI_req = 1*10**7):
        """
        model options: beam, unit, equiv_cant
        """
        self.n = n
        self.hf_model = hf_model
        self.lf_model = lf_model
        self.EI_req = EI_req
        self.hf_max_iter = hf_max_iter
        self.lf_max_iter = lf_max_iter
        self.mf_max_iter = mf_max_iter
        self.results = {'n': self.n, 'lf model': self.lf_model, 'hf iter': self.hf_max_iter, 'lf iter': self.lf_max_iter,'mf iter': self.mf_max_iter}

        node_domain = [{'name':'node_coord', 'type':'continuous', 'domain':(0, 1)}]
        d_domain = [{'name':'member_d', 'type':'continuous', 'domain':(0, 0.1)}]
        self.domain = node_domain*n*3 + d_domain*int((n+1)*8+n*(n-1)/2)
    
    def run_hf_opt(self):
        def f_hf(X):
            return(structure(self.hf_model, self.n, X[0], self.EI_req).score)
        self.hf_opt = BayesianOptimization(f_hf,
                             domain=self.domain,
                             acquisition_type="EI",
                             model_type='GP',
                             exact_feval=True)
        self.hf_opt.run_optimization(max_iter = self.hf_max_iter, eps=1e-6)
        self.X_H, self.y_H = self.hf_opt.get_evaluations()

    def run_lf_opt(self):
        def f_lf(X):
            return(structure(self.lf_model, self.n, X[0], self.EI_req).score)
        self.lf_opt = BayesianOptimization(f_lf,
                             domain=self.domain,
                             acquisition_type="EI",
                             model_type='GP',
                             exact_feval=True)
        self.lf_opt.run_optimization(max_iter = self.lf_max_iter, eps=1e-6)
        self.X_L, self.y_L = self.lf_opt.get_evaluations()
    
    def run_mf_opt(self):
        X_L, y_L = self.X_L[(self.y_L<1*10**9).T[0],:], self.y_L[(self.y_L<1*10**9).T[0],:]
        X_H, y_H = self.X_H[(self.y_H<1*10**9).T[0],:], self.y_H[(self.y_H<1*10**9).T[0],:]
        mf_model = Multifidelity_GP(X_L, y_L, X_H, y_H)
        mf_model.train()

        def f_mf(X):
            y_pred, y_var = mf_model.predict(X)
            if y_pred[0][0] > 0:
                return(y_pred[0][0])
            else:
                return(10**9)
        
        self.mf_opt = BayesianOptimization(f_mf,
                              X = X_H,
                              Y = y_H,
                             domain=self.domain,
                             acquisition_type="EI",
                             model_type='GP',
                             exact_feval=True)
        self.mf_opt.run_optimization(max_iter = self.mf_max_iter, eps=1e-6)
    
    def report_opt_results(self):
        mf = structure(self.hf_model, self.n, self.mf_opt.x_opt, self.EI_req)
        self.results['mf score'] = mf.score
        self.results['mf mass'] = mf.mystruct.mass
        self.results['mf EI'] = mf.mystruct.get_EI(100)
        hf = structure(self.hf_model, self.n, self.hf_opt.x_opt, self.EI_req)
        self.results['hf score'] = hf.score
        self.results['hf mass'] = hf.mystruct.mass
        self.results['hf EI'] = hf.mystruct.get_EI(100)
        lf = structure(self.hf_model, self.n, self.lf_opt.x_opt, self.EI_req)
        self.results['lf score'] = hf.score
        self.results['lf mass'] = hf.mystruct.mass
        self.results['lf EI'] = hf.mystruct.get_EI(100)
        
    

#iterate and test with different hyperparameters

for n in [1, 2, 3, 4, 5]:
    for hf_max_iter in [10, 20, 50, 100]:
        for lf_max_iter in [20, 50, 100, 200]:
            for mf_max_iter in [50, 100, 200, 400]:
                for lf_model in ["equiv_cant", "unit"]:
                    opt = optimise(n, "beam", lf_model, hf_max_iter, lf_max_iter, mf_max_iter)
                    print(opt.results)

                    start = time.time()
                    opt.run_hf_opt()
                    end = time.time()
                    hf_time = end-start

                    start = time.time()
                    opt.run_lf_opt()
                    end = time.time()
                    lf_time = end-start

                    start = time.time()
                    try:
                        opt.run_mf_opt()
                    except np.linalg.LinAlgError as err:
                        print(err)
                        continue
                    end = time.time()
                    mf_time = end-start

                    opt.report_opt_results()
                    results = opt.results
                    results['hf time'] = hf_time
                    results['lf time'] = lf_time
                    results['mf time'] = mf_time

                    with open('optimisation_results.csv', 'w') as output_file:
                        fieldnames = results.keys()
                        dict_writer = csv.DictWriter(output_file, dialect="excel", fieldnames=fieldnames)
                        dict_writer.writerow(results)
   

