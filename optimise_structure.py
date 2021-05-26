from structure import structure
from GPyOpt.methods import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
from GPTutorial.code.gaussian_process import Multifidelity_GP
import csv
import time
from scipy.optimize import minimize

class optimise:
    def __init__(self, n, hf_model, lf_model, hf_max_iter, lf_max_iter, EI_req = 1*10**7):
        """
        model options: beam, unit, equiv_cant
        """
        self.n = n
        self.hf_model = hf_model
        self.lf_model = lf_model
        self.EI_req = EI_req
        self.hf_max_iter = hf_max_iter
        self.lf_max_iter = lf_max_iter
        self.results = {'n': self.n, 'lf model': self.lf_model, 'hf iter': self.hf_max_iter, 'lf iter': self.lf_max_iter}

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
        while mf_model.jitter <= 0.1:
            try:
                mf_model.train()
            except np.linalg.LinAlgError:
                mf_model.jitter = mf_model.jitter*10
            else:
                print(mf_model.jitter)
                break

        def f_mf(X):
            X = np.atleast_2d(X)
            y_pred, y_var = mf_model.predict(X)
            if y_pred[0][0] > 0:
                return(y_pred[0][0])
            else:
                return(10**9)
        
        self.mf_opt = minimize(f_mf, self.hf_opt.x_opt, bounds = [i['domain'] for i in self.domain])
    
    def report_opt_results(self):
        mf = structure(self.hf_model, self.n, self.mf_opt.x, self.EI_req)
        self.results['mf score'] = mf.score
        self.results['mf mass'] = mf.mass
        self.results['mf EI'] = mf.EI
        self.results['mf equiv score'] = mf.equiv_score
        self.results['mf equiv EI'] = mf.equiv_EI
        hf = structure(self.hf_model, self.n, self.hf_opt.x_opt, self.EI_req)
        self.results['hf score'] = hf.score
        self.results['hf mass'] = hf.mass
        self.results['hf EI'] = hf.EI
        self.results['hf equiv score'] = hf.equiv_score
        self.results['hf equiv EI'] = hf.equiv_EI
        lf = structure(self.hf_model, self.n, self.lf_opt.x_opt, self.EI_req)
        self.results['lf score'] = lf.score
        self.results['lf mass'] = lf.mass
        self.results['lf EI'] = lf.EI
        self.results['lf equiv score'] = lf.equiv_score
        self.results['lf equiv EI'] = lf.equiv_EI
