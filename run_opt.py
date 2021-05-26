from optimise_structure import optimise
import time
import numpy as np
import csv
from structure import structure

#iterate and test with different hyperparameters

for i in range(5):
    for n in [1, 2]:
        for hf_max_iter in [10, 20, 50, 100]:
            for lf_max_iter in [20, 50, 100, 200]:
                for lf_model in ["equiv_cant", "unit"]:
                    if lf_max_iter <= hf_max_iter:
                        continue

                    opt = optimise(n, "beam", lf_model, hf_max_iter, lf_max_iter)

                    start = time.time()
                    opt.run_hf_opt()
                    end = time.time()
                    hf_time = end-start

                    start = time.time()
                    opt.run_lf_opt()
                    end = time.time()
                    lf_time = end-start

                    start = time.time()
                    opt.run_mf_opt()
                    end = time.time()
                    mf_time = end-start

                    opt.report_opt_results()
                    results = opt.results
                    results['hf time'] = hf_time
                    results['lf time'] = lf_time
                    results['mf time'] = mf_time

                    with open('temp_optimisation_results.csv', 'a+') as output_file:
                        print(results)
                        fieldnames = results.keys()
                        dict_writer = csv.DictWriter(output_file, dialect="excel", fieldnames=fieldnames)
                        dict_writer.writerow(results)