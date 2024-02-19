
"""
Optimize the airfoil shape in the latent space using Bayesian optimization, 
constrained on the running time

Author(s): Wei Chen (wchen459@umd.edu)
"""

from __future__ import division
import sys
sys.path.append('./')
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from option import opt as args

from bayesoptim.bayesian_opt import optimize as optimize_latent
from bayesoptim.functions import AirfoilDiffusion
from bayesoptim.genetic_alg import generate_first_population, select, create_children, mutate_population

def optimize_overall(noise0, perturb_type, perturb, n_eval, func):
    
    # Optimize in the latent+noise combined space
    n_best = 30
    n_random = 10
    n_children = 5
    chance_of_mutation = 0.1
    population_size = int((n_best+n_random)/2*n_children)
    x0 = noise0
    init_perf = func(x0)
    population = generate_first_population(x0, population_size, perturb_type, perturb)
    best_inds = [x0]
    best_perfs = [init_perf]
    opt_perfs = []
    i = 0
    print('Initial: x {} CL/CD {}'.format(x0, init_perf))
    while 1:
        breeders, best_perf, best_individual = select(population, n_best, n_random, func)
        best_inds.append(best_individual)
        best_perfs.append(best_perf)
        opt_perfs += [np.max(best_perfs)] * population_size # Best performance so far
        print('%d: fittest %.2f' % (i+1, best_perf))
        # No need to create next generation for the last generation
        if i < n_eval/population_size-1:
            next_generation = create_children(breeders, n_children)
            population = mutate_population(next_generation, chance_of_mutation, perturb_type, perturb)
            i += 1
        else:
            break
    
    opt_x = best_inds[np.argmax(best_perfs)]
    opt_airfoil = func.synthesize(opt_x)
    print('Optimal CL/CD: {}'.format(opt_perfs[-1]))
    
    return opt_x, opt_airfoil, opt_perfs


if __name__ == "__main__":
    
    n_runs = args.n_runs
    n_eval = args.n_eval
    n_init_eval_latent = 10
    n_eval_latent = n_eval
    n_eval_overall = n_eval
    
    opt_x_runs = []
    opt_airfoil_runs = []
    opt_perfs_runs = []
    time_runs = []
    
    for i in range(n_runs):
                
        print('')
        print('######################################################')
        print('# Method: Diffusion-TSO')
        print('# Run: {}/{}'.format(i+1, n_runs))
        print('######################################################')
                
        successful = False
        while not successful:
            try:
                start_time = time.time()
                # Optimize in the latent space
                func = AirfoilDiffusion()
                opt_latent, opt_airfoil, opt_perfs_latent = optimize_latent(n_eval_latent, n_init_eval_latent, func)
                np.savetxt(f'bayesoptim/bo_{i}.dat', opt_airfoil, header=f'bo_{i}', comments="")
                # Optimize in the latent+noise combined space
                func = AirfoilDiffusion()
                noise0 = opt_latent
                perturb_type = 'absolute'
                perturb = 1.0
                opt_x, opt_airfoil, opt_perfs_overall = optimize_overall(noise0, perturb_type, perturb, n_eval_overall, func)
                np.savetxt(f'bayesoptim/bo_refine_{i}.dat', opt_airfoil, header=f'bo_refine{i}', comments="")
                end_time = time.time()
                opt_x_runs.append(opt_x)
                opt_airfoil_runs.append(opt_airfoil)
                opt_perfs = opt_perfs_latent.tolist() + opt_perfs_overall
                opt_perfs_runs.append(opt_perfs)
                time_runs.append(end_time-start_time)
                successful = True
            except Exception as e:
                print(e)
    