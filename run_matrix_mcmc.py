from kstacker.orbit import corner_plots_mcmc
from kstacker.mcmc_reoptimization import make_plots
import kstacker.Matrix_Likelihood as ML

import numpy as np
import emcee
import h5py
from joblib import Parallel, delayed
from astropy.io import ascii
import time
from multiprocessing import Pool

def log_posterior(orbital_params):
    """

    Parameters
    ----------
    orbital_params : list
        a list containing a, e, t0, m0, omega, i, theta_0 value given by emcee.
    args : list
        a list containing ts, size, scale, fwhm, bounds, fast, treated_image.
    state : kstacker.Matrix_Likelihood.MCMCState
        manage the state of the sampler value.

    Returns
    -------
    float
       log likelihood and log posterior value for these x values.

    """
    global ts, size, scale, fwhm, bounds, fast, treated_image, state
    state.walker_index = (state.walker_index + 1) % state.n_walkers # index update
    if fast:
        if state.sampler.iteration > 1:
            chain = state.sampler.get_chain()
            accepted = np.any(chain[-1] != chain[-2], axis=1)[state.walker_index]
            if accepted:
                state.cached_terms[state.walker_index][0] = state.cached_terms[state.walker_index][1]
    lp = log_prior(orbital_params,bounds,state)
    if np.isinf(lp):
        return -np.inf
    log_likelihood_value = ML.log_likelihood(orbital_params, ts, size, scale, fwhm, fast, state, treated_image) 
    if np.isinf(log_likelihood_value):
        return -np.inf
    else:
        return lp + log_likelihood_value

def log_prior(orbital_params,bounds,state):
    if not all(bound[0] <= param <= bound[1] for param, bound in zip(orbital_params, bounds)):
        return -np.inf
    return 0

def set_globals(ts_, size_, scale_, fwhm_, bounds_, fast_, treated_image_, state_):
    global ts, size, scale, fwhm, bounds, fast, treated_image, state
    ts = ts_
    size = size_
    scale = scale_
    fwhm = fwhm_
    bounds = bounds_
    fast = fast_
    treated_image = treated_image_
    state = state_

def compute_mcmc_matrix(params, n_jobs=1, n_walkers=28, n_steps=100000, n_orbits=1000, n_check=1000):
    profile_dir = params.get_path("profile_dir")
    values_dir = params.get_path("values_dir")
    ts = np.array(params.get_ts())
    size = params.n
    scale = params.scale
    fwhm = params.fwhm
    data = params.load_data(method="aperture")
    images = data['images']
    N,M,_ = np.shape(images)
    
    treated_image = ML.extract_preteated_image(profile_dir,N,M)
    
    bounds = [
        (10, 70),
        (0, 0.9),
        (-400, 0),
        (1, 2),
        (0, 2*np.pi),
        (0, np.pi),
        (0, 2*np.pi) ]
    
    ndim = len(bounds)
    
    bounds = params.grid.bounds()
    
    with h5py.File(f"{values_dir}/res_grid.h5") as f:
        # note: results are already sorted by decreasing SNR
        results = f["Best solutions"][:]
    
    n_walkers = min(n_walkers, results.shape[0])
    p0 = results[:n_walkers, 0:7].copy()
    ndim = p0.shape[1]
    
    # Define search range
    nbr_psf = 1.
    delta_a = nbr_psf * (bounds[0][1]-bounds[0][0]) / params.grid.limits('a')[2]
    delta_e = nbr_psf * (bounds[1][1]-bounds[1][0]) / params.grid.limits('e')[2]
    delta_t0 = nbr_psf * (bounds[2][1] - bounds[2][0]) / params.grid.limits('t0')[2]
    delta_m0 = (bounds[3][1] - bounds[3][0]) / params.grid.limits('m0')[2]
    delta_omega = nbr_psf * (bounds[4][1] - bounds[4][0]) / params.grid.limits('omega')[2]
    delta_i = nbr_psf * (bounds[5][1] - bounds[5][0]) / params.grid.limits('i')[2]
    delta_theta0 = nbr_psf * (bounds[6][1] - bounds[6][0]) / params.grid.limits('theta_0')[2]
    
    # Loop over each walker and each parameter to add
    # a small random perturbation to create independence between walkers
    for walker in range(n_walkers):
        for param_index, (delta, bound) in enumerate(zip(
                [delta_a, delta_e, delta_t0, delta_m0, delta_omega, delta_i, delta_theta0], bounds)):
    
            # Set up the initial flag for checking bounds
            in_bounds = False
    
            # Loop until the random perturbation is within the bounds
            while not in_bounds:
                # Generate random factor in range [-1, 1]
                random_factor = (np.random.rand() - 0.5) * 2
                perturbation = random_factor * delta
    
                # Add the perturbation to the parameter
                new_value = p0[5, param_index] + perturbation # This line use only one of the best output of  brute-force+gradiant (line 5)
    
                # new_value = p0[walker, param_index] + perturbation # This line use all the outputs of brute-force+gradiant (Doesn't work! before using this line, take into account modulos pi on Omega and omega)
    
                # Check if new values are in the bounds
                if bound[0] <= new_value <= bound[1]:
                    p0[walker, param_index] = new_value
                    in_bounds = True
    
    # mcmc configuration
    
    # Calculate the mean for each column
    means = np.mean(p0, axis=0)
    a_moy, e_moy, t0_moy, m0_moy, omega_moy, i_moy, theta0_moy = means
    bounds = [
        (a_moy - delta_a, a_moy + delta_a),
        (e_moy - delta_e, e_moy + delta_e),
        (t0_moy - delta_t0, t0_moy + delta_t0),
        (m0_moy - delta_m0, m0_moy + delta_m0),
        (omega_moy - delta_omega, omega_moy + delta_omega),
        (max(0, i_moy - delta_i), i_moy + delta_i),
        (theta0_moy - delta_theta0, theta0_moy + delta_theta0) ]
    
    pos = p0
    
    pos = np.array(pos)
    
    state = ML.MCMCState(n_walkers)
    
    state.r_vals, state.j0_vals = ML.precompute_bessel_lookup()
    
    start = time.time()
    
    # pos doit être déjà défini à ce stade !
    ndim = len(bounds)
    
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    set_globals(ts, size, scale, fwhm, bounds, True, treated_image, state)
    state.set_sampler(sampler)
    
    with Pool(processes=n_jobs) as pool:
        sampler.pool = pool
        with open(f"{values_dir}/mcmc_log.txt", "w") as log_file:
            try:
                for i in range(0, n_steps, n_check):
                    pos, _, _ = sampler.run_mcmc(pos, n_check, progress=True)
    
                    if sampler.iteration > 1000:
                        tau = sampler.get_autocorr_time(tol=0)
                        print(f"Step {sampler.iteration}: Autocorrelation time = {tau}")
                        log_file.write(f"Step {sampler.iteration}: Autocorrelation time = {tau}\n")
                        print((tau * 50)/sampler.iteration)
                        log_file.write(f"Step {sampler.iteration}: tau*50/iter = {(tau * 50)/sampler.iteration}\n")
                        print(np.mean(sampler.acceptance_fraction))
                        log_file.write(f"Step {sampler.iteration}: mean acceptance = {np.mean(sampler.acceptance_fraction)}\n")
    
                        if np.all((tau * 50)/sampler.iteration < 1):
                            end = time.time()
                            print("Convergence criteria met")
                            print(f"Time taken : {end-start}")
                            log_file.write("Convergence criteria met\n")
                            log_file.write(f"Time taken : {end-start}")
                            break
                
                print("Convergence criteria not met")
                print(f"Time taken : {end-start}")
                log_file.write("Convergence criteria not met\n")
                log_file.write(f"Time taken : {end-start}")
    
            except Exception as e:
                print(f"An error occurred during MCMC execution: {e}")
                log_file.write(f"An error occurred during MCMC execution: {e}\n")
        
        try:
        
            # Get the final chain of parameters
            samples = sampler.get_chain(flat=True)  # shape: (n_steps * n_walkers, n_params)
            log_probs = sampler.get_log_prob(flat=True)  # shape: (n_steps * n_walkers,)
        
            # Remove invalid values from log_probs
            unique_samples, unique_indices = np.unique(samples, axis=0, return_index=True)
            unique_log_probs = log_probs[unique_indices]
            
            valid_indices = np.isfinite(unique_log_probs)
            filtered_samples = unique_samples[valid_indices]
            filtered_log_probs = unique_log_probs[valid_indices]
            
            # Tri décroissant selon log_prob
            sorted_indices = np.argsort(-filtered_log_probs)
            final_samples = filtered_samples[sorted_indices]
            final_log_probs = filtered_log_probs[sorted_indices]
            
            # Prepare an array to store the top 100 results
            reopt_mcmc = []
            for idx in sorted_indices:
                # Extract parameter values for each of the top 100 samples
                a, e, t0, m0, omega, i, theta_0 = final_samples[idx]
                log_prob = final_log_probs[idx]
                reopt_mcmc.append([idx, log_prob, a, e, t0, m0, omega, i, theta_0])
            
            reopt_mcmc = np.array(reopt_mcmc[:1000])
            # Add index column
            reopt_mcmc = np.concatenate([np.arange(reopt_mcmc.shape[0])[:, None], reopt_mcmc], axis=1)
            # Save results
            names = ("image_number", "best_indice", "log_prob", "a", "e", "t0", "m0", "omega", "i", "theta_0")
            ascii.write(
                reopt_mcmc,
                f"{values_dir}/results_mcmc.txt",
                names=names,
                format="fixed_width_two_line",
                formats={"image_number": "%d"},
                overwrite=True,
            )
         
            # Plots results
            Parallel(n_jobs=n_jobs)(
                delayed(make_plots)(
                    reopt_mcmc[k, 3:], k, params, data["images"], ts, values_dir)        
                for k in range(min(n_orbits, 100))
            )
            
            corner_plots_mcmc(params, nbins=5)
            
            print("Done!")
        
        except ValueError as e:
            print(f"ValueError: {e}")
            log_file.write(f"ValueError: {e}\n")
        
        except IOError as e:
            print(f"File error: {e}")
            log_file.write(f"File error: {e}\n")
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            log_file.write(f"Unexpected error: {e}\n")