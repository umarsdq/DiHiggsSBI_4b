#!/usr/bin/env python3
"""
Step 6b: Calculate coverage given the network outputs
Converted from 06b_evaluate_coverage.ipynb
"""

import argparse
import sys
import os
import psutil
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt
import scienceplots
plt.style.use("science")
import torch
from numba import cuda 
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import yaml

# Import helper functions
from helpers.likelihood_visualizers import make_c_grid, c_scan_1d, c_scan_2d, c_scan_3d, c_scan_3d_with_confusion
from helpers.test_statistics import get_test_statistic_rate_at_c_points, get_errorbands, get_N_sig_obs_at_c_point
from helpers.network_training import NeuralNet
from helpers.utils import np_to_torch, crop_feature
from helpers.plotting import plot_features


def main():
    parser = argparse.ArgumentParser(description='Evaluate coverage given network outputs')
    parser.add_argument('-p', '--parameter_code', type=str, help='Parameter code (e.g., c0, c1, c2)')
    parser.add_argument('-n', '--number_features', type=int, help='Number of features to use')
    args = parser.parse_args()

    parameter_code = args.parameter_code
    number_features = args.number_features
    
    print(f"Running 06b with parameter_code={parameter_code}, number_features={number_features}")
    
    # Check system resources
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024 ** 3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024 ** 3):.2f} GB")
    
    total, used, free = shutil.disk_usage("/")
    print(f"Total Disk Space: {total / (1024 ** 3):.2f} GB")
    print(f"Used Disk Space: {used / (1024 ** 3):.2f} GB")
    print(f"Free Disk Space: {free / (1024 ** 3):.2f} GB")
    
    # Set up computing environment
    torch.set_num_threads(2)
    device = torch.device("cpu")  # Force CPU usage
    print("Using device: " + str(device), flush=True)
    
    seed = 25
    
    # Load configuration
    run_id = f"dense_s1_{parameter_code}_f{number_features}"
    
    with open("workflow.yaml", "r") as file:
        workflow = yaml.safe_load(file)
        
    with open(f"run_configs/{run_id}.yml", "r") as file:
        run_configs = yaml.safe_load(file)
    
    # Data preparation
    N_train = int(run_configs["bkg.N_train"])
    samples_dir = workflow["sampling"]["output_dir"]
    identity_code = run_configs["input_precode"]
    features = run_configs["features"]
    n_features = len(features)
    parameter_code = run_configs["parameter_code"]
    network_id = run_configs["network_id"]
    
    print(f"Loading data with {n_features} features")
    
    # Load in the samples
    samples_SM = np.load(f'{samples_dir}/plain_real/{identity_code}/{parameter_code}/x_sm.npy')[:,features]
    samples_alt = np.load(f'{samples_dir}/plain_real/{identity_code}/{parameter_code}/x_alt_{parameter_code}.npy')[:,features]
    samples_bkg = np.load(f'{samples_dir}/plain_real/delphes_b0/{parameter_code}/x_bkg.npy')[:,features]
    
    # Load in the theta values
    theta_alt = np.load(f'{samples_dir}/plain_real/{identity_code}/{parameter_code}/theta_alt_{parameter_code}.npy')
    theta_alt_sm = np.load(f'{samples_dir}/plain_real/{identity_code}/{parameter_code}/theta_alt_{parameter_code}.npy')
    
    # Crop to the number of desired signal events
    samples_SM = samples_SM[:N_train]
    samples_alt = samples_alt[:N_train]
    samples_bkg = samples_bkg[:N_train]
    theta_alt = theta_alt[:N_train]
    theta_alt_sm = theta_alt_sm[:N_train]
    
    # Test sets
    if parameter_code == "c0":
        codes = ["m10", "m6", "m4", "p1"]
        targets = [0, -10, -6, -4, 1]
    elif parameter_code == "c1":
        codes = ["m16", "m8", "m4", "p8"]
        targets = [0, -1.6, -0.8, -0.4, 0.8]
    elif parameter_code == "c2":
        codes = ["m2", "p2"]
        targets = [0, -2, 2]
    elif parameter_code == "c0c1":
        if identity_code == "delphes_s":
            codes = ['m10p2p0', 'p3m2p0', 'm4p1p0']
            targets = [[0,0,0],[-1.0, 0.2, 0.0], [0.3, -0.2, 0.0], [-0.4, 0.1, 0.0]]
    elif parameter_code == "c0c2":
        if identity_code == "delphes_s":
            codes = ['m10p0p3', 'p3p0m2', 'm4p0p3']
            targets = [[0,0,0], [-1.0, 0.0, 0.3], [0.3, 0.0, -0.2], [-0.4, 0.0, 0.3]]
    elif parameter_code == "c1c2":
        if identity_code == "delphes_s":
            codes = ['p0m2p2', 'p0m3p1', 'p0m1p3']
            targets = [[0,0,0],[0.0, -0.2, 0.2], [0.0, -0.3, 0.1], [0.0, -0.1, 0.3]]
    
    test_set_codes = ["sm"]
    
    for c in codes:
        test_set_codes.append(f"alt_{parameter_code}_{c}")
    
    print(test_set_codes)
    
    test_sets = {}
    
    for test_code in test_set_codes:
        test_sets[test_code] = shuffle(np.load(f'{samples_dir}/plain_real/{identity_code}/{parameter_code}/x_{test_code}_test.npy')[:,features], random_state=7)
    
    test_sets["bkg"] = shuffle(np.load(f'{samples_dir}/plain_real/delphes_b0/{parameter_code}/x_bkg_test.npy')[:,features], random_state=7)
    
    # Preprocess (Standard Scale)
    with open(f"models/scaler_{network_id}", "rb") as ifile:
        scaler = pickle.load(ifile)
    
    # Transform
    samples_SM = scaler.transform(samples_SM)
    samples_alt = scaler.transform(samples_alt)
    samples_bkg = scaler.transform(samples_bkg)
    
    print(test_set_codes)
    
    for test_code in test_sets.keys():
        test_sets[test_code] = scaler.transform(test_sets[test_code])
    
    # Plot histograms
    nb = 60
    bins = [np.linspace(-2, 5, nb) for i in range(n_features)]
    axis_labels = ["$m_\mathrm{tot}$", "$p_{T_{aa}}$", "$p_{T_{bb}}$", "$\Delta R_{aa}$", "$\Delta R_{bb}$"]
    
    plot_features([samples_SM, samples_alt, samples_bkg], 
                 ["x ~ SM", "x ~ alt", "x ~ bkg"], ["" for i in range(n_features)], nb)
    
    # Plot theta
    fig, ax = plt.subplots(1, 3, figsize=(3*3, 3))
    
    for i in range(3):
        ax[i].hist(theta_alt[:,i], bins=nb, density=True, histtype="step", label="alt", linewidth=2)
        ax[i].set_xlabel(f"$\\theta_{i}$", fontsize=18)
        ax[i].set_yticks([])
        
    ax[0].set_ylabel("Density", fontsize=18)
    #plt.savefig(f"plots/theta_hist_coverage_{parameter_code}_f{n_features}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Evaluation
    # Load in the networks
    seeds = [1, 2, 3, 4, 5]
    seeds = [25]
    
    dense_nets_eval_seeds = {s:{"Ssm_Salt":NeuralNet(n_inputs=n_features + 3, layers=run_configs["network.layers"]), 
                       "B_Salt":NeuralNet(n_inputs=n_features + 3, layers=run_configs["network.layers"]), 
                       "Ssm_B":NeuralNet(n_inputs=n_features, layers=run_configs["network.layers"]), 
                      } for s in seeds}
    
    for key in dense_nets_eval_seeds[1].keys():
        for s in seeds:
            if key == "Ssm_B":
                if len(parameter_code) == 2:
                    checkpoint = torch.load(f"models/dense_s{s}_c0_f{n_features}_{key}_best_model.pt", map_location=device, weights_only=False)
                if len(parameter_code) == 4:
                    checkpoint = torch.load(f"models/dense_s{s}_c0c1_f{n_features}_{key}_best_model.pt", map_location=device, weights_only=False)
            else:
                checkpoint = torch.load(f"models/dense_s{s}_{parameter_code}_f{n_features}_{key}_best_model.pt", map_location=device, weights_only=False)
            dense_nets_eval_seeds[s][key].load_state_dict(checkpoint["model_state_dict"])
            dense_nets_eval_seeds[s][key].eval().to(device)
    
    # Set parameters
    lower_limits = [-1.4, -0.45, -.5]  # 1D
    upper_limits = [.4, 0.55, 0.5]  # 1D
    
    N_sig_SM_target = 3600
    N_bkg_SM_target = 8120
    
    n_coverage = 20
    
    # 1D coverage evaluation
    if len(parameter_code) == 2:
        # Single parameter case (c0, c1, c2)
        pass
    elif len(parameter_code) == 4:
        # Two parameter case (c0c1, c0c2, c1c2) - treat as 1D for coverage
        pass
    
    # Run coverage evaluation for both cases
    if len(parameter_code) == 2 or len(parameter_code) == 4:
        results_dict_rate_only = {}
        results_dict_all = {}
        
        # Make the c_grid
        if len(parameter_code) == 2:
            # Single parameter case
            c_grid, c_scans, edges = make_c_grid(101, lower_limits, upper_limits, parameter_code[-1])
        elif len(parameter_code) == 4:
            # Two parameter case - use 2D grid like in 06a
            c_grid, c_scans, edges = make_c_grid(71, lower_limits, upper_limits, parameter_code[1]+parameter_code[3])
        
        for i, test_code in enumerate(test_set_codes[:]):
            results_dict_rate_only[test_code] = {c:0 for c in range(n_coverage)}
            results_dict_all[test_code] = {c:0 for c in range(n_coverage)}
            
            if len(parameter_code) == 2:
                # Single parameter case
                if parameter_code[-1] == "0":
                    loc_c_point = (targets[i], 0, 0)
                elif parameter_code[-1] == "1":
                    loc_c_point = (0, targets[i], 0)
                elif parameter_code[-1] == "2":
                    loc_c_point = (0, 0, targets[i])
            elif len(parameter_code) == 4:
                # Two parameter case - use the full target coordinate
                loc_c_point = targets[i]
                        
            # Get the number of signal events that we would expect that c point to have
            if len(parameter_code) == 2:
                # Single parameter case
                loc_N_sig_obs = get_N_sig_obs_at_c_point(workflow["sampling"]["input_dir"], identity_code, loc_c_point, N_sig_SM_target)
            elif len(parameter_code) == 4:
                # Two parameter case - scale by 10 like in 06a
                loc_N_sig_obs = get_N_sig_obs_at_c_point(workflow["sampling"]["input_dir"], identity_code, [x*10 for x in targets[i]], N_sig_SM_target)
            
            # Get f_S for the c point
            loc_f_S = loc_N_sig_obs / (N_bkg_SM_target + loc_N_sig_obs)
            
            print(f"test code {test_code} at {loc_c_point} has N_sig {loc_N_sig_obs}, sig frac = {loc_f_S}")
            
            # Generate the test set
            for c in range(n_coverage):
                np.random.seed(c)
                np.random.shuffle(test_sets[test_code])
                np.random.shuffle(test_sets["bkg"])
                
                trial_dataset_N_sig = np.random.poisson(N_sig_SM_target)
                trial_dataset_N_bkg = np.random.poisson(N_bkg_SM_target)
                print("dataset #", c, "// num signal events:",  trial_dataset_N_sig, "// num background events:", trial_dataset_N_bkg)
            
                loc_test_set_sig = test_sets[test_code][:int(trial_dataset_N_sig)]
                loc_test_set_bkg = test_sets["bkg"][:int(trial_dataset_N_bkg)]
                if loc_test_set_sig.shape[0] < int(trial_dataset_N_sig):
                    print("CAUTION: not enough signal events in test set.")
                if loc_test_set_bkg.shape[0] < trial_dataset_N_bkg:
                    print("CAUTION: not enough background events in test set.")
                loc_test_set = np.vstack([loc_test_set_sig, loc_test_set_bkg])
                
                # Evaluate the rate-only test statistic for every point in the c_grid
                if len(parameter_code) == 2:
                    # Single parameter case
                    loc_q_rate, N_sig_c_scan = get_test_statistic_rate_at_c_points(workflow["sampling"]["input_dir"], 
                                                                                  identity_code, 10*c_grid, 
                                                                                  N_sig_SM_target, N_bkg_SM_target, loc_test_set.shape[0])
                elif len(parameter_code) == 4:
                    # Two parameter case - scale by 10 like in 06a
                    loc_q_rate, N_sig_c_scan = get_test_statistic_rate_at_c_points(workflow["sampling"]["input_dir"], 
                                                                                  identity_code, 10*c_grid, 
                                                                                  N_sig_SM_target, N_bkg_SM_target, loc_test_set.shape[0])
                
                results_dict_rate_only[test_code][c] = loc_q_rate
                results_dict_rate_only["cscans"] = c_scans
                
                if len(parameter_code) == 2:
                    # Single parameter case - use c_scan_1d
                    loc_q_all = c_scan_1d(dense_nets_eval_seeds, device, loc_test_set, N_sig_c_scan, N_sig_SM_target, N_bkg_SM_target, c_grid, c_scans, 
                                           parameter_code[-1], seeds, q_rate=loc_q_rate)
                elif len(parameter_code) == 4:
                    # Two parameter case - use c_scan_2d
                    loc_q_all = c_scan_2d(dense_nets_eval_seeds, device, loc_test_set, N_sig_c_scan, N_sig_SM_target, N_bkg_SM_target, c_grid, c_scans, edges,
                                           parameter_code[1]+parameter_code[3], seeds, q_rate=loc_q_rate, target=targets[i])
                
                results_dict_all[test_code][c] = loc_q_all
     
        print("Done")
        
        # Save results
        with open(f"preplot_pickles/coverage_dict_rate_only_{parameter_code}.pkl", "wb") as f:
            pickle.dump(results_dict_rate_only, f)
        
        with open(f"preplot_pickles/coverage_dict_all_{parameter_code}_f{n_features}.pkl", "wb") as f:
            pickle.dump(results_dict_all, f)
    
    print(f"Completed 06b with number_features={number_features}")


if __name__ == "__main__":
    main() 