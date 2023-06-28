"""
Description : Runs kernel ridge regression model with linear data generating process

Usage: run_kernel_model_linear_data.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --seed=<seed>                    Random seed.
  --plot                           Outputs plots.
"""

import os
import yaml
import logging
from docopt import docopt
import torch
from src.test import kernel_permutation_test
import src.data as data
import tqdm
import gpytorch.kernels as kernel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from datetime import datetime
from src.utils import *

weights_model_dict = {"Logistic_Regression": LogisticRegression(), "Adaboost":AdaBoostClassifier(),"Decision_Tree":DecisionTreeClassifier(),"GP":GaussianProcessClassifier(),"MLP":MLPClassifier()}
kernel_dict = {"RBF" : kernel.RBFKernel}

def main(args, cfg,direct_path):

    weights_model = weights_model_dict[cfg["experiment"]["weights_model"]]
    results_dict = {}

    X_ker = kernel_dict[cfg["experiment"]["X_ker"]]()
    Y_ker = kernel_dict[cfg["experiment"]["Y_ker"]]()
    n_bins = cfg["experiment"]["n_bins"]
    permute_weights = cfg["experiment"]["permute_weights"] 
    if permute_weights:
        print("Permuting")
    if not permute_weights:
        print("Not Permuting")
    cme_reg = cfg["experiment"]["cme_reg"]

    if type(cme_reg) is list:
        cross_val = True
        reg_param_range = np.linspace(cme_reg[0],cme_reg[1],num=cme_reg[2]) 
        data_train,data_val = make_data(cfg) 
        cme_reg = cme_cross_validate(data_train,data_val,X_ker,Y_ker,reg_param_range)
        print(cme_reg)
    else: 
        cross_val = False

    for i in tqdm.tqdm(range(cfg["experiment"]["n_iter"])):
        data_train,data_test = make_data(cfg)
        weights_model.fit(X= data_train.X, y=data_train.T)
        for stat in cfg["experiment"]["test_stat"]:
            for func in cfg["experiment"]["ker_regress"]:
                result = kernel_permutation_test(data_train,data_test,X_ker,Y_ker,weights_model,test_stat=stat,n_bins =n_bins,permute_weights=permute_weights , reg=cme_reg,func = func)
                results_dict[stat+func] = results_dict.get(stat+func,0) + float((result["p_val"] < cfg["experiment"]["significance_level"]))
    for stat in cfg["experiment"]["test_stat"]:
            for func in cfg["experiment"]["ker_regress"]:
                results_dict[stat+func] = results_dict[stat+func]/cfg["experiment"]["n_iter"]
                results_dict[stat+func+"CI"] = get_confidence_interval(results_dict[stat+func],cfg["experiment"]["n_iter"])
    # Dump scores
    if cross_val:
        results_dict["cme_reg"] = cme_reg.item()
    dump_path = os.path.join(direct_path, 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(results_dict, f)
    logging.info(f"\n Dumped scores at {direct_path}")
    return weights_model

def make_data(cfg):
    if cfg["data"]["generator"] == "shift_data_simulation":
        function_dict = {}
        for key in cfg["data"]["functions"]:
            function_dict[key] = getattr(data,cfg["data"]["functions"][key])
        data_train = data.shift_data_simulation(n_sample=cfg["data"]["n_train_sample"],**function_dict,**cfg["data"]["arguments"])
        data_test = data.shift_data_simulation(n_sample=cfg["data"]["n_test_sample"],**function_dict,**cfg["data"]["arguments"])
    
    if cfg["data"]["generator"] == "linear_data_simulation":
        data_train = data.linear_data_simulation(n_sample=cfg["data"]["n_train_sample"],**cfg["data"]["arguments"])
        data_test = data.linear_data_simulation(n_sample=cfg["data"]["n_test_sample"],**cfg["data"]["arguments"])
    return data_train,data_test

def get_confidence_interval(p,n):
    return [max(p-1.96*((p*(1-p)/n)**(1/2)),0),min(p+1.96*((p*(1-p)/n)**(1/2)),1)]

if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')
    logging.info(f'Configuration file: {cfg}\n')

    now = datetime.now()
    date_time_str = now.strftime("%m-%d %H:%M:%S")
    direct_path = os.path.join(args['--o'],date_time_str)
    # Create output directory if doesn't exists
    os.makedirs(direct_path, exist_ok=True)
    with open(os.path.join(direct_path, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    weights_model = main(args, cfg,direct_path)
    if args["--plot"]:
        function_dict = {}

        if cfg["data"]["generator"] == "shift_data_simulation":
            for key in cfg["data"]["functions"]:
                function_dict[key] = getattr(data,cfg["data"]["functions"][key])

            data_plot = data.shift_data_simulation(n_sample=1000,**function_dict,**cfg["data"]["arguments"])
            data_plot.save_data_plot(direct_path)
            save_plot_weights_hist(weights_model, data_plot,direct_path)

        if cfg["data"]["generator"] == "linear_data_simulation":
            data_plot = data.linear_data_simulation(n_sample=1000,**cfg["data"]["arguments"])
            data_plot.save_data_plot(direct_path)
            save_plot_weights_hist(weights_model, data_plot,direct_path)
    # Run session

