"""
Description : 

Usage: wrapped_simulated_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --seed=<seed>                    Random seed.
  --plot                           Outputs plots.
"""

from docopt import docopt
import os
import yaml
import logging
import torch
from src.test import goodness_of_fit_test
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
from src.comparison_models import tmle_test,double_ml_test

def update_cfg(cfg, value):
    if cfg["moving_param"]["beta_scalar"]:
        cfg["data"]["arguments"]["beta_scalar"] = value
    if cfg["moving_param"]["n_train_sample"]:
        cfg["data"]["n_train_sample"] = value
    if cfg["moving_param"]["n_test_sample"]:
        cfg["data"]["n_test_sample"] = value
    return cfg

def construct_dir_name(cfg, value):
    if cfg["moving_param"]["beta_scalar"]:
        return "beta_scalar="+str(value)
    if cfg["moving_param"]["n_train_sample"]:
        return "n_sample="+str(value)
    
weights_model_dict = {"LR": LogisticRegression(), "Adaboost":AdaBoostClassifier(),"Decision_Tree":DecisionTreeClassifier(),"GP":GaussianProcessClassifier(),"MLP":MLPClassifier()}
kernel_dict = {"RBF" : kernel.RBFKernel}

def main(args, cfg,result_dict):
 
    weights_model = weights_model_dict[cfg["experiment"]["weights_model"]]

    X_ker = kernel_dict[cfg["experiment"]["X_ker"]](ard_num_dims=cfg["data"]["dx"])
    Y_ker = kernel_dict[cfg["experiment"]["Y_ker"]](ard_num_dims=cfg["data"]["dy"])
    KMM_weights = cfg["experiment"]["KMM_weights"] 

    cme_reg = cfg["experiment"]["cme_reg"]

    data_train,data_val = make_data(cfg)
    data_train,data_test = make_data(cfg)
    data_full = data_train.join(data_test)
    X_ker.lengthscale = compute_median_heuristic(data_full.X)
    Y_ker.lengthscale = compute_median_heuristic(data_full.Y)

    if type(cme_reg) is list:
        cross_val = True
        reg_param_range = np.linspace(cme_reg[0],cme_reg[1],num=cme_reg[2])  
        cme_reg_0 = cme_cross_validate_weighted(data_train,data_val,X_ker,Y_ker,reg_param_range,T_val=0)
        cme_reg_1 = cme_cross_validate_weighted(data_train,data_val,X_ker,Y_ker,reg_param_range,T_val=1)
        cme_reg = [cme_reg_0,cme_reg_1]
    else: 
        cross_val = False
        cme_reg = [cme_reg,cme_reg]
    
    assement_data_f,assement_data_cf = make_assement_data(cfg)
    assement_data = assement_data_cf.join(assement_data_f)
    
    for i in tqdm.tqdm(range(cfg["experiment"]["n_iter"])):
    

        weights_model.fit(X = data_train.X, y = data_train.T)
        
        stat = "DATE"
        for func in cfg["experiment"]["ker_regress"]:

            result_dict["test_stat"] += [stat+func+"0"]
            result_dict["fit_score"] += [goodness_of_fit_test(assement_data.Y0,data_train,data_test,X_ker,Y_ker,weights_model,test_stat=stat,t=0,reg=cme_reg,func=func,KMM_weights=KMM_weights)]
            
            result_dict["test_stat"] += [stat+func+"1"]
            result_dict["fit_score"] += [goodness_of_fit_test(assement_data.Y1,data_train,data_test,X_ker,Y_ker,weights_model,test_stat=stat,t=1,reg=cme_reg,func=func,KMM_weights=KMM_weights)]

            if cfg["moving_param"]["beta_scalar"]:
                result_dict["beta_scalar"] += [cfg["data"]["arguments"]["beta_scalar"]]
            if cfg["moving_param"]["n_train_sample"]:
                result_dict["n_sample"] += [cfg["data"]["n_train_sample"]]

        stat = "DETT"
        for func in cfg["experiment"]["ker_regress"]:

            result_dict["test_stat"] += [stat+func+"0"]
            result_dict["fit_score"] += [goodness_of_fit_test(assement_data_cf.Y0,data_train,data_test,X_ker,Y_ker,weights_model,test_stat=stat,t=0,reg=cme_reg,func=func,KMM_weights=KMM_weights)]
            
            result_dict["test_stat"] += [stat+func+"1"]
            result_dict["fit_score"] += [goodness_of_fit_test(assement_data_cf.Y1,data_train,data_test,X_ker,Y_ker,weights_model,test_stat=stat,t=1,reg=cme_reg,func=func,KMM_weights=KMM_weights)]

            if cfg["moving_param"]["beta_scalar"]:
                result_dict["beta_scalar"] += [cfg["data"]["arguments"]["beta_scalar"]]
            if cfg["moving_param"]["n_train_sample"]:
                result_dict["n_sample"] += [cfg["data"]["n_train_sample"]]

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

def make_assement_data(cfg):
    if cfg["data"]["generator"] == "shift_data_simulation":
        function_dict = {}
        for key in cfg["data"]["functions"]:
            function_dict[key] = getattr(data,cfg["data"]["functions"][key])
        data_f = data.shift_data_simulation(n_sample=cfg["experiment"]["n_cf_sample"],**function_dict,**cfg["data"]["arguments"],counterfactual=False)    
        data_cf = data.shift_data_simulation(n_sample=cfg["experiment"]["n_cf_sample"],**function_dict,**cfg["data"]["arguments"],counterfactual=True)
    
    if cfg["data"]["generator"] == "linear_data_simulation":
        data_f = data.shift_data_simulation(n_sample=cfg["experiment"]["n_cf_sample"],**function_dict,**cfg["data"]["arguments"],counterfactual=False)
        data_cf = data.linear_data_simulation(n_sample=cfg["experiment"]["n_cf_sample"],**cfg["data"]["arguments"],counterfactual=True)
    return data_f,data_cf

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
    date_time_str = date_time_str.replace(" ","-")
    result_dict = {"test_stat":[], "fit_score":[]}

    if cfg["moving_param"]["beta_scalar"]:
        result_dict["beta_scalar"] = []
    if cfg["moving_param"]["n_train_sample"]:
        result_dict["n_sample"] = []
    direct_path = os.path.join(args['--o'],date_time_str)
    # Create output directory if doesn't exists
    os.makedirs(direct_path, exist_ok=True)
    with open(os.path.join(direct_path, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    for value in cfg["moving_param"]["values"]:
        cfg = update_cfg(cfg, value) 
        weights_model = main(args, cfg,result_dict)
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
    dump_path = os.path.join(direct_path, 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(result_dict, f)
    logging.info(f"\n Dumped scores at {direct_path}")
    # Run session