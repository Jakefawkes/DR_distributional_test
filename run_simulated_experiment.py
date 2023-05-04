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

weights_model_dict = {"Logistic_Regression": LogisticRegression()}
kernel_dict = {"RBF" : kernel.RBFKernel}

def main(args, cfg):

    weights_model = weights_model_dict[cfg["experiment"]["weights_model"]]
    results_dict = {}

    X_ker = kernel_dict[cfg["experiment"]["X_ker"]]()
    Y_ker = kernel_dict[cfg["experiment"]["Y_ker"]]()

    for i in tqdm.tqdm(range(cfg["experiment"]["n_iter"])):
        data_train,data_test = make_data(cfg)
        for stat in cfg["experiment"]["test_stat"]:
            result = kernel_permutation_test(data_train,data_test,X_ker,Y_ker,weights_model,test_stat=stat)
            results_dict[stat] = results_dict.get(stat,0) + float((result["p_val"] > cfg["experiment"]["significance_level"]))
    for key in results_dict:
        results_dict[key] = results_dict[key]/cfg["experiment"]["n_iter"]
    # Dump scores
    dump_path = os.path.join(args['--o'], 'scores.metrics')
    with open(dump_path, 'w') as f:
        yaml.dump(results_dict, f)
    logging.info(f"\n Dumped scores at {dump_path}")

def make_data(cfg):
    if cfg["data"]["generator"] == "shift_data_simulation":
        function_dict = {}
        for key in cfg["data"]["functions"]:
            function_dict[key] = getattr(data,cfg["data"]["functions"][key])
        data_train = data.shift_data_simulation(n_sample=cfg["data"]["n_train_sample"],**function_dict,**cfg["data"]["arguments"])
        data_test = data.shift_data_simulation(n_sample=cfg["data"]["n_test_sample"],**function_dict,**cfg["data"]["arguments"])
    return data_train,data_test

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

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)
    with open(os.path.join(args['--o'], 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Run session
    main(args, cfg)
