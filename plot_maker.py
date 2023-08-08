"""
Description : Create plots

Usage: run_kernel_model_linear_data.py  [options] --dir=<path_to_directory>

Options:
  --dir=<path_to_directory>           Path to directory containing results.
"""

import os
import yaml
import seaborn as sns
import pandas as pd
from docopt import docopt


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)
    path_to_dir = args['--dir']
    # Load config file
    os.chdir(path_to_dir)

    with open("scores.metrics", "r") as f:
        results_dict = yaml.safe_load(f)
    with open("cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    results_dict["result"] = [int(p_val <0.05) for p_val in results_dict["p_val"]]
    results_df = pd.DataFrame(results_dict)
    if cfg["moving_param"]["beta_scalar"]:
        moving_param = "beta_scalar"
    if cfg["moving_param"]["n_train_sample"]:
        moving_param = "n_sample"
    plot = sns.lineplot(data = results_df,x=moving_param,y="result",hue = "test_stat")
    fig = plot.get_figure()
    fig.savefig("results_plot")
    

