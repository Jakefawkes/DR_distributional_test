"""
Description : Create plots

Usage: run_kernel_model_linear_data.py  [options] --dir=<path_to_directory>

Options:
  --dir=<path_to_directory>           Path to directory containing results.
"""

import os
import yaml
import logging
from docopt import docopt
from src.utils import *

if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)
    path_to_dir = args['--dir']
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

