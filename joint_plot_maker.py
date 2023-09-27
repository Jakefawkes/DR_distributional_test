import os
import yaml
import seaborn as sns
import pandas as pd
from docopt import docopt
from src.utils import get_confidence_interval
import matplotlib.pyplot as plt
import scienceplots
# plt.style.use("science")
sns.set()
fig, axes = plt.subplots(nrows=1,ncols=3,sharey=True,figsize= (19.2,4.8 ))
fig.suptitle("Testing for Distributional Causal Effects on Simulated Data")
moving_param = "beta_scalar"
with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/experiments_for_paper/09-18-16:00:51/scores.metrics", "r") as f:
    results_dict_1 = yaml.safe_load(f)
 
with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/experiments_for_paper/09-06-23:43:45/scores.metrics", "r") as f:
    results_dict_2 = yaml.safe_load(f)    

with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/experiments_for_paper/09-06-23:45:06/scores.metrics", "r") as f:
    results_dict_3 = yaml.safe_load(f) 

results_df_1 = pd.DataFrame(results_dict_1)

results_df_2 = pd.DataFrame(results_dict_2)

results_df_3 = pd.DataFrame(results_dict_3)

sns.lineplot(ax=axes[0],data = results_df_1,x=moving_param,y="result",hue = "test_stat")
sns.lineplot(ax=axes[1],data = results_df_2,x=moving_param,y="result",hue = "test_stat")
sns.lineplot(ax=axes[2],data = results_df_3,x=moving_param,y="result",hue = "test_stat")

axes[0].set_title("(a) Constant Shift")
axes[1].set_title("(b) Bernoulli Shift")
axes[2].set_title("(C) Uniform Shift")

fig.savefig("/vols/ziz/fawkes/DR_distributional_test/sandbox/experiments_for_paper/paper_plot")