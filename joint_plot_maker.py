import os
import yaml
import seaborn as sns
import pandas as pd
from docopt import docopt
from src.utils import get_confidence_interval
import matplotlib.pyplot as plt


def test_stat_map(string):
    if "DATE" in string:
        if "1" in string:
            variable = "{Y(1)}"
        if "0" in string:
            variable = "{Y(0)}"
    if "DETT" in string:
        if "1" in string:
            variable = "{ \{Y(1) \mid T=0 \}}"
        if "0" in string:
            variable = "{ \{ Y(0) \mid T=1 \}}"
    if "cme" in string: 
        return r"$\hat{{\mu}}^{{\mathrm{{DR}}}}_{variable}$".format(variable=variable)
    else:
        return r"$\hat{{\mu}}_{variable}$".format(variable=variable)

test_map_dict = {"DATEcme":"DR-DATE","DETTcme":"DR-DETT","DETTzero":"DETT","DATEzero":"DATE","DML":"DML","TMLE":"TMLE"}


# plt.style.use("science")
sns.set_theme()
sns.set_style("whitegrid")

fig, axes = plt.subplots(nrows=1,ncols=3,sharey=True,figsize= (19.2,4.8 ))
fig.subplots_adjust(wspace=0.05, hspace=0)

fig.suptitle("Testing for Distributional Causal Effects on Simulated Data")

moving_param = "beta_scalar"
with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/experiments_for_paper/10-09-20:54:10/scores.metrics", "r") as f:
    results_dict_1 = yaml.safe_load(f)
results_dict_1["test_stat"] = [test_map_dict[test_stat] for test_stat in results_dict_1["test_stat"]]
with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/experiments_for_paper/10-09-20:55:50/scores.metrics", "r") as f:
    results_dict_2 = yaml.safe_load(f)    
results_dict_2["test_stat"] = [test_map_dict[test_stat] for test_stat in results_dict_2["test_stat"]]

with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/experiments_for_paper/10-09-20:56:12/scores.metrics", "r") as f:
    results_dict_3 = yaml.safe_load(f) 
results_dict_3["test_stat"] = [test_map_dict[test_stat] for test_stat in results_dict_3["test_stat"]]

results_df_1 = pd.DataFrame(results_dict_1)

results_df_2 = pd.DataFrame(results_dict_2)

results_df_3 = pd.DataFrame(results_dict_3)

sns.lineplot(ax=axes[0],data = results_df_1,x=moving_param,y="result",hue = "test_stat")
sns.lineplot(ax=axes[1],data = results_df_2,x=moving_param,y="result",hue = "test_stat")
sns.lineplot(ax=axes[2],data = results_df_3,x=moving_param,y="result",hue = "test_stat")

axes[0].set_title("(a) Constant Shift",y=-0.24)
axes[1].set_title("(b) Bernoulli Shift",y=-0.24)
axes[2].set_title("(C) Uniform Shift",y=-0.24)

axes[0].set_xlabel(r"$\beta$")
axes[1].set_xlabel(r"$\beta$")
axes[2].set_xlabel(r"$\beta$")
axes[0].set_ylabel("Rejection Rate")
for i in range(3):
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles=handles, labels=labels)
for i in range(1,3):
    axes[i].legend([],[], frameon=False)
fig.savefig("/vols/ziz/fawkes/DR_distributional_test/sandbox/experiments_for_paper/paper_plot.png",bbox_inches='tight')   
plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(nrows=1,ncols=4,sharey=False,figsize= (25.6,4.8 ))
fig.subplots_adjust(wspace=0.17, hspace=0)
fig.suptitle("Assesing fit of Doubly Robust Counterfactual Mean Embeddings on Simulated Data")

with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/fit_tests/10-08-20:25:12/scores.metrics", "r") as f:
    results_dict = yaml.safe_load(f)
results_dict["test_stat"] = [test_stat_map(test_stat) for test_stat in results_dict["test_stat"]] 
results_df = pd.DataFrame(results_dict)
plot = sns.lineplot(data = results_df,x="n_sample",y="fit_score",hue = "test_stat", ax=axes[0])
axes[0].set_ylabel("RKHS distance from True Embedddings")
axes[0].set_xlabel("Number of Samples")

axes[0].set_title("(a) DATE embeddings, \n Both sided overlap,\n Incorrectly specified propensity",y=-0.36,fontsize=15)

with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/fit_tests/10-08-17:17:09/scores.metrics", "r") as f:
    results_dict = yaml.safe_load(f)
results_dict["test_stat"] = [test_stat_map(test_stat) for test_stat in results_dict["test_stat"]] 
results_df = pd.DataFrame(results_dict)
plot = sns.lineplot(data = results_df,x="n_sample",y="fit_score",hue = "test_stat", ax=axes[1])
axes[1].set_ylabel("")
axes[1].set_xlabel("Number of Samples")
axes[1].set_title("(b) DETT embeddings, \n Both sided overlap, \n Incorrectly specified propensity",y=-0.36,fontsize=15)

with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/fit_tests/10-08-18:19:12/scores.metrics", "r") as f:
    results_dict = yaml.safe_load(f)
results_dict["test_stat"] = [test_stat_map(test_stat) for test_stat in results_dict["test_stat"]] 
results_df = pd.DataFrame(results_dict)
plot = sns.lineplot(data = results_df,x="n_sample",y="fit_score",hue = "test_stat", ax=axes[2])
axes[2].set_xlabel("Number of Samples")
axes[2].set_title("(c) DATE embeddings, \n One sided overlap,\n Incorrectly specified propensity",y=-0.36,fontsize=15)
axes[2].set_ylabel("")
with open("/vols/ziz/fawkes/DR_distributional_test/sandbox/fit_tests/10-08-20:25:43/scores.metrics", "r") as f:
    results_dict = yaml.safe_load(f)
results_dict["test_stat"] = [test_stat_map(test_stat) for test_stat in results_dict["test_stat"]] 
results_df = pd.DataFrame(results_dict)
plot = sns.lineplot(data = results_df,x="n_sample",y="fit_score",hue = "test_stat", ax=axes[3])
axes[3].set_title("(d) DETT embeddings, \n One sided overlap,\n Incorrectly specified propensity",y=-0.36,fontsize=15)
axes[3].set_ylabel("")
axes[2].set_xlabel("Number of Samples")

for i in range(4):
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles=handles, labels=labels,fontsize=14)

fig.savefig("/vols/ziz/fawkes/DR_distributional_test/sandbox/fit_tests",bbox_inches='tight')   
