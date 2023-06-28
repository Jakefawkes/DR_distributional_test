import torch 
import gpytorch
from src.test_statistics import *
from src.utils import get_W_matrix
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

def get_binned_weights(weights, n_bins):
    binner = KBinsDiscretizer(n_bins=n_bins,encode = "ordinal")
    binner.fit(weights.reshape(-1,1))
    binned_weights = binner.transform(weights.reshape(-1,1))
    return binned_weights

def binned_permutation(binned_weights):
    perm = np.array((range(len(binned_weights))))
    for i in range(int(binned_weights.max().item())):
        bin_i = ((binned_weights == i).squeeze(1))
        perm[ bin_i ] = np.random.choice(perm[ bin_i ], len(perm[ bin_i ]),replace = False)
    return perm

def kernel_permutation_test(data_train,data_test,X_ker,Y_ker,weights_model,test_stat="DATE",n_bins=10,n_permutations=200,reg=1,permute_weights=False,func="cme"):
    
    weights_train = torch.tensor(weights_model.predict_proba(data_train.X)[:,1]).float()
    weights_test = torch.tensor(weights_model.predict_proba(data_test.X)[:,1]).float()

    if test_stat == "DATE":
        W0 = get_W_matrix(X_ker(data_train.X0).evaluate(),reg,func)
        W1 = get_W_matrix(Y_ker(data_train.X1).evaluate(),reg,func)
        base_stat = DATE_test_stat(data_train,data_test,X_ker,Y_ker,weights_test,W0,W1)
    
    elif test_stat == "DETT":
        W1 = get_W_matrix(Y_ker(data_train.X1).evaluate(),reg,func)
        base_stat = DETT_test_stat(data_train,data_test,X_ker,Y_ker,weights_test,W1)

    binned_weights_train = get_binned_weights(weights_train, n_bins)
    binned_weights_test = get_binned_weights(weights_test, n_bins)

    train_permutation = binned_permutation(binned_weights_train)
    permuted_train_data = data_train.return_permuted_data(train_permutation)
    
    if test_stat == "DATE":
        W0_permuted = get_W_matrix(X_ker(permuted_train_data.X0).evaluate(),reg,func)
        W1_permuted = get_W_matrix(Y_ker(permuted_train_data.X1).evaluate(),reg,func)
    
    elif test_stat == "DETT":
        W1 = get_W_matrix(Y_ker(data_train.X1).evaluate(),reg,func)
        W1_permuted = get_W_matrix(Y_ker(permuted_train_data.X1).evaluate(),reg,func)

    permuted_stats = []

    for i in range(n_permutations):
        test_permutation = binned_permutation(binned_weights_test)
        permuted_test_data = data_test.return_permuted_data(test_permutation)
        if permute_weights:
            test_stat_weight = weights_test[test_permutation]
        else:
            test_stat_weight = weights_test
        if test_stat == "DATE":
            permuted_stats.append(DATE_test_stat(permuted_train_data,permuted_test_data,X_ker,Y_ker,test_stat_weight,W0_permuted,W1_permuted))
        
        elif test_stat == "DETT":
            permuted_stats.append(DETT_test_stat(data_train,permuted_test_data,X_ker,Y_ker,test_stat_weight,W1_permuted))
    
    p_val = np.mean(np.array(permuted_stats)>base_stat)

    return {"p_val": p_val,"stat": base_stat ,"permuted_stats": permuted_stats}
    