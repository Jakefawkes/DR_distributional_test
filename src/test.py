import torch 
from src.test_statistics import *
from src.utils import get_W_matrix,KMM_weights_for_W_matrix
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import clone
import numpy as np
import random
import math

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

def invert_permutation(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv

def weights_tol(weights,weights_minmax):
    weights[weights < weights_minmax] = weights_minmax
    weights[weights > 1-weights_minmax] = 1-weights_minmax
    return weights

def kernel_permutation_test(data_train,data_test,X_ker,Y_ker,weights_model,test_stat="DATE",n_bins=10,n_permutations=200,num_train_permutations=1,reg=[1,1],permute_weights=False,func="cme", KMM_weights = False,weights_minmax=10**(-5)):
    
    weights_model_train = clone(weights_model)
    weights_model_test = clone(weights_model)

    weights_model_train.fit(data_train.X,data_train.T)
    weights_model_test.fit(data_test.X,data_test.T)

    weights_train = weights_tol(torch.tensor(weights_model_test.predict_proba(data_train.X)[:,1]).float(),weights_minmax)
    weights_test = weights_tol(torch.tensor(weights_model_train.predict_proba(data_test.X)[:,1]).float(),weights_minmax)
    data_full = data_test.join(data_train)
    weights = torch.concat([weights_train,weights_test])
    
    shuffle = torch.randperm(len(data_full.T))
    weights = weights[shuffle]
    data_full = data_full.return_shuffled_data(shuffle)
    
    data_train,data_test = data_full.split()
    weights_train_perm,weights_test_perm = weights[:math.floor(len(weights)/2)],weights[math.floor(len(weights)/2):]
    
    weights_model_train.fit(data_train.X,data_train.T)

    weights_test = weights_tol(torch.tensor(weights_model_train.predict_proba(data_test.X)[:,1]).float(),weights_minmax)

    if test_stat == "DATE":
        
        W0_weights = 1/KMM_weights_for_W_matrix(X_ker,data_train.X0,data_train.X,KMM_weights)
        W1_weights = 1/KMM_weights_for_W_matrix(X_ker,data_train.X1,data_train.X,KMM_weights)

        W0 = get_W_matrix(X_ker(data_train.X0).evaluate(),reg[0],func,weights=W0_weights)
        W1 = get_W_matrix(X_ker(data_train.X1).evaluate(),reg[1],func,weights=W1_weights)

        base_stat = DATE_test_stat(data_train,data_test,X_ker,Y_ker,weights_test,W0,W1)
    
    elif test_stat == "DETT":

        W1_weights = 1/KMM_weights_for_W_matrix(X_ker,data_train.X1,data_train.X0,KMM_weights)
        W1 = get_W_matrix(X_ker(data_train.X1).evaluate(),reg[1],func,weights=W1_weights)
        base_stat = DETT_test_stat(data_train,data_test,X_ker,Y_ker,weights_test,W1)
    
    elif test_stat == "Diff":
        base_stat = diff_test_stat(data_train,data_test,X_ker,Y_ker)
        
    binned_weights_train = get_binned_weights(weights_train_perm, n_bins)
    binned_weights_test = get_binned_weights(weights_test_perm, n_bins)

    if test_stat == "DATE":
        permuted_model_list = [(data_train,weights_model_train,W0,W1)]
    
    elif test_stat == "DETT":
        permuted_model_list = [(data_train,weights_model_train,W1)]

    for i in range(num_train_permutations):
        train_permutation = binned_permutation(binned_weights_train)
        permuted_train_data = data_train.return_permuted_data(train_permutation)

        if test_stat == "DATE":

            W0_weights_perm = 1/KMM_weights_for_W_matrix(X_ker,permuted_train_data.X0,permuted_train_data.X,KMM_weights)
            W1_weights_perm = 1/KMM_weights_for_W_matrix(X_ker,permuted_train_data.X1,permuted_train_data.X,KMM_weights)
            
            W0_permuted = get_W_matrix(X_ker(permuted_train_data.X0).evaluate(),reg[0],func,W0_weights_perm)
            W1_permuted = get_W_matrix(X_ker(permuted_train_data.X1).evaluate(),reg[1],func,W1_weights_perm)
            
            permuted_weights_model = clone(weights_model)
            permuted_weights_model.fit(permuted_train_data.X,permuted_train_data.T)
            permuted_model_list.append((permuted_train_data,permuted_weights_model,W0_permuted,W1_permuted))
        
        elif test_stat == "DETT":

            W1_weights_perm = 1/KMM_weights_for_W_matrix(X_ker,permuted_train_data.X1,permuted_train_data.X,KMM_weights)
            
            W1_permuted = get_W_matrix(X_ker(permuted_train_data.X1).evaluate(),reg[1],func,W1_weights_perm)
            permuted_weights_model = clone(weights_model)
            permuted_weights_model.fit(permuted_train_data.X,permuted_train_data.T)
            permuted_model_list.append((permuted_train_data,permuted_weights_model,W1_permuted))

    permuted_stats = [base_stat]

    for i in range(n_permutations):
        test_permutation = binned_permutation(binned_weights_test)
        permuted_test_data = data_test.return_permuted_data(test_permutation)

        sample_index = random.randrange(len(permuted_model_list)-1)

        permuted_models = permuted_model_list[sample_index]
        if test_stat == "DATE":
            test_stat_weight_permuted = weights_tol(torch.tensor(permuted_models[1].predict_proba(permuted_test_data.X)[:,1]).float(),weights_minmax)
            permuted_stats.append(DATE_test_stat(permuted_models[0],permuted_test_data,X_ker,Y_ker,test_stat_weight_permuted,permuted_models[2],permuted_models[3]))

        elif test_stat == "DETT":
            test_stat_weight_permuted = weights_tol(torch.tensor(permuted_models[1].predict_proba(permuted_test_data.X)[:,1]).float(),weights_minmax)
            permuted_stats.append(DETT_test_stat(permuted_models[0],permuted_test_data,X_ker,Y_ker,test_stat_weight_permuted,permuted_models[2]))

        elif test_stat == "Diff":
            train_permutation = binned_permutation(binned_weights_train)
            permuted_train_data_diff = data_test.return_permuted_data(train_permutation)
            permuted_stats.append(diff_test_stat(permuted_train_data_diff,permuted_test_data,X_ker,Y_ker))

    p_val = np.mean(np.array(permuted_stats) >= base_stat)

    return {"p_val": p_val,"stat": base_stat ,"permuted_stats": permuted_stats}

def goodness_of_fit_test(fit_samples,data_train,data_test,X_ker,Y_ker,weights_model,t=1,test_stat="DATE",reg=1,func="cme", KMM_weights = False):
    
    weights = torch.tensor(weights_model.predict_proba(data_test.X)[:,1]).float()

    if test_stat == "DATE":
        
        if t==1:
            W1_weights = 1/KMM_weights_for_W_matrix(X_ker,data_train.X1,data_train.X,KMM_weights)            
            W1 = get_W_matrix(X_ker(data_train.X1).evaluate(),reg[1],func,weights=W1_weights)
            fit_stat = DATE_goodness_of_fit(fit_samples,data_train,data_test,X_ker,Y_ker,weights,W1,t=1)
        else:    
            W0_weights = 1/KMM_weights_for_W_matrix(X_ker,data_train.X0,data_train.X,KMM_weights)            
            W0 = get_W_matrix(X_ker(data_train.X0).evaluate(),reg[0],func,weights=W0_weights)
            fit_stat = DATE_goodness_of_fit(fit_samples,data_train,data_test,X_ker,Y_ker,1-weights,W0,t=0)
    
    elif test_stat == "DETT":
        if t==1:
            W1_weights = 1/KMM_weights_for_W_matrix(X_ker,data_train.X1,data_train.X,KMM_weights)
            W1 = get_W_matrix(X_ker(data_train.X1).evaluate(),reg[1],func,weights=W1_weights)
            fit_stat = DETT_goodness_of_fit(fit_samples,data_train,data_test,X_ker,Y_ker,weights,W1,t=1)
        else:    
            W0_weights = 1/KMM_weights_for_W_matrix(X_ker,data_train.X0,data_train.X,KMM_weights)            
            W0 = get_W_matrix(X_ker(data_train.X0).evaluate(),reg[0],func,weights=W0_weights)
            fit_stat = DETT_goodness_of_fit(fit_samples,data_train,data_test,X_ker,Y_ker,1-weights,W0,t=0)

    return fit_stat

