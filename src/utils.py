import seaborn as sns 
import pandas as pd
import torch
import os 
from qpth.qp import QPFunction,QPSolvers
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import pairwise_distances
import math
from cvxopt import matrix, solvers

class ker():
    """Implementation of CME which takes in the required matricies and
    kernel as input """

    def __init__(self, k):
          self.k = k
    
    def __call__(self,X1,X2 = None):
            if X2 == None:
                   return self.k(X1).evaluate()
            else:
                   return self.k(X1,X2).evaluate()
            
def get_W_matrix(K_X,c,func,weights = None):
    if weights is None:
         weights = torch.ones(K_X.shape[0])
    if func == "cme":
        return torch.cholesky_inverse(K_X + c * 1/K_X.shape[0] * torch.diag(weights))
    if func == "zero":
        return torch.zeros((K_X.shape[0],K_X.shape[0]))
          
def save_plot_weights(weights_model, data, path,name="weights_plot"):
    df_dict = {"weights" : weights_model.predict_proba(data.X)[:,1], "T" : data.T}
    df = pd.DataFrame(df_dict)
    plot = sns.kdeplot(data = df, x="weights", hue ="T",fill=True, alpha = 0.4)
    save_path = os.path.join(path,name)
    plot.figure.savefig(save_path)
    plot.figure.clf()
    return None

def save_plot_weights_hist(weights_model, data, path,name="weights_plot"):
    df_dict = {"weights" : weights_model.predict_proba(data.X)[:,1], "T" : data.T}
    df = pd.DataFrame(df_dict)
    plot = sns.histplot(data = df, x="weights", hue ="T",fill=True, alpha = 0.4)
    save_path = os.path.join(path,name)
    plot.figure.savefig(save_path)
    plot.figure.clf()
    return None

def cme_cross_validate_target(data_train,data_val,X_ker,Y_ker,reg_param):
        
        K = ker(X_ker)
        L = ker(Y_ker)

        W0 = get_W_matrix(X_ker(data_train.X0).evaluate(),reg_param,"cme")
        W1 = get_W_matrix(X_ker(data_train.X1).evaluate(),reg_param,"cme")

        val_stat_0 = torch.trace(L(data_val.Y0,data_val.Y0) -2 * K(data_train.X0,data_val.X0).T @ (W0 @ (L(data_train.Y0,data_val.Y0))) + (K(data_train.X0,data_val.X0).T @ (W0 @ (L(data_train.Y0,data_train.Y0)@ (W0 @ (K(data_train.X0,data_val.X0)))))))
        val_stat_1 = torch.trace(L(data_val.Y1,data_val.Y1) -2 * K(data_train.X1,data_val.X1).T @ (W1 @ (L(data_train.Y1,data_val.Y1))) + (K(data_train.X1,data_val.X1).T @ (W1 @ (L(data_train.Y1,data_train.Y1)@ (W1 @ (K(data_train.X1,data_val.X1)))))))
        val_stat = val_stat_0+val_stat_1
        return val_stat

def cme_cross_validate(data_train,data_val,X_ker,Y_ker,reg_param_range):
        
        K = ker(X_ker)
        L = ker(Y_ker)

        param_list = []

        for reg_param in reg_param_range:
              param_list.append(cme_cross_validate_target(data_train,data_val,X_ker,Y_ker,reg_param))
        print(dict(zip(reg_param_range,param_list)))
        index_min = min(range(len(param_list)), key=param_list.__getitem__)

        return reg_param_range[index_min]

# def kernel_mean_matching(X_ker, X0, X1 , eps=1, B=100 ):
#     '''
#     An implementation of Kernel Mean Matching, note that this implementation uses its own kernel parameter
#     References:
#     1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." 
#     2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data."
    
#     :param X1: two dimensional sample from population 1
#     :param X2: two dimensional sample from population 2
#     :param kern: kernel to be used, an instance of class Kernel in kernel_utils
#     :param B: upperbound on the solution search space 
#     :param eps: normalization error
#     :return: weight coefficients for instances x1 such that the distribution of weighted x1 matches x2
#     '''
    
#     kernel_X = ker(X_ker)
#     n0 = X0.shape[0]
#     n = X1.shape[0]
#     eps = eps * B / (n0**(-1/2))
#     G = torch.concat([torch.ones(1,n0),-torch.ones(1,n0),torch.eye(n0),-torch.eye(n0)])
#     h = torch.concat([torch.tensor([n0 * (1 + eps),n0 * (eps - 1)]), B * torch.ones((n0,)),torch.zeros(n0) ])
#     e = Variable(torch.Tensor())
#     K_mat = kernel_X(X0, X0)
#     kappa = torch.sum(kernel_X(X0, X1), axis=1) * float(n0) / float(n)      
    
#     coef = QPFunction(verbose=-1,solver= QPSolvers.CVXPY)(K_mat, -kappa, G, h, e, e)
        
#     return coef

def kernel_mean_matching(X_ker, Z, X , eps=1, B=100 ):
    
    kernel_X = ker(X_ker)
    nx = X.shape[0]
    nz = Z.shape[0]

    if eps == None:
        eps = B/math.sqrt(nz)
    
    else:
        eps = eps * B/math.sqrt(nz)
    
    K = np.array(kernel_X(Z, Z).detach(),dtype=np.double)
    kappa = np.sum(np.array(kernel_X(Z, X).detach(),dtype=np.double)*float(nz)/float(nx),axis=1)

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1,nz)), -np.ones((1,nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz*(1+eps), nz*(eps-1), B*np.ones((nz,)), np.zeros((nz,))])
    
    sol = solvers.qp(K, -kappa, G, h)
    coef = torch.tensor(np.array(sol['x']),dtype=torch.float)
    return coef

def KMM_weights_for_W_matrix(X_ker,X0,X,KMM_weights = False):
    if KMM_weights:
        return_weights = kernel_mean_matching(X_ker, X0, X).squeeze(0)
    else: 
        return_weights = torch.ones(X0.shape[0])
    return return_weights

def get_confidence_interval(p,n):
    return [max(p-1.96*((p*(1-p)/n)**(1/2)),0),min(p+1.96*((p*(1-p)/n)**(1/2)),1)]

def cme_cross_validate_target_weighted(data_train,data_val,X_ker,Y_ker,reg_param,T_val=0):
        
        K = ker(X_ker)
        L = ker(Y_ker)

        X_train_data = data_train.X[data_train.T == T_val]
        Y_train_data = data_train.Y[data_train.T == T_val]

        W = get_W_matrix(X_ker(X_train_data).evaluate(),reg_param,"cme")

        X_val_data = data_val.X[data_val.T == T_val]
        Y_val_data = data_val.Y[data_val.T == T_val]

        return_weights = kernel_mean_matching(X_ker, X_val_data, data_val.X).squeeze(0)
        return_weights_matrix = torch.diag(return_weights)

        val_stat = torch.trace(return_weights_matrix@(L(Y_val_data,Y_val_data) -2 * K(X_train_data,X_val_data).T @ (W @ (L(Y_train_data,Y_val_data))) + (K(X_train_data,X_val_data).T @ (W @ (L(Y_train_data,Y_train_data)@ (W @ (K(X_train_data,X_val_data))))))))
        
        return val_stat

def cme_cross_validate_weighted(data_train,data_val,X_ker,Y_ker,reg_param_range,T_val=0):
        
        K = ker(X_ker)
        L = ker(Y_ker)

        param_list = []

        for reg_param in reg_param_range:
              param_list.append(cme_cross_validate_target_weighted(data_train,data_val,X_ker,Y_ker,reg_param,T_val=T_val))
        print(dict(zip(reg_param_range,param_list)))
        index_min = min(range(len(param_list)), key=param_list.__getitem__)

        return reg_param_range[index_min]

def compute_median_heuristic(X):
    if len(X.shape) > 1:
        median_heuristic = [np.median(pairwise_distances(X[:, [i]].reshape(-1,1))) for i in range(X.shape[1])]
    else:
        median_heuristic = np.median(pairwise_distances(X.reshape(-1,1)))
    return torch.tensor(median_heuristic)