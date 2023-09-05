import torch
import math
import pandas as pd
import seaborn as sns
import numpy as np
import os 
class Data_object():

    def __init__(self,X,Y,T):
        self.X = X
        self.Y = Y
        self.T = T
        self.X0 = X[T==0]
        self.X1 = X[T==1]
        self.Y0 = Y[T==0]
        self.Y1 = Y[T==1]

    def return_permuted_data(self,permutation):
        return Data_object(self.X,self.Y,self.T[permutation])

    def plot_data(self):
        df = pd.DataFrame(np.array(self.X))
        df.columns = ["X"+str(i) for i in df.columns]
        df["Y"] = self.Y
        df["T"] = self.T
        return sns.pairplot(df, hue="T")
    
    def pd_df(self):
        df = pd.DataFrame(np.array(self.X))
        df.columns = ["X"+str(i) for i in df.columns]
        df["Y"] = self.Y
        df["T"] = self.T
        return df
    
    def save_data_plot(self,path,name="data_plot"):
        plot = self.plot_data()
        save_path = os.path.join(path,name)
        plot.savefig(save_path)
        plot.figure.clf()
        return None
    
    def join(self,data_object):
        X_join = torch.concat([data_object.X,self.X])
        Y_join = torch.concat([data_object.Y,self.Y])
        T_join = torch.concat([data_object.T,self.T])
        return Data_object(X_join,Y_join,T_join)

    def flip_T(self):
        self.T = 1-self.T
        self.X0,self.X1 = self.X1,self.X0
        self.Y0,self.Y1 = self.Y1,self.Y0
    
def shift_data_simulation(mu,sigma,g_0,g_1,noise,n_sample,counterfactual = False):

    mu = torch.tensor(mu)
    d = len(mu)
    covar = covar_from_text(sigma,d)

    T = torch.concat([torch.zeros(math.floor(n_sample/2)),torch.ones(math.ceil(n_sample/2))])

    X0 = torch.randn((math.floor(n_sample/2),d))
    X1 = torch.randn((math.ceil(n_sample/2),d)) @ covar.T + mu
    if counterfactual:
        Y1 = g_0(X0) + noise*torch.randn(g_0(X0).shape)
        Y0 = g_1(X1) + noise*torch.randn(g_1(X1).shape)
    else:
        Y0 = g_0(X0) + noise*torch.randn(g_0(X0).shape)
        Y1 = g_1(X1) + noise*torch.randn(g_1(X1).shape)
    X = torch.concat([X0,X1])
    Y = torch.concat([Y0,Y1])

    return Data_object(X,Y,T)

def linear_data_simulation(alpha_vec,beta_vec,beta_scalar,effect_var,noise_Y,n_sample,counterfactual=False):

    alpha_vec = torch.tensor(alpha_vec).float()
    beta_vec = torch.tensor(beta_vec).float()

    d = len(alpha_vec)

    X = torch.randn((n_sample,d)).float()
    T = torch.bernoulli(torch.sigmoid( X @ alpha_vec ))
    
    if effect_var == "Const":
        effect_vec = torch.ones(n_sample)

    if effect_var == "Ber":
        effect_vec = 2*torch.bernoulli(1/2*torch.ones(n_sample))-1

    if effect_var == "Unif":
        effect_vec = 2*torch.rand(n_sample)-1

    if counterfactual:
        Y =  X @ alpha_vec + beta_scalar * (effect_vec*(1-T)) + noise_Y * torch.randn((n_sample))
    else:
        Y =  X @ alpha_vec + beta_scalar * (effect_vec*(1-T)) + noise_Y * torch.randn((n_sample))
    
    return Data_object(X,Y,T)

f_0 = lambda X: X[:,0]**2 + X[:,1] 
f_1 = lambda X: X[:,0]**2 + 0.5*X[:,1] 
f_2 = lambda X: X[:,0]**2 + X[:,1] + (X[:,2]>2)
f_3 = lambda X: X[:,0]**2 + X[:,1] + (X[:,2]<(1/2))
f_4 = lambda X: X[:,0]
f_5 = lambda X: torch.sin(X[:,0])+ torch.sin(2*X[:,1])
f_6 = lambda X: torch.sin(X[:,0])+ torch.sin(X[:,1])
f_7 = lambda X: X[:,0]**2 + X[:,1] + X[:,2]>2


def covar_from_text(sigma,d):
    if sigma[0] == "ID":
        return torch.eye(d)
    if sigma[0] == "C_ID":
        return sigma[1] * torch.eye(d)
    
def load_IDHP():
    data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)
    col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,]
    for i in range(1,26):
        col.append("x"+str(i))
    data.columns = col
    data = data.astype({"treatment":'float'}, copy=False)
    data
    X = torch.tensor(data[["x"+str(i) for  i in range(1,26)]].values)
    Y = torch.tensor(data["y_factual"])
    T = torch.tensor(data["treatment"])
    Y_cf = torch.tensor(data["y_cfactual"])
    return X,Y,T,Y_cf

def IDHP_data_object(null_hypothesis = False):
    X,Y,T,Y_cf = load_IDHP()
    if null_hypothesis:
        rand_mat = torch.rand(a.shape)
        k_th_quant = torch.topk(rand_mat, 1, largest = False)[0][:,-1:]
        mask = rand_mat <= k_th_quant
        Y = (torch.concat([Y.unsqueeze(1),Y_cf.unsqueeze(1)],axis=1))[mask]
    perm = torch.randperm(len(Y))
    prop = torch.randn(len(Y)) > 0 
    X_train, Y_train, T_train = X[perm][prop], Y[perm][prop], T[perm][prop]
    X_test, Y_test, T_test = X[perm][~prop], Y[perm][~prop], T[perm][~prop]
    data_train = Data_object(X_train, Y_train, T_train)
    data_test = Data_object(X_test, Y_test, T_test)
    data_full =data_test.join(data_train)
    return data_train, data_test, data_full

