import torch
import math
import pandas as pd
import seaborn as sns
import numpy as np
import os 
from sklearn.preprocessing import StandardScaler

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
    
    def return_shuffled_data(self,permutation):
        return Data_object(self.X[permutation],self.Y[permutation],self.T[permutation])

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
    
    def split(self):
        n_half = math.floor(len(self.T)/2)
        return Data_object(self.X[:n_half],self.Y[:n_half],self.T[:n_half]),Data_object(self.X[n_half:],self.Y[n_half:],self.T[n_half:])
    
def shift_data_simulation(mu,sigma,g_0,g_1,noise,n_sample,counterfactual = False):

    mu = torch.tensor(mu)
    d = len(mu)
    covar = covar_from_text(sigma,d)

    T = torch.concat([torch.zeros(math.floor(n_sample/2)),torch.ones(math.ceil(n_sample/2))])
    if counterfactual:
        T = 1-T
    X0 = torch.randn((math.floor(n_sample/2),d))
    X1 = torch.randn((math.ceil(n_sample/2),d)) @ covar.T + mu
    if counterfactual:
        Y0 = g_1(X0) + noise*torch.randn(g_1(X0).shape)
        Y1 = g_0(X1) + noise*torch.randn(g_0(X1).shape)
    else:
        Y0 = g_0(X0) + noise*torch.randn(g_0(X0).shape)
        Y1 = g_1(X1) + noise*torch.randn(g_1(X1).shape)
    X = torch.concat([X0,X1])
    Y = torch.concat([Y0,Y1])

    return Data_object(X,Y,T)

def linear_data_simulation(alpha_vec,beta_vec,beta_scalar,effect_var,noise_Y,n_sample,counterfactual=False,squared = False):

    alpha_vec = torch.tensor(alpha_vec).float()
    beta_vec = torch.tensor(beta_vec).float()

    d = len(alpha_vec)

    X = torch.randn((n_sample,d)).float()
    
    if squared:
        T = torch.bernoulli(torch.sigmoid( (X @ alpha_vec)**2 - ((X @ alpha_vec)**2).mean() ))
    else:
        T = torch.bernoulli(torch.sigmoid( (X @ alpha_vec) ))

    if effect_var == "Const":
        effect_vec = torch.ones(n_sample)

    if effect_var == "Ber":
        effect_vec = 2*torch.bernoulli(1/2*torch.ones(n_sample))-1

    if effect_var == "Unif":
        effect_vec = 2*torch.rand(n_sample)-1

    if counterfactual:
        # Y =  X @ alpha_vec + beta_scalar * (effect_vec*(1-T)) + noise_Y * torch.randn((n_sample))
        T = 1-T
    Y =  X @ beta_vec + beta_scalar * (effect_vec*(T)) + noise_Y * torch.randn((n_sample))
    
    return Data_object(X,Y,T)

f_0 = lambda X: X[:,0]**2 + X[:,1]  + 3*torch.sin(2*X[:,2]) + X[:,4]
f_1 = lambda X: X[:,0]**2 + 0.8*X[:,1] + 3*torch.sin(2*X[:,2]) + 0.9*X[:,4] 
f_2 = lambda X: X[:,0]**2 + X[:,1] + (X[:,2]>2)
f_3 = lambda X: X[:,0]**2 + X[:,1] + (X[:,2]<(1/2))
f_4 = lambda X: X[:,0]
f_5 = lambda X: torch.sin(X[:,0])+ torch.sin(2*X[:,1])
f_6 = lambda X: torch.sin(X[:,0])+ torch.sin(X[:,1])
f_7 = lambda X: X[:,0]**2 + X[:,1] + X[:,2]>2
f_8 = lambda X: X[:,0]**2 
f_9 = lambda X: X[:,0]**2 + 3

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
    X = torch.tensor(data[["x"+str(i) for  i in range(1,26)]].values,dtype=torch.float)
    IDHP_scalar = StandardScaler().fit(X)
    X_scaled = torch.tensor(IDHP_scalar.transform(X),dtype=torch.float)
    Y = torch.tensor(data["y_factual"],dtype=torch.float)
    T = torch.tensor(data["treatment"],dtype=torch.float)
    Y_cf = torch.tensor(data["y_cfactual"],dtype=torch.float)
    return X_scaled,Y,T,Y_cf

def load_real_data_object(dataset= "IDHP",null_hypothesis = False):

    if dataset == "IDHP":
        X,Y,T,Y_cf = load_IDHP()
    if dataset == "twins":
        X,Y,T,Y_cf = load_twins()

    if null_hypothesis:
        a = (torch.concat([Y.unsqueeze(1),Y_cf.unsqueeze(1)],axis=1))
        rand_mat = torch.rand(a.shape)
        k_th_quant = torch.topk(rand_mat, 1, largest = False)[0][:,-1:]
        mask = rand_mat <= k_th_quant
        Y = a[mask]
    perm = torch.randperm(len(Y))
    prop = torch.randn(len(Y)) > 0 
    X_train, Y_train, T_train = X[perm][prop], Y[perm][prop], T[perm][prop]
    X_test, Y_test, T_test = X[perm][~prop], Y[perm][~prop], T[perm][~prop]
    twin_len= 3000
    twin_len_half = math.floor(twin_len/2)
    if dataset == "twins":
        X_train, Y_train, T_train = X_train[:twin_len_half], Y_train[:twin_len_half], T_train[:twin_len_half]
        X_test, Y_test, T_test = X_test[twin_len_half:twin_len], Y_test[twin_len_half:twin_len], T_test[twin_len_half:twin_len]

    data_train = Data_object(X_train, Y_train, T_train)
    data_test = Data_object(X_test, Y_test, T_test)
    data_full =data_test.join(data_train)
    return data_train, data_test, data_full

def load_twins():
    data = pd.read_csv("https://raw.githubusercontent.com/shalit-lab/Benchmarks/master/Twins/Final_data_twins.csv")
    data = data.dropna()
    X = torch.tensor(data.drop(['T', 'y0', 'y1', 'yf', 'y_cf', 'Propensity'],axis='columns').values,dtype=torch.float)
    twins_scalar = StandardScaler().fit(X)
    X_scaled = torch.tensor(twins_scalar.transform(X),dtype=torch.float)
    print("scaled")
    Y = torch.tensor(data["yf"],dtype=torch.float)
    T = torch.tensor(data["T"],dtype=torch.float)
    Y_cf = torch.tensor(data["y_cf"],dtype=torch.float)
    return X_scaled,Y,T,Y_cf

def twins_data_object(null_hypothesis = False):
    X,Y,T,Y_cf = load_IDHP()
    if null_hypothesis:
        a = (torch.concat([Y.unsqueeze(1),Y_cf.unsqueeze(1)],axis=1))
        rand_mat = torch.rand(a.shape)
        k_th_quant = torch.topk(rand_mat, 1, largest = False)[0][:,-1:]
        mask = rand_mat <= k_th_quant
        Y = a[mask]
    perm = torch.randperm(len(Y))
    prop = torch.randn(len(Y)) > 0 
    X_train, Y_train, T_train = X[perm][prop], Y[perm][prop], T[perm][prop]
    X_test, Y_test, T_test = X[perm][~prop], Y[perm][~prop], T[perm][~prop]
    data_train = Data_object(X_train, Y_train, T_train)
    data_test = Data_object(X_test, Y_test, T_test)
    data_full =data_test.join(data_train)
    return data_train, data_test, data_full

def LBIDD_data_object(size = "1k",null_hypothesis = False):
    X_data = pd.read_csv("data/x.csv")
    Y_data = pd.read_csv("data/"+size+"_f.csv")
    Y_CF_data = pd.read_csv("data/"+size+"_cf.csv")
    data = Y_data.merge(X_data,on="sample_id")
    data = Y_CF_data.merge(data,on="sample_id")
    
    X = torch.tensor(data.drop(["sample_id","y0","y1","z","y"],axis='columns').values,dtype=torch.float)
    Y = torch.tensor(data[["y"]].values,dtype=torch.float).squeeze(1)
    T = torch.tensor(data[["z"]].values,dtype=torch.float).squeeze(1)

    LBIDD_scalar = StandardScaler().fit(X)
    X = torch.tensor(LBIDD_scalar.transform(X),dtype=torch.float)
    print("scaled")
    if null_hypothesis:
        Y_cf = torch.tensor(data[["y0","y1"]].values,dtype=torch.float)
        rand_mat = torch.rand(Y_cf .shape)
        k_th_quant = torch.topk(rand_mat, 1, largest = False)[0][:,-1:]
        mask = rand_mat <= k_th_quant
        Y =  Y_cf[mask]
    perm = torch.randperm(len(Y))
    prop = torch.randn(len(Y)) > 0 
    X_train, Y_train, T_train = X[perm][prop], Y[perm][prop], T[perm][prop]
    X_test, Y_test, T_test = X[perm][~prop], Y[perm][~prop], T[perm][~prop]
    data_train = Data_object(X_train, Y_train, T_train)
    data_test = Data_object(X_test, Y_test, T_test)
    data_full =data_test.join(data_train)
    return data_train, data_test, data_full