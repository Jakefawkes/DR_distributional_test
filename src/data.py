import torch
import math

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

    

def shift_data_simulation(mu,sigma,g_0,g_1,noise,n_sample):

    mu = torch.tensor(mu)
    d = len(mu)
    covar = covar_from_text(sigma,d)

    T = torch.concat([torch.zeros(math.floor(n_sample/2)),torch.ones(math.ceil(n_sample/2))])

    X0 = torch.randn((math.floor(n_sample/2),d))
    X1 = torch.randn((math.ceil(n_sample/2),d)) @ covar.T + mu

    Y0 = g_0(X0) + noise*torch.randn(g_0(X0).shape)
    Y1 = g_1(X1) + noise*torch.randn(g_1(X1).shape)

    X = torch.concat([X0,X1])
    Y = torch.concat([Y0,Y1])

    return Data_object(X,Y,T)

    
f_0 = lambda X: X[:,0]**2 + X[:,1] 

def covar_from_text(sigma,d):
    if sigma == "ID":
        return torch.eye(d)