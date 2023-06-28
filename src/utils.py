import seaborn as sns 
import pandas as pd
import torch
import os 

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
            
def get_W_matrix(K_X,c,func):
    if func == "cme":
        return torch.cholesky_inverse(K_X + c * K_X.shape[0] * torch.eye(K_X.shape[0]))
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