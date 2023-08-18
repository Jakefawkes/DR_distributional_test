import torch 
from src.utils import ker


def DATE_test_stat(data_train,data_test,X_ker,Y_ker,weights,W0,W1):
        
        m = len(data_test.Y)
        K = ker(X_ker)
        L = ker(Y_ker)

        c = (data_test.T - weights)/(weights*(1-weights))

        test_stat = 0
        test_stat += L(data_test.Y) @ c
        test_stat += 2*torch.diag(weights).T @ (K(data_train.X0,data_test.X).T @ (W0 @ (L(data_train.Y0,data_test.Y)@ c)))
        test_stat += -2*torch.diag(1-weights).T @ (K(data_train.X1,data_test.X).T @ (W1 @ (L(data_train.Y1,data_test.Y) @ c)))
        test_stat += torch.diag(weights).T @ (K(data_train.X0,data_test.X).T @ (W0 @ (L(data_train.Y0,data_train.Y0)@ (W0 @ (K(data_train.X0,data_test.X) @ (torch.diag(weights)@ c))))))
        test_stat += torch.diag(1-weights).T @ (K(data_train.X1,data_test.X).T @ (W1 @ (L(data_train.Y1,data_train.Y1)@ (W1 @ (K(data_train.X1,data_test.X) @ (torch.diag(1-weights)@ c))))))
        test_stat += -2* torch.diag(weights).T @ (K(data_train.X0,data_test.X).T @ (W0 @ (L(data_train.Y0,data_train.Y1)@ (W1 @ (K(data_train.X1,data_test.X) @ (torch.diag(1-weights)@ c))))))    
        test_stat = 1/(m**2) * c @ test_stat 
        return test_stat.item()
 
def DETT_test_stat(data_train,data_test,X_ker,Y_ker,weights,W1):
        
        m = len(data_test.Y)
        K = ker(X_ker)
        L = ker(Y_ker)

        w = (data_test.T - weights)/(weights)

        test_stat = 0
        test_stat += L(data_test.Y) @ w
        test_stat += -2 * K(data_train.X1,data_test.X).T @ (W1 @ (L(data_train.Y1,data_test.Y) @ w))
        test_stat += (K(data_train.X1,data_test.X).T @ (W1 @ (L(data_train.Y1,data_train.Y1)@ (W1 @ (K(data_train.X1,data_test.X) @ w)))))
        test_stat = 1/(m**2) * w @ test_stat 
        return test_stat.item()

def DATE_goodness_of_fit(fit_samples,data_train,data_test,X_ker,Y_ker,weights,Wt,t=1):
        
        if t==0:
                data_test.flip_T()
                data_train.flip_T()

        n = len(fit_samples)
        m = len(data_test.Y)

        K = ker(X_ker)
        L = ker(Y_ker)

        t_vec = (data_test.T == t).long()
        w = (t_vec)*1/(weights)

        fit_stat = 1/(n**2)*L(fit_samples,fit_samples).sum()
        fit_stat += -2/(n*m)*(w @ L(data_test.Y,fit_samples)).sum()
        fit_stat += -2/(n*m)*((L(fit_samples,data_train.Y1) @ (Wt @ (K(data_train.X1,data_test.X) @ (1-w))))).sum()
        fit_stat += 1/(m**2) * (w @ (L(data_test.Y,data_test.Y)@ w))
        fit_stat += 1/(m**2)*2 * (w @ (L(data_test.Y,data_train.Y1) @ (Wt @ (K(data_train.X1,data_test.X)@ (1-w)))))
        fit_stat += 1/(m**2)*((1-w) @ (K(data_test.X,data_train.X1) @ (Wt @ ( L(data_train.Y1,data_train.Y1) @ (Wt @ (K(data_train.X1,data_test.X)@ (1-w)) ) ))))
        return fit_stat.item()

def DETT_goodness_of_fit(fit_samples,data_train,data_test,X_ker,Y_ker,weights,Wt,t=1):
        
        if t==0:
                data_test.flip_T()
                data_train.flip_T()
                
        n = len(fit_samples)
        m = len(data_test.Y0)

        K = ker(X_ker)
        L = ker(Y_ker)

        w = data_test.T * (1-weights)/(weights)

        fit_stat = 1/(n**2)*L(fit_samples,fit_samples).sum()
        fit_stat += -2/(n*m)*(w @ L(data_test.Y,fit_samples)).sum()
        fit_stat += -2/(n*m)*((L(fit_samples,data_train.Y1) @ (Wt @ (K(data_train.X1,data_test.X) @ (1-t-w))))).sum()
        fit_stat += 1/(m**2) * (w @ (L(data_test.Y,data_test.Y)@ w))
        fit_stat += 1/(m**2)*2 * (w @ (L(data_test.Y,data_train.Y1) @ (Wt @ (K(data_train.X1,data_test.X)@ (1-t-w)))))
        fit_stat += 1/(m**2)*((1-t-w) @ (K(data_test.X,data_train.X1) @ (Wt @ ( L(data_train.Y1,data_train.Y1) @ (Wt @ (K(data_train.X1,data_test.X)@ (1-t-w)) ) ))))
        return fit_stat.item()