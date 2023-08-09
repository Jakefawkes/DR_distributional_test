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
