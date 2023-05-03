import torch 
import gpytorch

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

def get_W_matrix(K_X,c):
    return torch.cholesky_inverse(K_X + c * K_X.shape[0] * torch.eye(K_X.shape[0]))

def DATE_test_stat(data_train,data_test,X_ker,Y_ker,weights,W0,W1):
        
        m = len(data_test.Y)
        K = ker(X_ker)
        L = ker(Y_ker)

        c = (data_test.T - weights)/(weights*(1-weights))

        test_stat = 0
        test_stat += L(data_test.Y) @ c
        test_stat += 2*torch.diag(weights).T @ (K(data_train.X0,data_test.X).T @ (W0 @ (L(data_train.Y0,data_train.Y)@ c)))
        test_stat += 2*torch.diag(1-weights).T @ (K(data_train.X1,data_test.X).T @ (W1 @ (L(data_train.Y1,data_train.Y) @ c)))
        test_stat += torch.diag(weights).T @ (K(data_train.X0,data_test.X).T @ (W0 @ (L(data_train.Y0,data_train.Y0)@ (W0 @ (K(data_train.X0,data_test.X) @ (torch.diag(weights)@ c))))))
        test_stat += torch.diag(1-weights).T @ (K(data_train.X1,data_test.X).T @ (W1 @ (L(data_train.Y1,data_train.Y1)@ (W1 @ (K(data_train.X1,data_test.X) @ (torch.diag(1-weights)@ c))))))
        test_stat += -2* torch.diag(weights).T @ (K(data_train.X0,data_test.X).T @ (W0 @ (L(data_train.Y0,data_train.Y1)@ (W1 @ (K(data_train.X1,data_test.X) @ (torch.diag(1-weights)@ c))))))    
        test_stat = 1/(m**2) * c @ test_stat 
        return test_stat.item()
 
def DETT_test_stat(data_train,data_test,X_ker,Y_ker,weights,W1):
        
        m = len(data_test.Y)
        K = ker(X_ker)
        L = ker(Y_ker)

        w = (data_test.T - weights)/(weights*(1-weights))

        test_stat = 0
        test_stat += L(data_test.Y) @ w
        test_stat += -2 * K(data_train.X1,data_test.X).T @ (W1 @ (L(data_train.Y1,data_train.Y) @ w))
        test_stat += K(data_train.X1,data_test.X).T @ (W1 @ (L(data_train.Y1,data_train.Y) @ w))
        test_stat += (K(data_train.X1,data_test.X).T @ (W1 @ (L(data_train.Y1,data_train.Y1)@ (W1 @ (K(data_train.X1,data_test.X) @ w)))))
        test_stat = 1/(m**2) * w @ test_stat 
        return test_stat.item()
