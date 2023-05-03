
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