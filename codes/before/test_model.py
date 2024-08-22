import torch
import time
from torch import nn
import scipy.sparse as sp
import numpy as np
from sparsesvd import sparsesvd
from torchdiffeq import odeint
import datasets

class BSPM(object):
    def __init__(self, adj_mat):
        self.adj_mat = adj_mat

        self.idl_solver = 'euler'
        self.blur_solver = 'euler'
        self.sharpen_solver = 'euler'
        
        self.idl_beta = 0.3
        self.factor_dim = 128
        print(r"IDL factor_dim: ",self.factor_dim)
        print(r"IDL $\beta$: ",self.idl_beta)
        idl_T = 1
        idl_K = 1
        
        blur_T = 1
        blur_K = 1
        
        sharpen_T = 1
        sharpen_K = 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.idl_times = torch.linspace(0, idl_T, idl_K+1).float().to(self.device)
        print("idl time: ",self.idl_times)
        self.blurring_times = torch.linspace(0, blur_T, blur_K+1).float().to(self.device)
        print("blur time: ",self.blurring_times)
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K+1).float().to(self.device)
        print("sharpen time: ",self.sharpening_times)

    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.01
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        del norm_adj, d_mat
        ut, s, self.vt = sparsesvd(self.norm_adj, self.factor_dim)

        linear_Filter = self.norm_adj.T @ self.norm_adj
        self.linear_Filter = self.convert_sp_mat_to_sp_tensor(linear_Filter).to_dense().to(self.device)

        left_mat = self.d_mat_i @ self.vt.T
        right_mat = self.vt @ self.d_mat_i_inv
        self.left_mat, self.right_mat = torch.FloatTensor(left_mat).to(self.device), torch.FloatTensor(right_mat).to(self.device)
        end = time.time()
        print('pre-processing time for BSPM', end-start)
    
    def sharpenFunction(self, t, r):
        out = r @ self.linear_Filter
        return -out

    def getUsersRating(self, batch_users):
        batch_test = batch_users.to(self.device)

        with torch.no_grad():
            idl_out = torch.mm(batch_test, self.left_mat @  self.right_mat)

            blurred_out = torch.mm(batch_test, self.linear_Filter)
            # blurred_out = torch.mm(batch_test.to_dense(), self.linear_Filter)
            del batch_test
            
            sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta*idl_out+blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
            # sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
        U_2 =  torch.mean(torch.cat([blurred_out.unsqueeze(0),sharpened_out[1:,...]],axis=0),axis=0)
        
        # U_2 =  blurred_out
        # del blurred_out

        # U_2 = sharpened_out[-1]
        # del sharpened_out

        # ret = U_2
        # del U_2
        ret = self.idl_beta * idl_out + U_2
        
        return ret

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
class BSPM_torch(object):
    def __init__(self, adj_mat):
        self.adj_mat = sp.csr_matrix(np.array(adj_mat.detach().cpu().numpy()))

        self.idl_solver = 'euler'
        self.blur_solver = 'euler'
        self.sharpen_solver = 'euler'
        
        self.idl_beta = 0.3
        self.factor_dim = 128
        # print(r"IDL factor_dim: ",self.factor_dim)
        # print(r"IDL $\beta$: ",self.idl_beta)
        idl_T = 1
        idl_K = 1
        
        blur_T = 1
        blur_K = 1
        
        sharpen_T = 1
        sharpen_K = 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.idl_times = torch.linspace(0, idl_T, idl_K+1).float().to(self.device)
        self.blurring_times = torch.linspace(0, blur_T, blur_K+1).float().to(self.device)
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K+1).float().to(self.device)

    def train(self):
        adj_mat = self.adj_mat
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.01
        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1/d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        del norm_adj, d_mat
        ut, s, self.vt = sparsesvd(self.norm_adj, self.factor_dim)

        linear_Filter = self.norm_adj.T @ self.norm_adj
        self.linear_Filter = convert_sp_mat_to_sp_tensor(linear_Filter).to_dense().to(self.device)

        left_mat = self.d_mat_i @ self.vt.T
        right_mat = self.vt @ self.d_mat_i_inv
        self.left_mat, self.right_mat = torch.FloatTensor(left_mat).to(self.device), torch.FloatTensor(right_mat).to(self.device)
        end = time.time()
    
    def sharpenFunction(self, t, r):
        out = r @ self.linear_Filter
        return -out

    def getUsersRating(self, batch_users):
        batch_test = batch_users.to(self.device)

        with torch.no_grad():
            idl_out = torch.mm(batch_test, self.left_mat @  self.right_mat)

            blurred_out = torch.mm(batch_test, self.linear_Filter)
            # blurred_out = torch.mm(batch_test.to_dense(), self.linear_Filter)
            del batch_test
            
            sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta*idl_out+blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
            # sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
        U_2 =  torch.mean(torch.cat([blurred_out.unsqueeze(0),sharpened_out[1:,...]],axis=0),axis=0)
        
        # U_2 =  blurred_out
        # del blurred_out

        # U_2 = sharpened_out[-1]
        # del sharpened_out

        # ret = U_2
        # del U_2
        ret = self.idl_beta * idl_out + U_2
        
        return ret
def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))    
if __name__ == "__main__":
    dataset = datasets.Tokyo()
    train_matrix, test_positive, val_positive, place_coords = dataset.generate_data(0,50)
    adj_mat = dataset.user_POI_Graph
    lm = BSPM(adj_mat)
    lm.train()

    batch_ratings = convert_sp_mat_to_sp_tensor(adj_mat)
    rating = lm.getUsersRating(batch_ratings)
    rating[rating > 0.0] = 1.0
    rating[rating < 0.0] = 0.0
    for i in range(len(dataset.trainUser)):
        print(rating[dataset.trainUser[i]][dataset.trainItem[i]])