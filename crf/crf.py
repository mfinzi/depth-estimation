import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .gaussian_matrix import BatchedAdjacency


def gaussian_weights(f):
    """[f] n x c flattened reference 'image' to filter by"""
    I = torch.eye(f.shape[0]).to(f.device)
    W = torch.exp(-((f[None,:,:] - f[:,None,:])**2).sum(-1))-I
    D = W@torch.ones(f.shape[0]).to(f.device)
    D_invsqrt = torch.diag(1/torch.sqrt(D))
    W_normalized = D_invsqrt@W@D_invsqrt - I
    return W_normalized

def gaussian_weights_u(f):
    I = torch.eye(f.shape[0]).to(f.device)
    W = torch.exp(-((f[None,:,:] - f[:,None,:])**2).sum(-1))-I
    return W

def lazy_W(f):
    def W(i,j):
        square_dist = ((f-f[i,j])**2).sum(-1).reshape(-1)
        a = np.exp(-square_dist)
        d = a@np.ones(a.shape[0])-1 # for self include
        a /= np.sqrt(d)
        weights = a.reshape(f.shape[:2])
        return weights
    return W

def charbonneir(a,b,gamma=.1):
    return torch.sqrt(gamma**2 + (a-b)**2) - gamma

def compatibility_matrix(compat,labels):
    return compat(labels[:,None],labels[None,:])

def mean_field_infer(E_0,W,Mu,niters=10):
    """ Performs niters iterations of mean field inference for CRF,
        Inputs:
            [E_0]   n x L    Unary Potentials
            [W]     n x n    Binary Weights
            [Mu]    L x L    Compatibility matrix 
        Outputs: 
            [Q]     n x L:   output label probabilities"""
    Q = F.softmax(-E_0, dim=1)
    for i in range(niters):
        E = E_0 + W@Q@Mu
        Q = F.softmax(-E, dim=1)
    return Q

def potts_init(linear_layer):
    tensor = linear_layer.weight
    L,LL = tensor.shape
    assert L == LL #(square matrix)
    with torch.no_grad():
        tensor.fill_(1)
        tensor.sub_(torch.eye(L))
    
class CRFasRNN(nn.Module):
    def __init__(self,num_classes,niters=5,num_threads=8):
        super().__init__()
        self.Mu = nn.Linear(L,L,bias=False) # The compatibility matrix
        potts_init(self.Mu)
        self.niters= niters 
        # The adjacency matrix (also takes in reference image as argument)
        self.W = BatchedAdjacency(num_threads=num_threads) 
    
    def forward(self,E0,Refs):
        """Assuming E0 and Refs are shape BxLxHxW and BxCxHxW"""
        Q = F.softmax(-E0, dim=1)
        for i in range(niters):
            E = E0 + self.W(Mu(Q),Refs)
            Q = F.softmax(-E, dim=1)
        return Q

