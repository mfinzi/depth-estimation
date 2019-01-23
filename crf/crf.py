import torch
import torch.nn.functional as F
import numpy as np

def gaussian_weights(f):
    """[f] n x c flattened reference 'image' to filter by"""
    I = torch.eye(f.shape[0]).to(f.device)
    W = torch.exp(-((f[None,:,:] - f[:,None,:])**2).sum(-1)/2)-I
    D = W@torch.ones(f.shape[0]).to(f.device)
    D_invsqrt = torch.diag(1/torch.sqrt(D))
    W_normalized = D_invsqrt@(W-I)@D_invsqrt
    return W_normalized

def gaussian_weights_u(f):
    I = torch.eye(f.shape[0]).to(f.device)
    W = torch.exp(-((f[None,:,:] - f[:,None,:])**2).sum(-1)/2)-I
    return W

def lazy_W(f):
    def W(i,j):
        square_dist = ((f-f[i,j])**2).sum(-1).reshape(-1)
        a = np.exp(-square_dist/2)
        d = a@np.ones(a.shape[0])-1
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