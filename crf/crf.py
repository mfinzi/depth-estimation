import torch
import torch.nn.functional as F


def gaussian_weights(f):
    """[f] n x c flattened reference 'image' to filter by"""
    return torch.exp(-((f[None,:,:] - f[:,None,:])**2).sum(-1)/2) - torch.eye(f.shape[0]).to(f.device)

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