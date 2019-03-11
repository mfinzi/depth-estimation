import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from crf.gaussian_matrix import BatchedAdjacency, BatchedGuidedAdjacency


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

def charbonneir2(a,b,gamma=3):
    return torch.sqrt(1 + ((a-b)/gamma)**2) - 1

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

def potts(num_classes):
    layer = nn.Conv2d(num_classes,num_classes,kernel_size=1,bias=False)
    tensor = layer.weight
    L,LL,a,b = tensor.shape
    assert L == LL #(square matrix)
    assert a==b==1
    with torch.no_grad():
        tensor.fill_(1)
        tensor.sub_(torch.eye(L)[...,None,None])
    return layer

class charb(nn.Module):
    def __init__(self,gamma):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.s = nn.Parameter(torch.tensor(0.))
    def forward(self,x):
        L = x.shape[1]
        labels = torch.arange(L).float().cuda()
        Mu = charbonneir(labels[None,:],labels[:,None],self.gamma*L)
        return F.conv2d(x,Mu[...,None,None])*torch.exp(self.s)

class CRFasRNN(nn.Module):
    def __init__(self,mu_init,niters=5,r=20,eps=1e-5,notrain_mu=False,gaussian=False,gchannels=1):
        super().__init__()
        self.Mu = mu_init
        if notrain_mu:
            for param in self.Mu.parameters(): param.requires_grad=False
        self.niters= niters 
        # The adjacency matrix (also takes in reference image as argument)
        # softplus parametrization of the epsilon parameter
        #self.gamma = nn.Parameter(torch.tensor(1.))
        self.W = BatchedGuidedAdjacency(gchannels,r,eps,gaussian=gaussian)
    
    def forward(self,refs,logits,confidence=None):
        """Assuming E0 and Refs are shape BxLxHxW and BxCxHxW"""
        if confidence is None:
            confidence = 1
        E0 = -logits*confidence
        Q = F.softmax(-E0, dim=1)
        for i in range(self.niters):
            E = E0 + self.W(self.Mu(Q),refs)
            Q = F.softmax(-E, dim=1)
        out_logits = -E
        return out_logits

class ijrgbGuide(nn.Module):
    def __init__(self,s_ij=.1,s_rgb=.1,trainable=True):
        super().__init__()
        self.s_ij = nn.Parameter(torch.tensor(s_ij)) if trainable else s_ij
        self.s_rgb = nn.Parameter(torch.tensor(s_rgb)) if trainable else s_rgb
    def forward(self,x):
        bs,c,h,w = x.shape
        ij = torch.from_numpy(np.tile(np.mgrid[:h,:w]/np.sqrt(h**2+w**2),(bs,1,1,1))).float().to(x.device)
        return torch.cat([ij/self.s_ij,x/self.s_rgb],dim=1)

class ijGuide(nn.Module):
    def __init__(self,s_ij=.1,trainable=True):
        super().__init__()
        self.s_ij = nn.Parameter(torch.tensor(s_ij)) if trainable else s_ij
    def forward(self,x):
        bs,c,h,w = x.shape
        ij = torch.from_numpy(np.tile(np.mgrid[:h,:w]/np.sqrt(h**2+w**2),(bs,1,1,1))).float().to(x.device)
        return ij/self.s_ij
