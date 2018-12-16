import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

import crf.lsh

lattice = load(name="lattice",sources=["../crf/lattice/lite/lattice.cpp"])
latticefilter = lattice.filter

class LatticeGaussian(nn.Module):
    def __init__(self,ref):
        super().__init__()
        self.ref = ref

    def __matmul__(self,U):
        return self(U)
    def __rmatmul__(self,U):
        return self(U)

    def forward(self,U):
        return latticefilter(U,self.ref) - U

    def backward(self,*args,**kwargs):
        raise NotImplementedError


class LSHGaussian(nn.Module):
    def __init__(self,ref):
        super().__init__()
        self.ref = ref

    def __matmul__(self,U):
        return self(U)
    def __rmatmul__(self,U):
        return self(U)

    def forward(self,U):
        return lsh.filter(U,self.ref,5,5,30) - U

    def backward(self,*args,**kwargs):
        raise NotImplementedError
