import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import crf.lsh
from torch.autograd import Function


lattice = load(name="lattice",sources=[os.path.expanduser("~/depth-estimation/crf/lattice/lite/lattice.cpp")])
latticefilter = lattice.filter

class LatticeGaussian(nn.Module):
    def __init__(self,ref):
        super().__init__()
        self.ref = ref

    def __matmul__(self,U):
        return self(U)
    # def __rmatmul__(self,U):
    #     return self(U)

    def forward(self,U):
        return latticefilter(U,self.ref) - U

class LatticeFilter(Function):
    @staticmethod
    def forward(ctx, source, reference):
        # Typical runtime of O(nd^2 + n*L), Worst case O(nd^2 + n*L*d)
        # assert source and reference are (b x) n x L and n x d respectively
        ctx.save_for_backward(source,reference) # TODO: add batch compatibility
        return latticefilter(source,reference)
    @staticmethod
    def backward(ctx,grad_output):
        # Typical runtime of O(nd^2 + 2L*n*d), Worst case  O(nd^2 + 2L*n*d^2)
        # Does not support second order autograd at the moment
        src, ref = ctx.saved_tensors
        g = grad_output
        n,L = src.shape[-2:]
        d = ref.shape[-1]
        grad_source = grad_reference = None
        if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            grad_source = latticefilter(g,ref) # Matrix is symmetric
        if ctx.needs_input_grad[1]:
            gf = grad_and_ref = grad_output[...,None]*ref[...,None,:] # n x L x d
            grad_and_ref_flat = grad_and_ref.view(grad_and_ref.shape[:-2]+(L*d,))
            sf = src_and_ref = src[...,None]*ref[...,None,:] # n x L x d
            src_and_ref_flat = src_and_ref.view(src_and_ref.shape[:-2]+(L*d,))
            #n x (L+Ld+L+Ld):   n x L       n x Ld     n x L   n x Ld 
            all_ = torch.cat([grad_source,grad_and_ref_flat,src,src_and_ref_flat],dim=-1)
            filtered_all = latticefilter(all_,ref)
            [wg,wgf,ws,wsf] = torch.split(filtered_all,[L,L*d,L,L*d],dim=-1)
            # has shape n x d
            grad_reference = -2*(sf*wg[...,None] - src*wgf + gf*ws[...,None] - g*wsf).sum(-2) # sum over L dimension
            grad_source = wg
        return grad_source, grad_reference
        






    @staticmethod
    def backward(ctx,grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias




class LSHGaussian(nn.Module):
    def __init__(self,ref):
        super().__init__()
        self.ref = ref

    def __matmul__(self,U):
        return self(U)
    # def __rmatmul__(self,U):
    #     return self(U)

    def forward(self,U):
        return crf.lsh.filter(U,self.ref,5,5,30) - U

    def backward(self,*args,**kwargs):
        raise NotImplementedError
