import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import crf
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
        assert source.shape[0] == reference.shape[0], \
            "Incompatible shapes {}, and {}".format(source.shape,reference.shape)
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
            all_ = torch.cat([g,grad_and_ref_flat,src,src_and_ref_flat],dim=-1)
            filtered_all = latticefilter(all_,ref)
            [wg,wgf,ws,wsf] = torch.split(filtered_all,[L,L*d,L,L*d],dim=-1)
            # has shape n x d # Should be -2 here, there is a bug still
            grad_reference = -1*(sf*wg[...,None] - src[...,None]*wgf.view(-1,L,d) + gf*ws[...,None] - g[...,None]*wsf.view(-1,L,d)).sum(-2) # sum over L dimension
            if ctx.needs_input_grad[0]: grad_source = wg
        return grad_source, grad_reference
        



# class LSHGaussian(nn.Module):
#     def __init__(self,ref):
#         super().__init__()
#         self.ref = ref

#     def __matmul__(self,U):
#         return self(U)
#     # def __rmatmul__(self,U):
#     #     return self(U)

#     def forward(self,U):
#         return crf.lsh.filter(U,self.ref,5,5,30) - U

#     def backward(self,*args,**kwargs):
#         raise NotImplementedError


if __name__=="__main__":
    from torch.autograd import gradcheck
    ref = torch.rand(15,3,dtype=torch.float,requires_grad=True)
    src = torch.rand(15,2,dtype=torch.float,requires_grad=False) # Because of single precision
    test = gradcheck(LatticeFilter.apply,(src,ref),eps=1e-3,rtol=5e-2,atol=1e-2)
    print(test) # Gradients are perhaps wrong still (need to implement double precision method)