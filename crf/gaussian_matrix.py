import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import crf
from torch.autograd import Function
from guided_filter_pytorch.guided_filter import GuidedFilter, BoxFilter
import time
import numpy as np
from concurrent.futures

lattice = load(name="lattice",sources=[os.path.expanduser("~/depth-estimation/crf/lattice/lite/lattice.cpp")])
latticefilter = lattice.filter

class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return mean_A * x + mean_b

class GuidedAdjacency(GuidedFilter):
    def __init__(self,guide_img,r,eps):
        super().__init__(r,eps)
        self.guide_img = torch.from_numpy(guide_img).float().cuda()

    def __matmul__(self,U):
        img_U = U.t().reshape((-1,)+self.guide_img.shape[-2:]).float().cuda()[None,...]
        return (self(img_U,self.guide_img) - img_U).data.cpu().squeeze().permute(1,2,0).reshape(U.shape)

class LatticeGaussian(nn.Module):
    def __init__(self,ref):
        super().__init__()
        self.ref = ref

    def __matmul__(self,U):
        return self(U)
    # def __rmatmul__(self,U):
    #     return self(U)

    def forward(self,U):
        return LatticeFilter.apply(U,self.ref) - U

class RbfLaplacian(nn.Module):
    def __init__(self,ref,normalize=True):
        """If normalize is true, will apply symmetric normalization
            returns D - W or I - D^-1/2 W D^-1/2 if normalize is true"""
        super().__init__()
        self.ref = ref
        self.shape = self.ref.shape[:1]*2
        self.normalize = normalize
        #if self.normalize:
        self.D = LatticeFilter.apply(torch.ones(self.shape[:1]+(1,)),self.ref)
    def __matmul__(self,U):
        if self.normalize:
            return (U - LatticeFilter.apply(U/self.D.sqrt(),self.ref)/self.D.sqrt())
        else:
            return (self.D*U - LatticeFilter.apply(U,self.ref))
    

class BatchedAdjacency(nn.Module):
    def __init__(self,num_threads=8):
        super().__init__()
        self.num_threads = num_threads
        
    def forward(self,Us,refs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            filtered_mb = torch.stack(list(executor.map(LatticeFilter.apply,Us,refs)))
        return filtered_mb - Us


class LatticeFilter(Function):
    @staticmethod
    def forward(ctx, source, reference):
        #W = torch.exp(-((reference[None,:,:] - reference[:,None,:])**2).sum(-1)).double()
        #ctx.W = W
        # Typical runtime of O(nd^2 + n*L), Worst case O(nd^2 + n*L*d)
        assert source.shape[0] == reference.shape[0], \
            "Incompatible shapes {}, and {}".format(source.shape,reference.shape)
        ctx.save_for_backward(source,reference) # TODO: add batch compatibility
        s0 = time.time()
        filtered_output = latticefilter(source,reference)
        return filtered_output
    @staticmethod
    def backward(ctx,grad_output):
        # Typical runtime of O(nd^2 + 2L*n*d), Worst case  O(nd^2 + 2L*n*d^2)
        # Does not support second order autograd at the moment
        with torch.no_grad():
            src, ref = ctx.saved_tensors
            g = grad_output
            n,L = src.shape[-2:]
            d = ref.shape[-1]
            grad_source = grad_reference = None
            if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
                grad_source = latticefilter(g,ref)#ctx.W@g#latticefilter(g,ref) # Matrix is symmetric
            if ctx.needs_input_grad[1]: # try torch.no_grad ()
                s = []
                s.append(time.time())
                gf = grad_and_ref = grad_output[...,None]*ref[...,None,:] # n x L x d
                grad_and_ref_flat = grad_and_ref.view(grad_and_ref.shape[:-2]+(L*d,))
                sf = src_and_ref = src[...,None]*ref[...,None,:] # n x L x d
                src_and_ref_flat = src_and_ref.view(src_and_ref.shape[:-2]+(L*d,))
                s.append(time.time())
                #n x (L+Ld+L+Ld):   n x L       n x Ld     n x L   n x Ld 
                all_ = torch.cat([g,grad_and_ref_flat,src,src_and_ref_flat],dim=-1)
                s.append(time.time())
                filtered_all = latticefilter(all_,ref)#ctx.W@all_#torch.randn_like(all_)#
                s.append(time.time())
                [wg,wgf,ws,wsf] = torch.split(filtered_all,[L,L*d,L,L*d],dim=-1)
                s.append(time.time())
                # has shape n x d 
                grad_reference = -2*(sf*wg[...,None] - src[...,None]*wgf.view(-1,L,d) + gf*ws[...,None] - g[...,None]*wsf.view(-1,L,d)).sum(-2) # sum over L dimension
                if ctx.needs_input_grad[0]: grad_source = wg
                s.append(time.time())
                s = np.array(s)
            print(f"{s[1:]-s[:-1]}")
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
    import numpy as np
    from PIL import Image
    def read_img(filename):
        img = Image.open(filename).convert('RGB')
        img = np.array(img).astype(float)/255
        return img
    sigma_p = .01
    sigma_c = .125
    img = read_img('./lattice/lite/images/input.bmp')[::64,::64]
    h,w,c = img.shape
    position = np.mgrid[:h,:w].transpose((1,2,0))/np.sqrt(h**2+w**2)
    reference = np.zeros((h,w,5))
    reference[...,:3] = img/sigma_c
    reference[...,3:] = position/sigma_p
    #reference = position/sigma_p
    homo_src = np.ones((h,w,3+1))
    homo_src[...,:c] = img
    ref_arr = torch.tensor(reference.reshape((h*w,-1)).astype(np.float64),requires_grad=True)
    src_arr = torch.tensor(homo_src.reshape((h*w,-1)).astype(np.float64),requires_grad=False)
    #ref_arr.requires_grad=True#False
    #src_arr.requires_grad=True#True
    ref_arr = torch.rand(80,3,dtype=torch.double,requires_grad=True)
    src_arr = torch.rand(80,2,dtype=torch.double,requires_grad=False) # Because of single precision
    #print("AAAA")
    #test = gradcheck(LatticeFilter.apply,(src,ref),eps=1e-3,rtol=5e-2,atol=1e-2)
    test = gradcheck(LatticeFilter.apply,(src_arr,ref_arr),eps=1e-5,rtol=5e-4,atol=1e-5)
    print(test) # Gradients are perhaps wrong still (need to implement double precision method)
    