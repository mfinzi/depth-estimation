import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import crf
from torch.autograd import Function
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter, BoxFilter
import time
import numpy as np
import concurrent.futures
import torch.multiprocessing as mp
#import multiprocessing as mp
lattice = load(name="lattice",sources=[os.path.expanduser("~/depth-estimation/crf/lattice/lite/lattice.cpp")])
latticefilter = lattice.filter

def encode_outer(x,y):
    """Encodes the outer product of two images of shape bs x c1 x h x w
       and bs x c2 x h x w into an image of shape bs x (c1*c2) x h x w """
    xy = x[:,:,None,:,:]*y[:,None,:,:,:]
    bs,c1,c2,h,w = xy.shape
    return xy.reshape(bs,c1*c2,h,w)

def decode_outer(xy):
    """"""
    pass

def img_mvm(A,x):
    #print(A.shape)
    #print(x.permute(0,2,3,1).unsqueeze(-1).shape)
    return (A @ x.permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)


# class MemoryEfficientBMM()
def memoryEfficientBMM(A1,A2):
    """Assumes input matrices A1, and A2 are of shape b x n x m and b x m x p
        to produce an output of shape b x n x p, works well when n,m,p << b"""
    b,n,m = A1.shape
    b2,m2,p = A2.shape
    assert b==b2 and m==m2, "incompatible shapes"
    inner = torch.zeros_like(A1)
    output = A1.data.new().resize_(b,n,p)
    
    for i in range(n):
        for j in range(p):
            inner = A1[:,i,:]*A2[:,:,j]
            inner_sum = inner.sum(-1)
            output[:,i,j] = inner_sum#torch.einsum('bnm,bmp->bnp',A1,A2)#(A1[:,:,:,None]*A2[:,None,:,:]).sum(2)
    return output

class mBoxFilter(nn.Module):
    def __init__(self,r):
        super().__init__()
        self.r = r
    def forward(self,x):
        assert x.dim()==4, f"Got shape {x.shape}"
        r = self.r
        summed = F.pad(x,(r+1,r,r+1,r)).cumsum(dim=2).cumsum(dim=3)
        intersection = summed[:,:,:-2*r-1,:-2*r-1]
        union = summed[:,:,2*r+1:,2*r+1:]
        left = summed[:,:,:-2*r-1,2*r+1:]
        right = summed[:,:,2*r+1:,:-2*r-1]
        #print(intersection.shape,union.shape,left.shape,right.shape)
        return union - left - right + intersection

def batchedInv(batchedTensor):
    if batchedTensor.shape[0] >= 256 * 256 - 1:
        temp = []
        #print(batchedTensor.shape)
        for t in torch.split(batchedTensor, 256 * 256 - 1):
            temp.append(torch.inverse(t))
        return torch.cat(temp)
    else:
        return torch.inverse(batchedTensor)

def batchedMM(A,B):
    if A.shape[0] >= 256 * 256 - 1:
        temp = []
        for Ai,Bi in zip(torch.split(A, 256 * 256 - 1),torch.split(B,256*256-1)):
            temp.append(Ai@Bi)
        return torch.cat(temp)
    else:
        return A@B

class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, y, x):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert h_x == h_y and w_x == w_y
        n,h,w = n_x,h_x,w_x
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N

        # cov_yx # shape: {n x h x w x c_y x c_x}
        cov_yx = (self.boxfilter(encode_outer(y,x)) / N - encode_outer(mean_y,mean_x)).permute(0,2,3,1).reshape(n,h,w,c_y,c_x)
        # var_x  # shape: {n x h x w x c_x x c_x}
        cov_xx = (self.boxfilter(encode_outer(x,x)) / N - encode_outer(mean_x,mean_x)).permute(0,2,3,1).reshape(n,h,w,c_x,c_x)

        # I # shape: {c_x x c_x}
        I = torch.eye(c_x).to(x.device)
        # A
        #A = torch.einsum('nhwij,nhwjk->nhwik',(cov_yx, torch.inverse(cov_xx + self.eps*I)))
        #inverse_mat = torch.inverse(cov_xx+self.eps*I)
        #A = cov_yx@inverse_mat
        #A = (cov_yx.reshape(-1,c_y,c_x) @ batchedInv(cov_xx.reshape(-1,c_x,c_x) + self.eps*I)).reshape(n,h,w,c_y,c_x)
        cov_xx_inv = batchedInv(cov_xx.reshape(-1,c_x,c_x) + self.eps*I)
        A = batchedMM(cov_yx.reshape(-1,c_y,c_x),cov_xx_inv).reshape(n,h,w,c_y,c_x)
        #print(torch.cuda.memory_allocated()) 
        A_vec = A.reshape(n,h,w,c_y*c_x).permute(0,3,1,2)
        # b
        b = mean_y - img_mvm(A,mean_x)

        # mean_A; mean_b
        mean_A_vec = self.boxfilter(A_vec) / N
        mean_A = mean_A_vec.permute(0,2,3,1).reshape(n,h,w,c_y,c_x)
        mean_b = self.boxfilter(b) / N

        return img_mvm(mean_A,x) + mean_b


class TaylorFilter(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,src,guide_img):
        exp_img = torch.exp(-(guide_img**2).sum(1))[:,None,:,:]
        weighted_src = exp_img*src
        offset = weighted_src.sum(-1).sum(-1)[:,:,None,None]
        interaction_term = (guide_img[:,:,None,:,:]*weighted_src[:,None,:,:,:]).sum(-1).sum(-1) # Sum over pixels
        interaction_img = (interaction_term[:,:,:,None,None]*guide_img[:,:,None,:,:]).sum(1) # sum over guide dimensions
        return exp_img*(offset + 2*interaction_img)

class TaylorAdjacency(TaylorFilter):
    def __init__(self,guide_img):
        super().__init__()
        self.guide_img = torch.from_numpy(guide_img).float().permute(2,0,1)[None,...]

    def __matmul__(self,U):
        img_U = U.t().reshape((-1,)+self.guide_img.shape[-2:]).float()[None,...]#[None,...]
        filtered_output = self(img_U,self.guide_img)
        return (filtered_output - img_U).data.cpu().squeeze().permute(1,2,0).reshape(U.shape)

class GuidedAdjacency(GuidedFilter):
    def __init__(self,guide_img,r,eps):
        super().__init__(r,eps)
        self.guide_img = guide_img.float().cuda()

    def __matmul__(self,U):
        img_U = U.t().reshape((-1,)+self.guide_img.shape[-2:]).float()[None,...].cuda()#[None,...]
        filtered_output = self(img_U,self.guide_img)
        return (filtered_output*.5*(2*self.r+1)**2 - img_U).data.cpu().squeeze().permute(1,2,0).reshape(U.shape)

class BatchedGuidedAdjacency(GuidedFilter):
    def forward(self,src_imgs,guide_imgs):
        return super().forward(src_imgs,guide_imgs)*.5*(2*self.r+1)**2 - src_imgs
        
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
    
class RbfLaplacianC(LatticeGaussian):
    def __init__(self,ref,normalize='sym'):
        """If normalize is true, will apply symmetric normalization
            returns D - W or I - D^-1/2 W D^-1/2 if normalize is true"""
        super().__init__(ref)
        self.shape = self.ref.shape[:1]*2
        self.normalize = normalize
        #if self.normalize:
        self.D = super().__matmul__(torch.ones(self.shape[:1]+(1,)))
    def __matmul__(self,U):
        if self.normalize=='sym':
            WsqrtDU = super().__matmul__(U/self.D.sqrt())
            return (U - WsqrtDU/self.D.sqrt())
        if self.normalize=='right':
            return U - super().__matmul__(U/self.D)
        else:
            WU = super().__matmul__(U)
            return (self.D*U - WU)


class BatchedAdjacency(nn.Module):
    def __init__(self,num_threads=8):
        super().__init__()
        self.num_threads = num_threads
    def forward(self, src_imgs,guide_imgs):
        bs,L,h,w = src_imgs.shape
        bs,d,h,w = guide_imgs.shape
        flat_srcs = src_imgs.view(bs,L,-1).permute(0,2,1)
        flat_refs = guide_imgs.view(bs,d,-1).permute(0,2,1)
        filtered_imgs = BatchedLatticeFilter.apply(flat_srcs,flat_refs,\
                                self.num_threads).permute(0,2,1).reshape(src_imgs.shape)
        return filtered_imgs - src_imgs
    # def forward(self,Us,refs):
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_threads) as executor:
    #         filtered_mb = torch.stack(list(executor.map(lattice_filter_img,Us,refs)))
    #     return filtered_mb - Us

# def lattice_filter_img(src,ref):
#         c,h,w = src.shape
#         k,h,w = ref.shape
#         flat_src = src.view(c,-1).t()
#         flat_ref = ref.view(k,-1).t()
#         return LatticeFilter.apply(flat_src,flat_ref).t().reshape(src.shape)

# def batched_filter(flat_srcs,flat_refs,num_threads):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
#         filtered_mb = torch.stack(list(executor.map(latticefilter,flat_srcs,flat_refs)))
#     return filtered_mb

def batched_filter(flat_srcs,flat_refs,num_threads):
    print(f"numthreads: {num_threads}")
    process_pool = mp.Pool(processes=num_threads)
    filtered_srcs = process_pool.starmap(latticefilter,list(zip(flat_srcs,flat_refs)))
    process_pool.close()
    process_pool.join()
    filtered_mb = torch.stack(filtered_srcs)
    return filtered_mb

class BatchedLatticeFilter(Function):
    @staticmethod
    def forward(ctx,flat_srcs,flat_refs,num_threads):
        assert flat_srcs.shape[:2] == flat_refs.shape[:2], \
            "Incompatible shapes {}, and {}".format(flat_srcs.shape,flat_refs.shape)
        ctx.save_for_backward(flat_srcs,flat_refs)
        ctx.num_threads = num_threads
        filtered_mb = batched_filter(flat_srcs,flat_refs,num_threads)
        return filtered_mb
    @staticmethod
    def backward(ctx,grad_output):
        with torch.no_grad():
            srcs, refs = ctx.saved_tensors
            num_threads = ctx.num_threads
            g = grad_output
            n,L = srcs.shape[-2:]
            d = refs.shape[-1]
            bs = srcs.shape[0]
            grad_source = grad_reference = None
            if ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
                grad_source = batched_filter(g,refs,num_threads) # Matrix is symmetric
            if ctx.needs_input_grad[1]: # try torch.no_grad ()
                s = []
                s.append(time.time())
                gf = grad_and_ref = grad_output[...,None]*refs[...,None,:] # bs x n x L x d
                grad_and_ref_flat = grad_and_ref.view(grad_and_ref.shape[:-2]+(L*d,))
                sf = src_and_ref = srcs[...,None]*refs[...,None,:] # bs x n x L x d
                src_and_ref_flat = src_and_ref.view(src_and_ref.shape[:-2]+(L*d,))
                s.append(time.time())
                #bs x n x (L+Ld+L+Ld):   bs x n x L       bs x n x Ld     bs x n x L   bs x n x Ld 
                all_ = torch.cat([g,grad_and_ref_flat,srcs,src_and_ref_flat],dim=-1)
                s.append(time.time())
                filtered_all = batched_filter(all_,refs,num_threads)#ctx.W@all_#torch.randn_like(all_)#
                s.append(time.time())
                [wg,wgf,ws,wsf] = torch.split(filtered_all,[L,L*d,L,L*d],dim=-1)
                s.append(time.time())
                # has shape bs x n x d 
                grad_reference = -2*(sf*wg[...,None] - srcs[...,None]*wgf.view(bs,n,L,d) + gf*ws[...,None] - g[...,None]*wsf.view(bs,n,L,d)).sum(-2) # sum over L dimension
                if ctx.needs_input_grad[0]: grad_source = wg
                s.append(time.time())
                s = np.array(s)
            print(f"{s[1:]-s[:-1]}")
        return grad_source, grad_reference, None # num_threads needs no grad

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
    