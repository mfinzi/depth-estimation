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
import math
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
    if batchedTensor.shape[0] >= 128 * 128 - 1:
        temp = []
        #print(batchedTensor.shape)
        for t in torch.split(batchedTensor, 128 * 128 - 1):
            temp.append(torch.inverse(t))
        return torch.cat(temp)
    else:
        return torch.inverse(batchedTensor)

def batchedMM(A,B):
    if A.shape[0] >= 128 * 128 - 1:
        temp = []
        for Ai,Bi in zip(torch.split(A, 128 * 128 - 1),torch.split(B,128*128-1)):
            temp.append(Ai@Bi)
        return torch.cat(temp)
    else:
        return A@B

def box_filter(array,r,dim):
    m = len(array.shape)
    padding = ((0,0,)*dim + (r,r,)+(0,0,)*(m-dim-1))
    padded_arr = F.pad(array,padding[::-1]) #zero padding
    # try:
    #     padded_arr = F.pad(array,padding[::-1],mode='replicate') #bug with replicate padding on nd
    # except (AssertionError,NotImplementedError):
    #     padded_arr = F.pad(array[None,None],padding[::-1],mode='replicate')[0,0]
    cumsum = padded_arr.cumsum(dim=dim)
    
    upper_slice = (slice(None),)*dim + (slice(2*r,None,None),)+(slice(None),)*(m-dim-1)
    lower_slice = (slice(None),)*dim + (slice(None,-2*r,None),)+(slice(None),)*(m-dim-1)
    h = array.shape[dim]

    i = np.arange(h)
    reshape_slice = (None,)*dim + (slice(None),)+(None,)*(m-dim-1)
    #print(np.minimum(i,r) + np.minimum(h-i-1,r)+1)
    counts = torch.from_numpy(np.minimum(i,r) + np.minimum(h-i-1,r)+1)[reshape_slice].type(array.dtype).to(array.device)
    #print(counts)
    return (cumsum[upper_slice] - cumsum[lower_slice])/counts

def gaussian_blur(array,sigma,dim):
    return GaussianBlur.apply(array,sigma,dim)

class GaussianBlur(Function):
    @staticmethod
    def forward(ctx,array, sigma, dim,niters=3):
        #torch.cuda.empty_cache()
        with torch.no_grad():
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                ctx.save_for_backward(array,sigma)
                ctx.dim=dim
            r= int(np.floor(np.sqrt(12*sigma.cpu()**2/niters+1))//2) # Box width
            for i in range(niters):
                array = box_filter(array,r,dim)
            return array

    @staticmethod
    def backward(ctx,grad_output):
        with torch.no_grad():
            s = []
            s.append(time.time())
            array,sigma = ctx.saved_tensors
            v = array
            dim = ctx.dim
            g = grad_output
            grad_v = grad_sigma = None
            h = v.shape[dim]
            # shape 1,1,1,h,1,1,1
            f = (torch.arange(h)/sigma).reshape((1,)*dim+(h,)+(1,)*(len(v.shape)-dim-1)).type(array.dtype).to(array.device)
            s.append(time.time())
            if ctx.needs_input_grad[1]:
                gf = g*f
                vf = v*f
                s.append(time.time())
                inner_terms = torch.stack([g,-gf,v,-vf],dim=-1)
                s.append(time.time())
                #print(inner_terms.shape)
                filtered_inner = GaussianBlur.apply(inner_terms,sigma,dim)#ctx.W@all_#torch.randn_like(all_)#
                outer_terms = torch.stack([vf,v,gf,g],dim=-1)
                s.append(time.time())
                grad_f = -(outer_terms*filtered_inner).sum(-1)
                grad_v = filtered_inner[...,0]
                s.append(time.time())
                grad_sigma = -1*(grad_f*f).sum()/sigma -(grad_v*v).sum()/sigma
                s.append(time.time())
                s = np.array(s)
                #print(f"{s[1:]-s[:-1]}")
            elif ctx.needs_input_grad[0]:
                grad_v = GaussianBlur.apply(g,sigma,dim)
        return grad_v, grad_sigma, None # no gradient to dim




class GuidedFilter(nn.Module):
    def __init__(self,channels=1, r=20, eps=1e-8,gaussian=False):
        super(GuidedFilter, self).__init__()

        
        self.omega = nn.Parameter(torch.log(torch.exp(torch.tensor(eps))-1).expand(channels))
        self.splus = nn.Softplus()
        if gaussian:
            self.omega2 = nn.Parameter(torch.log(torch.tensor(r).float()))#torch.log(torch.tensor(r).float())#
            self.boxfilter = lambda arr: gaussian_blur(gaussian_blur(arr,self.r(),dim=2),self.r(),dim=3)
        else:
            self._r = r
            self.boxfilter = BoxFilter(r)
        self.gaussian = gaussian
    #@property
    def r(self):
        if self.gaussian:
            return torch.exp(self.omega2)
        else:
            return self._r
    @property
    def eps(self):
        return self.splus(self.omega)
    def get_coeffs(self,y,x):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert h_x == h_y and w_x == w_y
        n,h,w = n_x,h_x,w_x
        assert h_x > 2 * self.r() + 1 and w_x > 2 * self.r() + 1

        # N
        N = self.boxfilter(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N

        # cov_yx # shape: {n x h x w x c_y x c_x}
        cov_yx = (self.boxfilter(encode_outer(y,x)) / N - encode_outer(mean_y,mean_x)).permute(0,2,3,1).reshape(n,h,w,c_y,c_x)
        # var_x  # shape: {n x h x w x c_x x c_x}
        #cov_xx = (self.boxfilter(encode_outer(x,x)) / N - encode_outer(mean_x,mean_x)).permute(0,2,3,1).reshape(n,h,w,c_x,c_x)

        # I # shape: {c_x x c_x}
        I = torch.eye(c_x).to(x.device)
        # A
        #A = torch.einsum('nhwij,nhwjk->nhwik',(cov_yx, torch.inverse(cov_xx + self.eps*I)))
        #inverse_mat = torch.inverse(cov_xx+self.eps*I)
        #A = cov_yx@inverse_mat
        #A = (cov_yx.reshape(-1,c_y,c_x) @ batchedInv(cov_xx.reshape(-1,c_x,c_x) + self.eps*I)).reshape(n,h,w,c_y,c_x)

        #->#cov_xx_inv = batchedInv(cov_xx.reshape(-1,c_x,c_x) + self.eps*I)
        cov_xx_lite = (self.boxfilter(x*x) / N -mean_x*mean_x).permute(0,2,3,1).reshape(-1,c_x)
        A_lite = cov_yx.reshape(-1,c_y,c_x)/(cov_xx_lite[...,None,:] + self.eps)
        A = A_lite.reshape(n,h,w,c_y,c_x)
        #->#A = batchedMM(cov_yx.reshape(-1,c_y,c_x),cov_xx_inv).reshape(n,h,w,c_y,c_x)
        #print(torch.cuda.memory_allocated()) 
        A_vec = A.reshape(n,h,w,c_y*c_x).permute(0,3,1,2)
        # b
        b = mean_y - img_mvm(A,mean_x)

        # mean_A; mean_b
        mean_A_vec = self.boxfilter(A_vec) / N
        mean_A = mean_A_vec.permute(0,2,3,1).reshape(n,h,w,c_y,c_x)
        mean_b = self.boxfilter(b) / N
        return mean_A,mean_b

    def forward(self, y, x):
        mean_A, mean_b = self.get_coeffs(y,x)
        return img_mvm(mean_A,x) + mean_b

class FastGuidedFilter(GuidedFilter):
    def __init__(self,*args,subsample_ratio=2,mode='nearest',**kwargs):
        super().__init__(*args,**kwargs)
        self.subsample_ratio=subsample_ratio
        self.boxfilter = BoxFilter(self._r//subsample_ratio)
        self.mode = mode

    def forward(self,y,x):
        s = self.subsample_ratio
        n, c_x, h, w = x.size()
        n, c_y, h, w = y.size()
        y_lowres = F.interpolate(y,size=(h//s,w//s),mode=self.mode)
        x_lowres = F.interpolate(x,size=(h//s,w//s),mode=self.mode)
        mean_A_lowres,mean_b_lowres = self.get_coeffs(y_lowres,x_lowres)
        mean_A_lowres_vec = mean_A_lowres.reshape(n,h//s,w//s,c_y*c_x).permute(0,3,1,2)
        mean_A_ = F.interpolate(mean_A_lowres_vec,size=(h,w),mode=self.mode).permute(0,2,3,1)
        #print(f'A {mean_A_.shape}, x {x.shape}, y {y.shape}')
        mean_A = mean_A_.reshape(n,h,w,c_y,c_x)
        mean_b = F.interpolate(mean_b_lowres,size=(h,w),mode=self.mode)
        return img_mvm(mean_A,x) + mean_b
# class TaylorFilter(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self,src,guide_img):
#         exp_img = torch.exp(-(guide_img**2).sum(1))[:,None,:,:]
#         weighted_src = exp_img*src
#         offset = weighted_src.sum(-1).sum(-1)[:,:,None,None]
#         interaction_term = (guide_img[:,:,None,:,:]*weighted_src[:,None,:,:,:]).sum(-1).sum(-1) # Sum over pixels
#         interaction_img = (interaction_term[:,:,:,None,None]*guide_img[:,:,None,:,:]).sum(1) # sum over guide dimensions
#         return exp_img*(offset + 2*interaction_img)

# class TaylorAdjacency(TaylorFilter):
#     def __init__(self,guide_img):
#         super().__init__()
#         self.guide_img = torch.from_numpy(guide_img).float().permute(2,0,1)[None,...]

#     def __matmul__(self,U):
#         img_U = U.t().reshape((-1,)+self.guide_img.shape[-2:]).float()[None,...]#[None,...]
#         filtered_output = self(img_U,self.guide_img)
#         return (filtered_output - img_U).data.cpu().squeeze().permute(1,2,0).reshape(U.shape)

# class GuidedAdjacency(GuidedFilter):
#     def __init__(self,guide_img,r,eps):
#         super().__init__(r,eps)
#         self.guide_img = guide_img.float().cuda()

#     def __matmul__(self,U):
#         img_U = U.t().reshape((-1,)+self.guide_img.shape[-2:]).float()[None,...].cuda()#[None,...]
#         filtered_output = self(img_U,self.guide_img)
#         return (filtered_output*.5*(2*self.r+1)**2 - img_U).data.cpu().squeeze().permute(1,2,0).reshape(U.shape)

class BatchedGuidedAdjacency(FastGuidedFilter):
    def forward(self,src_imgs,guide_imgs):#*.5*(2*self.r()+1)**2#(.5*self.r()*math.sqrt(2*math.pi))
        return super().forward(src_imgs,guide_imgs)*.5*(2*self.r()+1)**2 - src_imgs

#class MattingLaplacian(GuidedFilter):


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
    