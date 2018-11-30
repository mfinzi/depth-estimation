import numpy as np
import scipy as sp
import scipy.ndimage
from sklearn.feature_extraction.image import extract_patches_2d
from .utils import crop_patch, compute_histogram, read_image
from scipy.ndimage.filters import convolve as conv
import cv2
from cv2.ximgproc import guidedFilter

def centroids(masks):
    imax, jmax = masks.shape[1:]
    i,j = np.mgrid[0:imax,0:jmax]
    return np.array([(i*masks).mean((1,2)), (j*masks).mean((1,2))])

def rbf(center,sigma):
    sigma = np.array(sigma)[None,None,...]
    center = np.array(center)[None,None,...]
    ic, jc = center[:,:,0,...], center[:,:,1,...]
    return lambda i,j: np.exp(-((i[...,None]-ic)**2 + (j[...,None]-jc)**2)/(2*sigma**2)).squeeze()

def laplacian(img):
    lap_2d = np.array([[ 0,-1, 0],[-1, 4,-1],[ 0,-1, 0]])
    out = sp.ndimage.filters.convolve(img,lap_2d, mode='constant')
    return out #sp.signal.convolve2d

def convolve_op(filter, img_shape):
    n = img_shape[0]*img_shape[1]
    def mvm(v):
        img = v.reshape(img_shape)
        out_img = sp.ndimage.filters.convolve(img, filter, mode='constant')
        return out_img.reshape(n)
    return sp.sparse.linalg.LinearOperator((n,n),mvm)

def laplacian_op(img_shape):
    lap_2d = np.array([[ 0,-1, 0],[-1, 4,-1],[ 0,-1, 0]])
    return convolve_op(lap_2d,img_shape)

def guided_op(guide_img,radius,eps):
    n = guide_img.shape[0]*guide_img.shape[1]
    def mvm(v):
        img = v.reshape(guide_img.shape)
        out_img = guidedFilter(guide_img,np.uint8(img*255),radius,eps)
        return out_img.reshape(-1)/255.
    return sp.sparse.linalg.LinearOperator((n,n),mvm)

def identity_op(img_shape):
    n = img_shape[0]*img_shape[1]
    return sp.sparse.linalg.LinearOperator((n,n),lambda v:v)
    #sp.sparse.linalg.cg(A,y)[0]

def diag_op(img):
    n = img.shape[0]*img.shape[1]
    img_vec = img.reshape(-1)
    return sp.sparse.linalg.LinearOperator((n,n),lambda v:img_vec*v)
#def extract_patches(img1,patch_size):

def normalized(img,window_shape=None):
    if window_shape is None: 
        mfunc = lambda img: img.mean(axis=(0,1))
    else:
        box = np.ones(window_shape)/(window_shape[0]*window_shape[1])
        if len(img.shape) == 3:
            box = box[...,None]
        mfunc = lambda img: conv(img,box)

    diff = img-mfunc(img)
    std = np.sqrt(mfunc(diff**2))
    normalized_img = diff/(std+1e-6)
    return normalized_img

def SD(imga,imgb):
    return (imga-imgb)**2
def AD(imga,imgb):
    return np.abs(imga-imgb)
def nprod(imga,imgb):
    return -1*imga*imgb

def disparity_measurements(img1,img2,window_size=9):
    ws = window_size
    max_disp = int(img1.shape[1]/4)
    nimg1 = normalized(img1,ws)
    nimg2 = normalized(img2,ws)

    h,w,c = nimg1.shape
    #padded_im1 = np.pad(nimg1,((0,0),(max_disp,0),(0,0)), mode='constant')
    padded_im2 = np.pad(nimg2,((0,0),(max_disp,0),(0,0)), mode='constant')

    out = np.zeros((h,w,max_disp))
    for i in np.arange(max_disp):#1+np.arange(-w//2,w//2):
        shifted_nimg2 = padded_im2[:,max_disp-i:w+max_disp-i]
        out[:,:,i] = AD(nimg1,shifted_nimg2).sum(2)
    box = np.ones((ws,ws,1))
    aggregated = sp.ndimage.filters.convolve(out,box)
    disps = np.argmin(aggregated,axis=-1)
    return disps

def NCC_disp(img,template):
    nimg = img#normalized(img)
    ntemplate = template#normalized(template)
    ncorr = sp.signal.convolve(nimg,ntemplate,mode='valid')
    reduced = np.linalg.norm(ncorr,axis=-1)
    i,j = np.where(reduced==np.max(reduced))
    i,j = i[0],j[0]
    return np.minimum(j, img.shape[1]-j)


def get_poc_offset(img1,img2):
    I1 = sp.fftpack.fft2(img1,axes=(0,1))
    I2 = sp.fftpack.fft2(img2,axes=(0,1))
    normalized = I1.conj()*I2/(np.abs(I1.conj()*I2)+1e-4)
    cross_corr = sp.fftpack.ifft2(normalized).real
    centered = sp.fftpack.fftshift(cross_corr,axes=(0,1))
    reduced = np.linalg.norm(centered,axis=2)
    i,j = np.where(reduced==np.max(reduced))
    i,j = i[0],j[0]
    return np.minimum(j, img2.shape[1]-j)