import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as ds
import numpy as np
from oil.utils.utils import Named
import os
import glob
from crf.utils import read_image,read_pfm
import crf.depth
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from crf.features import Vgg16features
import skimage
import scipy.ndimage as simg

def vis(img,unary_depth,target_depth,vmax=255):
    plt.rcParams.update({'font.size': 22})
    f, axarr = plt.subplots(1,3,figsize=(15,10))
    a0 = axarr[0].imshow(img)
    a1 = axarr[1].imshow(unary_depth,vmin=0,vmax=vmax)
    a2 = axarr[2].imshow(target_depth[...,0],vmin=0,vmax=vmax)
    axarr[0].set_title("Img")
    axarr[1].set_title("Unary")
    axarr[2].set_title("Ground Truth")
    plt.show()

def replace_inf(x,val=-1):
    x[~np.isfinite(x)]=val
    return x

def replace_0(x,val=-1):
    x[x==0] = val
    return x

class MBStereo14(Dataset,metaclass=Named):
    def __init__(self,root='~/datasets/mbstereo2014/',downsize=2):
        self.data_path = os.path.expanduser(root)
        left_img_pths = glob.glob(self.data_path+'*/im0.png')
        right_img_pths = glob.glob(self.data_path+'*/im1.png')
        left_disp_pths = glob.glob(self.data_path+'*/disp0.pfm')
        self.downsize=downsize
        ds = lambda img: simg.gaussian_filter(simg.zoom(img,(1/downsize,1/downsize,1),order=2),1.4)
        self.left_imgs = [ds(read_image(pth)) for pth in left_img_pths]
        self.right_imgs = [ds(read_image(pth)) for pth in right_img_pths]
        self.left_target = [ds(replace_inf(read_pfm(pth))[...,None])/downsize for pth in left_disp_pths]

    def __getitem__(self,index):
        return ((self.left_imgs[index],self.right_imgs[index]),self.left_target[index])
    def __len__(self):
        return len(self.left_imgs)

def planar_sweep_algorithm(ws=1,criterion=crf.depth.AD):
    def get_disp(img_left,img_right):
        return -1*crf.depth.disparity_badness(img_left,img_right,ws,criterion=criterion)
    return get_disp
        
class MBStereo14Unary(MBStereo14):
    def __init__(self,root='~/datasets/mbstereo2014/',downsize=2,unary_algorithm=planar_sweep_algorithm()):
        cache_file = os.path.expanduser(root)+'cachelist.pkl'
        if not os.path.exists(cache_file):
            cached_args_table = dict()
            torch.save(cached_args_table,cache_file)
        cached_args_table = torch.load(cache_file)
        if downsize in cached_args_table:
            print("Using cached dataset")
            with open(cached_args_table[downsize],'rb') as f:
                self.__dict__ = pickle.load(f)
        else:
            print("No cached logits found, computing dataset")
            super().__init__(root=root,downsize=downsize)
            self.unary_logits = [unary_algorithm(left,right) for left,right in zip(self.left_imgs,self.right_imgs)]
            VGG = Vgg16features().cuda().eval()
            self.nn_features = [VGG.get_features(img,k=1) for img in self.left_imgs]
            fname = os.path.expanduser(root)+'cached_{}.pkl'.format(downsize)
            with open(fname,'wb') as f:
                pickle.dump(self.__dict__,f)
            cached_args_table[downsize] = fname
            torch.save(cached_args_table,cache_file)

    def __getitem__(self,index):
        left_logits = torch.from_numpy(self.unary_logits[index]).float().permute((2,0,1))
        left_img = torch.from_numpy(self.left_imgs[index]).float().permute((2,0,1))
        left_features = torch.from_numpy(np.concatenate(self.nn_features[index],axis=-1)).float().permute((2,0,1))
        left_target = torch.from_numpy(self.left_target[index]).float().permute((2,0,1))
        return ((left_logits,left_img,left_features),left_target)

    def show(self):
        for img, unary, target in zip(self.left_imgs,self.unary_logits,self.left_target):
            #L = unary.shape[-1]
            print(img.shape,unary.shape,target.shape)
            #labels = np.arange(L)[None,None,:] #softmax(unary,axis=-1)*labels).sum(-1)
            unary_depth =np.argmax(unary,axis=-1)
            vis(img,unary_depth,target,512/self.downsize)

def softmax(x,axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)# - np.max(x,axis=axis,keepdims=True))
    return e_x / e_x.sum(axis=axis,keepdims=True)

class StereoUpsampling05(Dataset,metaclass=Named):
    def __init__(self,root='~/datasets/mbstereo2005/',downsize=16,val=False,use_vgg=True):
        self.data_path = os.path.expanduser(root)
        self.downsize=downsize
        if val: self.img_names = ['Art','Books','Moebius']
        else: self.img_names = ['Laundry','Dolls','Reindeer']

        self.imgs_highres = [read_image(self.data_path+img_name+'/view1.png') for img_name in self.img_names]
        self.disps_highres = [read_image(self.data_path+img_name+'/disp1.png')[...,0][...,None] for img_name in self.img_names]

        #downsample =  lambda img: simg.gaussian_filter(simg.zoom(img,(1/downsize,1/downsize,1),order=2),1)
        downsample = lambda img: skimage.transform.pyramid_reduce(img,downscale=downsize,multichannel=True)
        self.disps_lowres = [downsample(disp) for disp in self.disps_highres]

        self.use_vgg = use_vgg
        if use_vgg:
            self.vgg_features = self.get_vgg_features(downsize,val)

    @staticmethod
    def to_tensor(img_numpy):
        return torch.from_numpy(img_numpy).float().permute((2,0,1)).cuda()
    def __getitem__(self,index):
        x = (self.to_tensor(self.disps_lowres[index]),self.to_tensor(self.imgs_highres[index]))
        if self.use_vgg:
            x += (self.vgg_features[index],)
        y = self.to_tensor(self.disps_highres[index])
        return x,y
    def __len__(self):
        return len(self.imgs_highres)

    def get_vgg_features(self,downsize,val):
        cache_file = self.data_path+'cachelist.pkl'
        if not os.path.exists(cache_file):
            cached_args_table = dict()
            torch.save(cached_args_table,cache_file)
        cached_args_table = torch.load(cache_file)
        if False:#(downsize,val) in cached_args_table:
            print("Using cached dataset")
            with open(cached_args_table[(downsize,val)],'rb') as f:
                vgg_features = pickle.load(f)
        else:
            print("No cached logits found, computing dataset")
            VGG = Vgg16features().cuda().eval()
            vgg_features = [VGG.get_torch_features(self.to_tensor(img)[None],k=0)[0] for img in self.imgs_highres]
           #print(len(self.vgg_features))
            fname = os.path.expanduser(self.data_path)+'cached_{}.pkl'.format(downsize)
            with open(fname,'wb') as f:
                pickle.dump(vgg_features,f)
            cached_args_table[(downsize,val)] = fname
            torch.save(cached_args_table,cache_file)
        return vgg_features