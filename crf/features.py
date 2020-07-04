import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
from scipy.ndimage import zoom
import torch.nn.functional as F


class Vgg16features(torch.nn.Module):
    def __init__(self):
        super().__init__()
        features = list(models.vgg16(pretrained = True).features)[:23]
        # features: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval() 
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22}:
                results.append(x)
        return results

    def preprocess(self,x):
        """converts numpy images in hxwx3 format with values in [0,1] to images
         with mean 0, std 1 per channel in (1,3,h,w)"""
        #normalized = (x - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        normalized = (x-np.array([0.29298669, 0.26512041, 0.21699697]))/np.array([0.24798678, 0.19988715, 0.18761264])
        batch_input_reshaped = normalized[None,...].transpose((0,3,1,2))
        device = next(self.features.parameters()).device
        img_batch = torch.from_numpy(batch_input_reshaped).float().to(device)
        resized = nn.functional.interpolate(img_batch,size=(224,224),mode='bilinear',align_corners=True)
        return resized

    def rescale_reshape(self,img_torch,img_shape):
        h,w,_ = img_shape
        img_np = img_torch.cpu().data.numpy()[0].transpose((1,2,0))
        hn,wn,_ = img_np.shape
        zoom_factor = h/hn,w/wn,1
        return zoom(img_np,zoom_factor,order=2)

    def get_features(self,x,k=3):
        xpt = self.preprocess(x)
        feature_list = self(xpt)
        np_features = [self.rescale_reshape(feature,x.shape) for feature in feature_list[:k]]
        return np_features

    def get_torch_features(self,x,k=0):
        x = nn.functional.interpolate(x,size=(224,224),mode='bilinear',align_corners=True)
        normalized_x = (x-x.mean(dim=1,keepdim=True))/(x.std(dim=1,keepdim=True)+1e-8)
        features = self(normalized_x)[k]
        #resized = F.upsample(features,x.size()[2:],mode='bilinear')
        return features.data#resized

    def get_all_features(self,x):
        xpt = self.preprocess(x)
        feature_list = self(xpt)
        np_features = [self.rescale_reshape(feature,x.shape) for feature in feature_list]
        return np_features

    def get_random_features(self,x,i=0,num_features=10):
        """x is a numpy image with shape (h x w x 3), i selects from block 1,2,3,4"""
        np_features = self.get_all_features(x)[i]
        projection_matrix = np.random.rand(np_features.shape[-1],num_features)
        projected = np_features@projection_matrix
        normalized = (projected - projected.mean((0,1)))/projected.std((0,1))
        return normalized



