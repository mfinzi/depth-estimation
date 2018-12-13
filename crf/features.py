import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn

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
        normalized = (x - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        batch_input_reshaped = normalized[None,...].transpose((0,3,1,2))
        device = next(self.features.parameters()).device
        img_batch = torch.from_numpy(batch_input_reshaped).float().to(device)
        resized = nn.functional.interpolate(img_batch,size=(224,224),mode='bilinear')
        return img_batch

    def get_all_features(self,x):
        xpt = self.preprocess(x)
        feature_list = self(xpt)
        np_features = [feature.cpu().data.numpy()[0].transpose((1,2,0)) for feature in feature_list]
        return np_features

    def get_random_features(self,x,i=0,num_features=10):
        """x is a numpy image with shape (h x w x 3), i selects from block 1,2,3,4"""
        xpt = self.preprocess(x)
        feature_list = self(xpt)
        np_features = feature_list[i].cpu().data.numpy()[0].transpose((1,2,0))
        projection_matrix = np.random.rand(np_features.shape[-1],num_features)
        return np_features@projection_matrix



