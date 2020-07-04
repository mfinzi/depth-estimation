import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from oil.utils.utils import Eval, cosLr
from oil.model_trainers import Trainer,Classifier,Regressor
from oil.utils.mytqdm import tqdm
from crf.crf_module import CRFasRNN, charb
from crf.dataloader import MBStereo14Unary,StereoUpsampling05
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo
import torchvision.utils as vutils
from oil.architectures.parts.blocks import conv2d,ConvBNrelu,ResBlock
class L1Regressor(Trainer):
    """ """

    def loss(self, minibatch):
        x,y = minibatch
        return nn.L1Loss()(self.model(x),y)

    def metrics(self,loader):
        L1Loss = lambda mb: nn.L1Loss()(self.model(mb[0]),mb[1]).cpu().data.numpy()
        return {'L1Loss':self.evalAverageMetrics(loader,L1Loss)}

    def logStuff(self,step,minibatch=None):
        if minibatch is not None:
            target_img = minibatch[1][0].cpu().data
            depth_output_img = self.model(minibatch[0])[0].cpu().data
            unary_depth = logits2average_depth(minibatch[0][0])[0].cpu().data
            img_pair = [unary_depth,depth_output_img,target_img]
            img_grid = vutils.make_grid(img_pair, normalize=True)
            self.logger.add_image('depth_maps', img_grid, step)

        print(self.model.CRF.Mu.s,self.model.CRF.Mu.gamma)
        super().logStuff(step,minibatch)

class L1UncRegressor(Trainer):
    def loss(self, minibatch):
        x,y = minibatch
        depth,conf = self.model(x)
        return nn.L1Loss()(depth,y)#nn.L1Loss()(conf*depth,conf*y) - torch.log(conf).mean()

    def metrics(self,loader):
        L1Loss = lambda mb: nn.L1Loss()(self.model(mb[0])[0],mb[1]).cpu().data.numpy()
        return {'L1Loss':self.evalAverageMetrics(loader,L1Loss)}

    def logStuff(self,step,minibatch=None):
        if minibatch is not None:
            target_img = minibatch[1][0].cpu().data
            depth,conf = self.model(minibatch[0])
            depth_output_img = depth[0].cpu().data
            confimg = conf[0].cpu().data
            unary_depth = logits2average_depth(minibatch[0][0])[0].cpu().data
            img_pair = [unary_depth,depth_output_img,target_img]
            img_grid = vutils.make_grid(img_pair, normalize=True)
            self.logger.add_image('depth_maps', img_grid, step)
            self.logger.add_image('conf_img', vutils.make_grid([confimg],normalize=True), step)

        print(self.model.CRF.Mu.s,self.model.CRF.Mu.gamma)
        super().logStuff(step,minibatch)

def logits2average_depth(logits,labels=None):
    probs = F.softmax(logits,dim=1)
    if labels is None: labels = torch.arange(probs.shape[1]).float().to(probs.device)[None,:,None,None]
    average_depths = (probs*labels).sum(1)
    return average_depths[:,None,:,:]

class CRFdepthRefiner(nn.Module):
    def __init__(self, d_in=64,d_guide=16,r=15,niters=2,eps=1e-2,gamma=.05):
        super().__init__()
        self.CRF = CRFasRNN(charb(gamma),niters=niters,r=r,eps=eps,gchannels=d_guide)
        self.projection = nn.Conv2d(d_in,d_guide-3,kernel_size=1)

    def forward(self,inputs):
        logits,imgrgb,features = inputs
        projected_features = self.projection(features)
        guide = torch.cat((imgrgb,projected_features),dim=1)
        filtered_logits = self.CRF(guide,logits)
        return logits2average_depth(filtered_logits)

class CRFwUncertainty(CRFdepthRefiner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.uncertainty_net = nn.Sequential(
            conv2d(3,16,coords=True),
            nn.GroupNorm(4,16),
            nn.ReLU(),
            conv2d(16,16,coords=True),
            nn.GroupNorm(4,16),
            nn.ReLU(),
            conv2d(16,1,coords=True),
        ) # Outputs log(sigma), bs x 1 x h x w

    def forward(self,inputs):
        logits,imgrgb,features = inputs
        projected_features = self.projection(features)
        guide = torch.cat((imgrgb,projected_features),dim=1)
        s = self.uncertainty_net(imgrgb)
        confidence = torch.exp(-s)
        filtered_logits = self.CRF(guide,logits,confidence)
        depth_out = logits2average_depth(filtered_logits)
        return depth_out,confidence


class Dupsampling(Trainer):
    """ """

    def loss(self, minibatch):
        x,y = minibatch
        model_out = self.model(x)
        #print(model_out)
        nonzero_mask = (y>0).float()
        return nn.L1Loss()(model_out*nonzero_mask,y*nonzero_mask)

    def metrics(self,loader):
        L1Loss = lambda mb: nn.L1Loss()(self.model(mb[0]),mb[1]).cpu().data.numpy()
        return {'L1Loss':self.evalAverageMetrics(loader,L1Loss)}

    def logStuff(self,step,minibatch=None):
        if minibatch is not None:
            target_img = minibatch[1][0].cpu().data
            depth_output_img = self.model(minibatch[0])[0].cpu().data
            f, axarr = plt.subplots(1,3,figsize=(15,10))
            bilinear= F.upsample(minibatch[0][0],target_img.size()[-2:],mode='bilinear')[0].cpu().data
            vmax = depth_output_img.max().cpu().data.numpy()
            axarr[0].imshow(bilinear[0].numpy(),cmap='bone',vmin=0,vmax=vmax)
            axarr[1].imshow(depth_output_img[0].numpy(),cmap='bone',vmin=0,vmax=vmax)
            axarr[2].imshow(target_img[0].numpy(),cmap='bone',vmin=0,vmax=vmax)
            
            plt.show()
            img_pair = [bilinear,depth_output_img,target_img]
            img_grid = vutils.make_grid(img_pair, normalize=True)
            self.logger.add_image('depth_maps', img_grid, step)

        print(self.model.CRF.Mu.s,self.model.CRF.Mu.gamma)
        super().logStuff(step,minibatch)
import matplotlib.pyplot as plt
class CRFdepthUpsampler(nn.Module):
    def __init__(self,d_in=64,d_guide=3,r=15,niters=2,eps=1e-2,gamma=.05):
        super().__init__()
        self.CRF = CRFasRNN(charb(gamma),niters=niters,r=r,eps=eps,gchannels=d_guide)
        #self.projection = nn.Conv2d(d_in,d_guide-3,kernel_size=1)

    def forward(self,inputs):
        disp_lowres,img_highres,vgg_features = inputs
        upsampled_disp = F.upsample(disp_lowres,img_highres.size()[2:],mode='bilinear')
        #projected_features = F.upsample(self.projection(vgg_features),img_highres.size()[2:],mode='bilinear')
        #print(projected_features)
        guide = img_highres#torch.cat((img_highres,projected_features),dim=1)
        max_depth = upsampled_disp.max()
        labels = torch.linspace(0,max_depth,18).float().cuda()
        logits = -10*self.CRF.Mu.get_energies_from_scalar(upsampled_disp,labels[None,:,None,None])
        confidence = (upsampled_disp>1e-2).float()
        #print(confidence.cpu().data.numpy().reshape(-1).mean())
        #plt.hist(upsampled_disp.cpu().data.numpy().reshape(-1))
        filtered_logits = self.CRF(guide,logits,confidence=confidence,labels=labels)
        # if torch.any(torch.isnan(filtered_logits)):
        #     if torch.any(torch.isnan(logits)): print("logits has nans")
        #     if torch.any(torch.isnan(guide)): print("guide has nans")
        #     print("vgg_features_nans?:{}".format(torch.any(torch.isnan(vgg_features))))
        #     assert False, "nans encountered"
        avg_logits = logits2average_depth(filtered_logits,labels[None,:,None,None])
        return avg_logits

if __name__=='__main__':
    eps_start=1e-3
    r = 5
    ds=16
    niters=1
    device = torch.device('cuda')
    model = CRFdepthUpsampler(r=r,eps=eps_start,niters=niters).to(device)
    trainset= StereoUpsampling05(downsize=ds,val=False,use_vgg=False)
    valset = StereoUpsampling05(downsize=ds,val=True,use_vgg=False)
    train_loader = DataLoader(trainset,batch_size=1,shuffle=True)
    val_loader = DataLoader(valset,batch_size=1,shuffle=True)
    dataloaders = {'train':train_loader,'train_':train_loader,'val':val_loader}
    dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
    opt_constr = lambda params: torch.optim.Adam(params, lr=3e-3,betas=(.9,.9))
    trialname = 'upsampling/r_{}_{}_niters{}'.format(r,ds,niters)
    trainer = Dupsampling(model,dataloaders,opt_constr,log_suffix=trialname,log_args={'minPeriod':.1})
    trainer.train(100)

# if __name__=='__main__':
#     device = torch.device('cuda')
#     eps_start=1e-2
#     r = 50
#     model = CRFdepthRefiner(r=r,eps=eps_start).to(device)
#     trainset=MBStereo14Unary(downsize=8)
#     train_loader = DataLoader(trainset,batch_size=1,shuffle=True)
#     dataloaders = {'train':train_loader,'train_':train_loader}
#     dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
#     opt_constr = lambda params: torch.optim.Adam(params, lr=3e-3,betas=(.9,.9))
#     trialname = 'r_{}_e2_c1'.format(r)
#     trainer = L1Regressor(model,dataloaders,opt_constr,log_suffix=trialname)
#     trainer.train(100)

# if __name__=='__main__':
#     device = torch.device('cuda')
#     eps_start=1e-2
#     r = 25
#     ds=8
#     niters=4
#     model = CRFwUncertainty(r=r,eps=eps_start,niters=niters).to(device)
#     trainset=MBStereo14Unary(downsize=ds)
#     train_loader = DataLoader(trainset,batch_size=1,shuffle=True)
#     dataloaders = {'train':train_loader,'train_':train_loader}
#     dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
#     opt_constr = lambda params: torch.optim.Adam(params, lr=3e-3,betas=(.9,.9))
#     trialname = 'r_{}_e2_conf_gn_{}_niters{}'.format(r,ds,niters)
#     trainer = L1UncRegressor(model,dataloaders,opt_constr,log_suffix=trialname)
#     trainer.train(100)