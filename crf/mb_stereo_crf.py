import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from oil.utils.utils import Eval, cosLr
from oil.model_trainers import Trainer,Classifier,Regressor
from oil.utils.mytqdm import tqdm
from crf.crf_module import CRFasRNN, charb
from crf.dataloader import MBStereo14Unary
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

def logits2average_depth(logits):
    probs = F.softmax(logits,dim=1)
    labels = torch.arange(probs.shape[1]).float().to(probs.device)[None,:,None,None]
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

if __name__=='__main__':
    device = torch.device('cuda')
    eps_start=1e-2
    r = 25
    ds=8
    niters=4
    model = CRFwUncertainty(r=r,eps=eps_start,niters=niters).to(device)
    trainset=MBStereo14Unary(downsize=ds)
    train_loader = DataLoader(trainset,batch_size=1,shuffle=True)
    dataloaders = {'train':train_loader,'train_':train_loader}
    dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
    opt_constr = lambda params: torch.optim.Adam(params, lr=3e-3,betas=(.9,.9))
    trialname = 'r_{}_e2_conf_gn_{}_niters{}'.format(r,ds,niters)
    trainer = L1UncRegressor(model,dataloaders,opt_constr,log_suffix=trialname)
    trainer.train(100)