import torch
import time

a = torch.rand(8,50000)#.cuda()
b = torch.rand(50000,8)#.cuda()
start = time.time()
b@a
print(time.time()-start)
