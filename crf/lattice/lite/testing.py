import torch
print(torch.zeros(10))
#import lattice
from torch.utils.cpp_extension import load
sigmoid = load(name="sigmoid",sources=["testing.cpp"])
ref = torch.rand(1000,5)
src = torch.rand(1000,2)
out = torch.zeros(1000,2)
#print(lattice.testfunc(torch.ones((2,3))))
print(sigmoid.d_sigmoid(ref))