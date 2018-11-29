import torch
#import lattice
from torch.utils.cpp_extension import load
lattice = load(name="lattice",sources=["lattice.cpp"])
ref = torch.rand(1000000,6)
src = torch.rand(1000000,6)
#out = torch.zeros(1000000,16)
#print(lattice.testfunc(torch.ones((2,3))))
print(lattice.filter(src,ref))
# Problem with undefined symbols is in the gcc version whihc is too new, need 4.9.2