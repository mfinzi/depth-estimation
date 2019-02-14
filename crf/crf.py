import torch
import torch.nn.functional as F
import numpy as np

def gaussian_weights(f):
    """[f] n x c flattened reference 'image' to filter by"""
    I = torch.eye(f.shape[0]).to(f.device)
    W = torch.exp(-((f[None,:,:] - f[:,None,:])**2).sum(-1))-I
    D = W@torch.ones(f.shape[0]).to(f.device)
    D_invsqrt = torch.diag(1/torch.sqrt(D))
    W_normalized = D_invsqrt@W@D_invsqrt - I
    return W_normalized

def gaussian_weights_u(f):
    I = torch.eye(f.shape[0]).to(f.device)
    W = torch.exp(-((f[None,:,:] - f[:,None,:])**2).sum(-1))-I
    return W

def lazy_W(f):
    def W(i,j):
        square_dist = ((f-f[i,j])**2).sum(-1).reshape(-1)
        a = np.exp(-square_dist)
        d = a@np.ones(a.shape[0])-1 # for self include
        a /= np.sqrt(d)
        weights = a.reshape(f.shape[:2])
        return weights
    return W

def charbonneir(a,b,gamma=.1):
    return torch.sqrt(gamma**2 + (a-b)**2) - gamma

def compatibility_matrix(compat,labels):
    return compat(labels[:,None],labels[None,:])

def mean_field_infer(E_0,W,Mu,niters=10):
    """ Performs niters iterations of mean field inference for CRF,
        Inputs:
            [E_0]   n x L    Unary Potentials
            [W]     n x n    Binary Weights
            [Mu]    L x L    Compatibility matrix 
        Outputs: 
            [Q]     n x L:   output label probabilities"""
    Q = F.softmax(-E_0, dim=1)
    for i in range(niters):
        E = E_0 + W@Q@Mu
        Q = F.softmax(-E, dim=1)
    return Q


import gc
import inspect
def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [k for k, v in callers_local_vars if v is var][0]
## MEM utils ##
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
            #print("Name: {}".format(get_var_name(tensor)))
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)