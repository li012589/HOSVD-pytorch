import torch
import numpy as np

def unfolding(n,A):
    shape = A.shape
    size = np.prod(shape)
    lsize = size // shape[n]
    sizelist = list(range(len(shape)))
    sizelist[n] = 0
    sizelist[0] = n
    return A.permute(sizelist).reshape(shape[n],lsize)

def modalsvd(n,A):
    nA = unfolding(n,A)
    return torch.svd(nA)

def hosvd(A):
    Ulist = []
    S = A
    for i,ni in enumerate(A.shape):
        u,_,_ = modalsvd(i,A)
        Ulist.append(u)
        S = torch.tensordot(S,u.t(),dims=([0],[0]))

    return S,Ulist