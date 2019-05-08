import torch
from numpy.testing import assert_array_almost_equal

import os
import sys
sys.path.append(os.getcwd())

from hosvd import hosvd

def shape2index(n):
    indexs = ""
    assert n < 8
    for i in range(n):
        indexs = indexs + chr(97+i)
    return indexs

def transformationStrBuilder(n):
    metaStr = shape2index(n)
    metaStr += ","
    for i in range(n):
        metaStr += chr(97+i)+chr(97+n+i)
        if i != n-1:
            metaStr += ","
    metaStr += "->"
    for i in range(n):
        metaStr += chr(97+n+i)
    return metaStr

def taskn(n,decimal):
    shape = range(n,2*n)
    A = torch.randn(*shape)
    S,Ulist = hosvd(A)
    metaStr = transformationStrBuilder(n)
    AA = torch.einsum(metaStr,S,*Ulist)
    assert_array_almost_equal(A.numpy(),A.numpy(),decimal=decimal)

def test_3_dims():
    taskn(3,6)