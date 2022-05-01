import numpy as np
import paddle

def pdnorm(x,dim=None):
    out = np.linalg.norm(x,axis=dim)
    return paddle.to_tensor(out,dtype='float32')

def pdzeros(size, dtype=None, device=None):
    out = paddle.zeros(shape=size,dtype=dtype)
    return out

def pdones(size, dtype=None, device=None):
    out = paddle.ones(shape=size,dtype=dtype)
    return out

def pdtensordot(x,y,dims=2):
    out = paddle.tensordot(x,y,axes=dims)
    return out

def pdtensor(x, dtype=None, device=None):
    out = paddle.to_tensor(x,dtype=dtype)
    return out

def pdsum(x, dim=None):
    out = paddle.sum(x,axis=dim)
    return out

def pdprod(x, dim=None):
    out = paddle.prod(x,axis=dim)
    return out

def pdstack(x,dim=None):
    out = paddle.stack(x,axis=dim)
    return out

def pdmax(x,dim=None):
    out = paddle.max(x,axis=dim)
    return out

def pdmean(x,dim=None):
    out = paddle.mean(x,axis=dim)
    return out

def pdroll(input,shifts,dims=None):
    out = paddle.roll(input,shifts,axis=dims)
    return out

def cuda_available():
    return paddle.is_compiled_with_cuda()