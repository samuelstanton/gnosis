# https://github.com/mfinzi/olive-oil-ml/blob/master/oil/utils/utils.py

import scipy
import numpy as np
import torch
import functools
from gnosis.utils.metrics import Eval


class Wrapper(object):
    # Special methods are dispatched by what is defined in the class rather
    # than the instance, so it bypasses __getattr__, as a result for a
    # wrapper that makes use of any of these methods, we must dynamically dispatch
    # the special methods at the instance level (using getattr)
    def __init__(self, obj):
        self._wrapped_obj = obj

    def __getattr__(self, attr):
        if attr =='_wrapped_obj': raise AttributeError
        if attr == '__dict__': assert False
        #if attr not in self.__dict__: raise AttributeError
        return getattr(self._wrapped_obj, attr)


smethods =    '''__bool__ __int__ __float__ __complex__ __index__
                 __len__ __getitem__ __setitem__ __delitem__ __contains__
                 __iter__ __next__ __reversed__
                 __call__ __enter__ __exit__
                 __str__ __repr__  __bytes__ __format__
                 __eq__ __ne__ __lt__ __le__ __gt__ __ge__ __hash__
                 __add__ __mul__ __sub__ __truediv__ __floordiv__ __mod__
                 __and__ __or__ __xor__ __invert__ __lshift__ __rshift__
                 __pos__ __neg__ __abs__ __pow__ __divmod__
                 __round__ __ceil__ __floor__ __trunc__
                 __radd__ __rmul__ __rsub__ __rtruediv__ __rfloordiv__ __rmod__
                 __rand__ __ror__ __rxor__ __rlshift__ __rrshift__
                 __rpow__ __rdivmod__ __getitem__ 
                 __get__ __set__ __delete__
                 __dir__ __sizeof__'''.split()
for sm in smethods:
    setattr(Wrapper, sm, lambda self, *args, sm=sm: Wrapper.__getattr__(self,sm)(*args))


class dmap(Wrapper):
    def __init__(self,func,dataset):
        super().__init__(dataset)
        self._func = func

    def __getitem__(self,i):
        return self._func(super().__getitem__(i))


class imap(Wrapper):
    def __init__(self,func,loader):
        super().__init__(loader)
        self._func = func

    def __iter__(self):
        return map(self._func,super().__iter__())


def minibatch_to(mb,device=None,dtype=None):
    try:
        return mb.to(device=device,dtype=dtype)
    except AttributeError:
        if isinstance(mb,dict):
            return type(mb)(((k,minibatch_to(v,device,dtype)) for k,v in mb.items()))
        else:
            return type(mb)(minibatch_to(elem,device,dtype) for elem in mb)


def LoaderTo(loader,device=None,dtype=None):
    return imap(functools.partial(minibatch_to,device=device,dtype=dtype),loader)


class GanLoader(object):
    """ Dataloader class for the generator"""
    def __init__(self,G,N=10**10,bs=64):
        self.G, self.N, self.bs = G,N,bs
    def __len__(self):
        return self.N
    def __iter__(self):
        with torch.no_grad(),Eval(self.G):
            for i in range(self.N//self.bs):
                yield self.G.sample(self.bs)
            if self.N%self.bs!=0:
                yield self.G.sample(self.N%self.bs)

    def write_imgs(self,path):
        np_images = np.concatenate([img.cpu().numpy() for img in self],axis=0)
        for i,img in enumerate(np_images):
            scipy.misc.imsave(path+'img{}.jpg'.format(i), img)
