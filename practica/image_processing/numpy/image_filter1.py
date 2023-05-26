import io
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

class ImageFilter(object):
    
    def __init__(self, kernel=None):
        if kernel is not None:
            self.imgKernel = kernel
    
    def convolve(self, imgTensor):
        imgTensorRGB = imgTensor.copy() 
        outputImgRGB = np.empty_like(imgTensorRGB)

        for dim in range(imgTensorRGB.shape[-1]):  # loop over rgb channels
            outputImgRGB[:, :, dim] = sp.signal.convolve2d (
                imgTensorRGB[:, :, dim], self.imgKernel, mode="same", boundary="symm"
            )

        return outputImgRGB
 
    def downSample(self, imgTensor):
        # return block_reduce(image=imgTensor, block_size=(2,2,1), func=np.max) 
        return block_reduce(image=imgTensor, block_size=(2,2,1), func=np.max)
        # 2,2,1 is lengte, breedte, kleur. Je kan ook 1 -> 3
        # np.max kan ook np.mean of sum of alles door de helft delen 