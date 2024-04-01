import torch.utils.data as data
import torch
import os
import json
import random
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as tr
import torchvision.transforms.functional as F
import monai.transforms as mtr
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    )
'''

 Flip Intensities
 Flip Labels
 Horizontal/Vertical Flip
 Sobel-Edge Label
 Task Affine Shift
 Task Brightness Contrast Change
 Task Elastic Warp
 Task Gaussian Blur
 Task Gaussian Noise
 Task Sharpness Change
'''
class Task_FlipIntensity(object):
    '''
    Flip Intensities for all images, using 1-image for each
    shape of data['image'] is [npairs,C,H,W]
    '''
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        data['image'] = 1 - data['image'] # flip intensity
        return data

class Task_FlipLabel(object):
    '''
    Flip Labels for all label, using 1-label for each
    shape of data['image'] is [npairs,C,H,W]
    '''
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        data['label'] = 1 - data['label'] # flip label
        return data


class Task_HorizontalFlip(object):
    '''
    Horizontal Flip for all images and labels
    shape of data['image'] is [npairs,C,H,W]
    '''
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        
        data['image'] = F.hflip(data['image']) # horizontal flip
        data['label'] = F.hflip(data['label']) # horizontal flip
        return data

class Task_VerticalFlip(object):
    '''
    Vertical Flip for all images and labels
    shape of data['image'] is [npairs,C,H,W]
    '''
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        data['image'] = F.vflip(data['image']) # vertical flip
        data['label'] = F.vflip(data['label']) # vertical flip
        return data

class Task_SobelEdge(object):
    '''
    Sobel Edge for all labels
    apply sobel filter in the x and y direction, compute the squard norm of the result
    shape of data['image'] is [npairs,C,H,W]
    '''
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        assert data['label'].shape[1] == 1
        label = data['label']
        for i in range(label.shape[0]):
            label_grads = mtr.SobelGradients()(label[i])
            label[i,0,:,:] = ((label_grads[0]**2 + label_grads[1]**2)>0).float()
        data['label'] = label
        return data
    
class Task_AffineShift(object):
    '''
    Affine Shift for all images and labels
    shape of data['image'] is [npairs,C,H,W]
    rotate_range: radians
    translate_range: float, or tuple of float (min, max), 0 means no change
    scale_range: float, or tuple of float (min, max), 1 means no change
    shear_range: float, or tuple of float (min, max), 0 means no change
    '''
    def __init__(self, 
                prob=0.5, 
                rotato_range=None, # degrees
                translate_range=None,
                scale_range=None,
                shear_range=None,
                interpolation: InterpolationMode = InterpolationMode.NEAREST,
                fill: List[float] | None = None,
                center: List[int] | None = None
                ):
        self.prob = prob
        self.rotato_range = rotato_range
        self.shear_range = shear_range
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.fill = fill
        self.center = center
        self.param_init()

    def param_init(self):
        if self.rotato_range is None:
            raise ValueError('rotato_range must be set')
        elif isinstance(self.rotato_range, float):
            self.rotato_range = (-self.rotato_range, self.rotato_range)

        if self.translate_range is not None:
            if isinstance(self.translate_range, float):
                self.translate_range = (self.translate_range, self.translate_range)
        if self.scale_range is not None and isinstance(self.scale_range, float):
            self.scale_range = (1-self.scale_range, 1+self.scale_range)
        
    def get_params(self,
                degrees: Tuple[float, float],
                translate,
                scale_ranges,
                shears,
                img_size,
                )-> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation.
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])

        if translate is not None:
            assert 0 <= translate[0] <= 1.0 and 0<= translate[1] <= 1.0
            translate_params = random.uniform(translate[0], translate[1])
            max_dx = float(translate_params * img_size[0])
            max_dy = float(translate_params * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear


    def __call__(self, data):
        if random.random() > self.prob:
            return data
        
        img_size = data['image'].shape[-2:]

        angle, translations, scale, shear = self.get_params(
            self.rotato_range,
            self.translate_range,
            self.scale_range,
            self.shear_range,
            img_size,
        )
        for i in range(data['image'].shape[0]):
            data['image'][i] = F.affine(data['image'][i], angle, translations, scale, shear, self.interpolation, self.fill, self.center)
            data['label'][i] = F.affine(data['label'][i], angle, translations, scale, shear, self.interpolation, self.fill, self.center)
        return data 


class Task_BrightnessContrast(object):
    '''
    Brightness Contrast Change for all images
    shape of data['image'] is [npairs,C,H,W]
    brightness_range: float, or tuple of float (min, max), 1 means no change
    contrast_range: float, or tuple of float (min, max), 1 means no change
    '''
    def __init__(self, 
                prob=0.5, 
                brightness_range=None, 
                contrast_range=None,
                device=None,
                ):
        self.prob = prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.device = device
        self.param_init()

    def param_init(self):
        if self.brightness_range is None:
            self.brightness_range = (1, 1)
        elif isinstance(self.brightness_range, float):
            self.brightness_range = (1-self.brightness_range, 1+self.brightness_range)
        if self.contrast_range is None:
            self.contrast_range = (1, 1)
        elif isinstance(self.contrast_range, float):
            self.contrast_range = (1-self.contrast_range, 1+self.contrast_range)

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        
        brightness_params = random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast_params = random.uniform(self.contrast_range[0], self.contrast_range[1])
        for i in range(data['image'].shape[0]):
            data['image'][i] = F.adjust_brightness(data['image'][i], brightness_params)
            data['image'][i] = F.adjust_contrast(data['image'][i], contrast_params)
        return data 

class Task_ElasticWarp(object):
    '''
    Elastic Warp for all images and labels
    shape of data['image'] is [npairs,C,H,W]
    alpha: float, alpha for the random elastic deformation
    sigma: float, sigma for the random elastic deformation
    '''
    def __init__(self, 
                prob=0.5, 
                alpha=50, 
                sigma=5,
                mode=GridSampleMode.NEAREST,
                device=None,
                ):
        self.prob = prob
        self.alpha = alpha
        self.sigma = sigma
        self.mode = mode
        self.device = device
        self.param_init()

        self.elastic = mtr.Rand2DElasticd(
            ['image','label'],
            self.alpha,
            self.sigma,
            prob=1,
            mode=self.mode,
        )

    def param_init(self):
        if isinstance(self.sigma,(int,float)):
            self.sigma = (self.sigma, self.sigma)
        if isinstance(self.alpha,(int,float)):
            self.alpha = (self.alpha, self.alpha)

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        rand_state = np.random.RandomState(None)

        for i in range(data['image'].shape[0]):
            self.elastic.set_random_state(rand_state) 
            data_dict = self.elastic({'image':data['image'][i],'label':data['label'][i]})
            data['image'][i] = data_dict['image']
            data['label'][i] = data_dict['label']
        return data
    
class Task_GaussianBlur(object):
    '''
    Gaussian Blur for all images
    shape of data['image'] is [npairs,C,H,W]
    sigma: tuple of floats, sigma for the gaussian blur
    '''
    def __init__(self, 
                prob=0.5, 
                kernel_size=5,
                sigma=(0.1,1.1),
                device=None,
                ):
        self.prob = prob
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device


    def __call__(self, data):
        if random.random() > self.prob:
            return data
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        for i in range(data['image'].shape[0]):
            data['image'][i] = F.gaussian_blur(data['image'][i], self.kernel_size, sigma)
        return data

class Task_GaussianNoise(object):
    '''
    Gaussian Noise for all images
    shape of data['image'] is [npairs,C,H,W]
    mean: float, mean for the gaussian noise
    std: float, std for the gaussian noise
    '''
    def __init__(self, 
                prob=0.5, 
                mean=0,
                std=0.1,
                device=None,
                ):
        self.prob = prob
        self.mean = mean
        self.std = std
        self.device = device
        self.param_init()
    
    def param_init(self):
        if isinstance(self.std,(int,float)):
            self.std = (self.std, self.std)
        if isinstance(self.mean,(int,float)):
            self.mean = (self.mean, self.mean)

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        std = random.uniform(self.std[0], self.std[1])
        mean = random.uniform(self.mean[0], self.mean[1])
        noise = mtr.RandGaussianNoise(
            prob=1,
            mean=mean,
            std=std,
        )
        for i in range(data['image'].shape[0]):
            data['image'][i] = noise(data['image'][i])
        return data

class Task_Sharpness(object):
    '''
    Sharpness Change for all images
    shape of data['image'] is [npairs,C,H,W]
    alpha: float, alpha for the sharpness change
    '''
    def __init__(self, 
                prob=0.5, 
                sharpness=5,
                device=None,
                ):
        self.prob = prob
        self.sharpness = sharpness
        self.device = device
        self.param_init()
    
    def param_init(self):
        if isinstance(self.sharpness,(int,float)):
            self.sharpness = (self.sharpness, self.sharpness)

    def __call__(self, data):
        if random.random() > self.prob:
            return data
        sharpness = random.uniform(self.sharpness[0], self.sharpness[1])
        for i in range(data['image'].shape[0]):
            data['image'][i] = F.adjust_sharpness(data['image'][i], sharpness)
        return data



   