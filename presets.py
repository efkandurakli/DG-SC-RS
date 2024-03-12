import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(429.9430203, 614.21682446, 590.23569706),
        std=(572.41639287, 582.87945694, 675.88746967),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
    ):
        transforms = []
        transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
            
        transforms.extend([T.Normalize(mean=mean, std=std)])
        
        self.transforms = T.Compose(transforms)
    
    def __call__(self, img):
        return self.transforms(img)
           

class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(429.9430203, 614.21682446, 590.23569706),
        std=(572.41639287, 582.87945694, 675.88746967),
        interpolation=InterpolationMode.BILINEAR,
    ):
        transforms = [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
        ]
            
        transforms.extend([T.Normalize(mean=mean, std=std)])
        
        self.transforms = T.Compose(transforms)
    
    def __call__(self, img):
        return self.transforms(img)

