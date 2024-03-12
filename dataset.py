import os
import json
import numpy as np
from osgeo import gdal
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

CLASSES = [
    "Marine waters",
    "Arable land",
    "Urban fabric",
    "Pastures",
    "Coniferous forest",
    "Broad-leaved forest",
    "Inland waters",
    "Mixed forest"
]

DOMAINS = [
    "Finland",
    "Portugal",
    "Ireland",
    "Serbia",
    "Lithuania",
    "Austria",
    "Belgium"
]

class BigEarthNetDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        
        self.classes = CLASSES
        self.root = root
        self.transform = transform
        self.img_labels = os.listdir(os.path.join(root, "labels"))
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        json_path = os.path.join(self.root, "labels", self.img_labels[idx])
        with open(json_path) as file:
            data = json.load(file)
        
        label = CLASSES.index(data["labels"][0])
        domain = DOMAINS.index(data["country"])
        
        images_folder = "_".join(self.img_labels[idx].split("_")[:-2])
        image_files = os.listdir(os.path.join(self.root, "images", images_folder))
        
        b2_file = [img_file for img_file in image_files if img_file.split("_")[5][:-4] == "B02"][0]
        b3_file = [img_file for img_file in image_files if img_file.split("_")[5][:-4] == "B03"][0]
        b4_file = [img_file for img_file in image_files if img_file.split("_")[5][:-4] == "B04"][0]
        
        band_b2_ds = gdal.Open(os.path.join(self.root, "images", images_folder, b2_file),  gdal.GA_ReadOnly)
        raster_band_b2 = band_b2_ds.GetRasterBand(1)
        band_data_b2 = raster_band_b2.ReadAsArray()
        
        band_b3_ds = gdal.Open(os.path.join(self.root,"images", images_folder, b3_file),  gdal.GA_ReadOnly)
        raster_band_b3 = band_b3_ds.GetRasterBand(1)
        band_data_b3 = raster_band_b3.ReadAsArray()
        
        band_b4_ds = gdal.Open(os.path.join(self.root, "images", images_folder, b4_file),  gdal.GA_ReadOnly)
        raster_band_b4 = band_b4_ds.GetRasterBand(1)
        band_data_b4 = raster_band_b4.ReadAsArray()
        
        image = np.stack((band_data_b2, band_data_b3, band_data_b4), axis=0).astype(np.float16)
        
        
        if self.transform:
            image = self.transform(torch.from_numpy(image))
        
        return image, label, domain
