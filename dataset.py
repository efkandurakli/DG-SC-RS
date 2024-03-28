import os
import json
import numpy as np
import cv2
from osgeo import gdal
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import presets
from constants import *
from torchvision.transforms.functional import InterpolationMode

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
    "Serbia",
    "Ireland",
    "Lithuania",
    "Austria",
    "Belgium"
]

class BigEarthNetDataset(Dataset):
    def __init__(self, root, bands = ["B01", "B02", "B03", "B04"], transform=None):
        super().__init__()
        
        self.classes = CLASSES
        self.root = root
        self.transform = transform
        self.img_labels = os.listdir(os.path.join(root, "labels"))
        self.bands = bands
        
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

        bands = []

        for band in self.bands:
            tiff_file = [img_file for img_file in image_files if img_file.split("_")[5][:-4] == band][0]
            band_data = cv2.imread(os.path.join(self.root, "images", images_folder, tiff_file),  cv2.IMREAD_UNCHANGED)
            bands.append(band_data)

        image = np.stack(bands, axis=0).astype(np.float32)

        if self.transform:
            image = self.transform(torch.from_numpy(image))
        
        return image, label, domain


def get_bands_mean_std(band_groups):

    bands = []
    mean = []
    std = []

    if "rgb" in band_groups:
        bands.extend(["B01", "B02", "B03", "B04"])
        mean.extend([
            BAND_STATS["S2"]["mean"]["B01"],
            BAND_STATS["S2"]["mean"]["B02"],
            BAND_STATS["S2"]["mean"]["B03"],
            BAND_STATS["S2"]["mean"]["B04"]
        ])
        std.extend([
            BAND_STATS["S2"]["std"]["B01"],
            BAND_STATS["S2"]["std"]["B02"],
            BAND_STATS["S2"]["std"]["B03"],
            BAND_STATS["S2"]["std"]["B04"]
        ])

    if "nir" in band_groups:
        bands.extend(["B05", "B06", "B07", "B08", "B8A"])
        mean.extend([
            BAND_STATS["S2"]["mean"]["B05"],
            BAND_STATS["S2"]["mean"]["B06"],
            BAND_STATS["S2"]["mean"]["B07"],
            BAND_STATS["S2"]["mean"]["B08"],
            BAND_STATS["S2"]["mean"]["B8A"]
        ])

        std.extend([
            BAND_STATS["S2"]["std"]["B05"],
            BAND_STATS["S2"]["std"]["B06"],
            BAND_STATS["S2"]["std"]["B07"],
            BAND_STATS["S2"]["std"]["B08"],
            BAND_STATS["S2"]["std"]["B8A"]
        ])

    if "swir" in band_groups:
        bands.extend(["B09", "B11", "B12"])
        mean.extend([
            BAND_STATS["S2"]["mean"]["B09"],
            BAND_STATS["S2"]["mean"]["B11"],
            BAND_STATS["S2"]["mean"]["B12"],
        ])
        std.extend([
            BAND_STATS["S2"]["std"]["B09"],
            BAND_STATS["S2"]["std"]["B11"],
            BAND_STATS["S2"]["std"]["B12"],
        ])
    return bands, mean, std

def load_train_val_data(
        data_path,
        train_folders,
        interpolation,
        val_folder,
        val_resize_size,
        val_crop_size,
        train_crop_size,
        band_groups
    ):
    print("Loading data")
    
    interpolation = InterpolationMode(interpolation)
    
    val_dir = os.path.join(data_path, val_folder)


    bands, mean, std = get_bands_mean_std(band_groups)


    val_dataset = BigEarthNetDataset(val_dir,  bands=bands, transform=presets.ClassificationPresetEval(
                        crop_size=val_crop_size,
                        resize_size=val_resize_size,
                        interpolation=interpolation,
                        mean=mean,
                        std=std
                    ))

    tr_datasets = []

    for train_folder in train_folders:
        train_dir = os.path.join(data_path, train_folder)
        tr_datasets.append(
            BigEarthNetDataset(train_dir, bands=bands,  transform=presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                mean=mean,
                std=std
            ))
        )

    tr_dataset = torch.utils.data.ConcatDataset(tr_datasets)
    tr_dataset.classes = val_dataset.classes

    train_sampler = torch.utils.data.RandomSampler(tr_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    
    return tr_dataset, val_dataset, train_sampler, val_sampler

def load_test_data(
        testdir,
        test_resize_size,
        test_crop_size,
        interpolation,
        band_groups
    ):
    print("Loading data")
    
    interpolation = InterpolationMode(interpolation)

    bands, mean, std = get_bands_mean_std(band_groups)
    

    test_dataset = BigEarthNetDataset(testdir, bands=bands, transform=presets.ClassificationPresetEval(
                        crop_size=test_crop_size,
                        resize_size=test_resize_size,
                        interpolation=interpolation,
                        mean=mean,
                        std=std
                    ))
    
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    
    
    return test_dataset, test_sampler
