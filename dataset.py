import copy
import random
import os
import json
import bisect
import warnings
import numpy as np
import cv2
import torch
import presets
from osgeo import gdal
from torch.utils.data import Dataset as PytorchDataset
from constants import *
from torchvision.transforms.functional import InterpolationMode
from typing import Iterator, Iterable, Optional, List, TypeVar, Generic, Sized

T_co = TypeVar("T_co", covariant=True)

class Dataset(Generic[T_co]):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        return ConcatDataset([self, other])

class ConcatDataset(Dataset[T_co]):

    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "cummulative_sizes attribute is renamed to " "cumulative_sizes",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cumulative_sizes
    
class Sampler(Generic[T_co]):

    def __init__(self, data_source: Optional[Sized] = None) -> None:
        if data_source is not None:
            import warnings

            warnings.warn("`data_source` argument is not used and will be removed in 2.2.0."
                          "You may still have custom implementation that utilizes it.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

class RandomDomainSampler(Sampler):
    def __init__(self, data_source: ConcatDataset, batch_size: int, n_domains_per_batch: int):
        super(Sampler, self).__init__()
        self.n_domains_in_dataset = len(data_source.cumulative_sizes)
        self.n_domains_per_batch = n_domains_per_batch
        assert self.n_domains_in_dataset >= self.n_domains_per_batch

        self.sample_idxes_per_domain = []
        start = 0
        for end in data_source.cumulative_sizes:
            idxes = [idx for idx in range(start, end)]
            self.sample_idxes_per_domain.append(idxes)
            start = end

        assert batch_size % n_domains_per_batch == 0
        self.batch_size_per_domain = batch_size // n_domains_per_batch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.n_domains_in_dataset)]
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            selected_domains = random.sample(domain_idxes, self.n_domains_per_batch)

            for domain in selected_domains:
                sample_idxes = sample_idxes_per_domain[domain]
                if len(sample_idxes) < self.batch_size_per_domain:
                    selected_idxes = np.random.choice(sample_idxes, self.batch_size_per_domain, replace=True)
                else:
                    selected_idxes = random.sample(sample_idxes, self.batch_size_per_domain)
                final_idxes.extend(selected_idxes)

                for idx in selected_idxes:
                    if idx in sample_idxes_per_domain[domain]:
                        sample_idxes_per_domain[domain].remove(idx)

                remaining_size = len(sample_idxes_per_domain[domain])
                if remaining_size < self.batch_size_per_domain:
                    stop_flag = True

        return iter(final_idxes)

    def __len__(self):
        return self.length


CLASSES = [
    "Marine waters",
    "Arable land",
    "Urban fabric",
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

class BigEarthNetDataset(PytorchDataset):
    def __init__(self, root, labels_directory="labels", bands = ["B01", "B02", "B03", "B04"], transform=None):
        super().__init__()
        
        self.classes = CLASSES
        self.root = root
        self.labels_directory = labels_directory
        self.transform = transform
        self.img_labels = os.listdir(os.path.join(root, labels_directory))
        self.bands = bands
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        json_path = os.path.join(self.root, self.labels_directory, self.img_labels[idx])
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

def load_train_val_data_for_coral(
        data_path,
        train_folders,
        interpolation,
        val_folder,
        val_resize_size,
        val_crop_size,
        train_crop_size,
        batch_size,
        band_groups,
        n_domains_per_batch
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

    train_folder = train_folders[0]

    train_dir = os.path.join(data_path, train_folder)

    finland_dataset =  BigEarthNetDataset(train_dir, bands=bands,labels_directory="labels_domain/Finland", 
                                          transform=presets.ClassificationPresetTrain(
                                              crop_size=train_crop_size,
                                              interpolation=interpolation,mean=mean,
                                              std=std
                                        ))
    
    portugal_dataset =  BigEarthNetDataset(train_dir, bands=bands,labels_directory="labels_domain/Portugal", 
                                          transform=presets.ClassificationPresetTrain(
                                              crop_size=train_crop_size,
                                              interpolation=interpolation,mean=mean,
                                              std=std
                                        ))

    serbia_dataset =  BigEarthNetDataset(train_dir, bands=bands,labels_directory="labels_domain/Serbia", 
                                          transform=presets.ClassificationPresetTrain(
                                              crop_size=train_crop_size,
                                              interpolation=interpolation,mean=mean,
                                              std=std
                                        ))


    tr_dataset = ConcatDataset([finland_dataset, portugal_dataset, serbia_dataset])
    tr_dataset.classes = val_dataset.classes

    train_sampler = RandomDomainSampler(tr_dataset, batch_size, n_domains_per_batch=n_domains_per_batch)

    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    
    return tr_dataset, val_dataset, train_sampler, val_sampler

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
