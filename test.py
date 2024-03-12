
import os
import json
import numpy as np
from osgeo import gdal
import shutil

countries = [
    "Finland",
    "Portugal",
    "Serbia",
    "Lithuania",
    "Austria",
    "Ireland",
    "Belgium"
]

train_countries = ["Finland", "Portugal", "Serbia"]
val_countries = ["Lithuania", "Austria"]
test_countries = ["Ireland", "Belgium"]

root = "data/bigarthnet/train"

label_files = os.listdir(os.path.join(root, "labels"))

for label_file in label_files:
    with open(os.path.join(root, "labels", label_file)) as file:
        data = json.load(file)

    country = data["country"]

    if country in val_countries:
        folder_name = "_".join(label_file.split("_")[:-2])
        shutil.move(os.path.join(root, "images", folder_name), os.path.join("data/bigarthnet/val/images", folder_name))
        shutil.move(os.path.join(root, "labels", label_file), os.path.join("data/bigarthnet/val/labels", label_file))
        
    elif country in test_countries:
        folder_name = "_".join(label_file.split("_")[:-2])
        shutil.move(os.path.join(root, "images", folder_name), os.path.join("data/bigarthnet/test/images", folder_name))
        shutil.move(os.path.join(root, "labels", label_file), os.path.join("data/bigarthnet/test/labels", label_file))


