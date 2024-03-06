import os
import json
import argparse
from constants import *


def find_key_by_value(d, v):
    for key, value in d.items():
        if value == v:
            return key
    
    return None

def find_label_conversion_index(index):
    for i, indexes in enumerate(LABEL_CONVERSION):
        for j in indexes:
            if j == index:
                return i
    return -1

def convert_labels_43_to_19(json_path):
    with open(json_path) as file:
        data = json.load(file)
    
    labels = data["labels"]
    new_labels = []
    for label in labels:
        index = ORIGINAL_LABELS[label]
        labelConversionIndex = find_label_conversion_index(index)
        if labelConversionIndex >= 0:
            new_label = find_key_by_value(LABELS_19, labelConversionIndex)
            new_labels.append(new_label)
    
    data["labels"] = new_labels
    return data

    
def convert(in_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    json_files = os.listdir(in_path)
    for json_file in json_files:
        print(json_file)
        data = convert_labels_43_to_19(os.path.join(in_path, json_file))
        if data["labels"] > 0:
            with open(os.path.join(out_path, json_file), "w") as outfile:
                json.dump(data, outfile)



if __name__ == '__main__':

    parser = argparse.ArgumentParser("Convert BigEarthNet 43 labels to 19 labels")

    parser.add_argument('--in-folder-path', help='input folder path', required=True)
    parser.add_argument('--out-folder-path', help='output folder path', required=True)

    args = parser.parse_args()

    in_folder_path = args.in_folder_path
    out_folder_path = args.out_folder_path

    convert(in_folder_path, out_folder_path)