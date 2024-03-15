import os
import time
import torch
import torchvision
import datetime
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

import utils
import presets
from torch import nn
from dataset import BigEarthNetDataset




def evaluate(model, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target, _ in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            batch_size = image.shape[0]
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg
    

def load_data(testdir, args):
    print("Loading data")
    test_resize_size, test_crop_size = (
        args.test_resize_size,
        args.test_crop_size,
    )
    
    interpolation = InterpolationMode(args.interpolation)
    

    test_dataset = BigEarthNetDataset(testdir, transform=presets.ClassificationPresetEval(
                        crop_size=test_crop_size,
                        resize_size=test_resize_size,
                        interpolation=interpolation,
                    ))
    
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    
    
    return test_dataset, test_sampler

def main(args):
    
    device = torch.device(args.device)
    print(args)

        
    test_dir = os.path.join(args.data_path, "test")
    
    test_dataset, test_sampler = load_data(test_dir, args)
    
    num_classes = len(test_dataset.classes)
    
    collate_fn = default_collate
    
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=None, num_classes=num_classes)
    model.to(device)
    

    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    print("Start testing")
    start_time = time.time()
    evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="BigearthNet Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default="data/bigarthnet", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")

    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--test-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--test-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )

    parser.add_argument("--model-path", default="lower_bound/best_model.pth", type=str, help="model path")



    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)