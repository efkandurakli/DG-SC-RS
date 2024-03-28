import os
import time
import torch
import torchvision
import datetime
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import confusion_matrix
import numpy as np
import utils
from dataset import load_test_data
from dg_model import DGModel
from resnet import resnet18



def evaluate(model, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    all_predictions = []
    all_targets = []
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
            _, predictions = output.max(1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    return metric_logger.acc1.global_avg
    


def main(args):
    
    device = torch.device(args.device)
    print(args)

        
    test_dir = os.path.join(args.data_path, args.test_folder)
    
    test_dataset, test_sampler = load_test_data(
        test_dir,
        args.test_resize_size,
        args.test_crop_size,
        args.interpolation,
        args.band_groups
    )
    
    num_classes = len(test_dataset.classes)
    num_channels = len(test_dataset.bands)
    
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
    if args.dg:
        model = DGModel(args.model, weights=None, num_classes=num_classes, num_channels=num_channels)
    else:
        model = resnet18(weights=None, num_classes=num_classes, num_channels=num_channels)
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

    parser.add_argument("--data-path", default="data/bigearthnet", type=str, help="dataset path")
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

    parser.add_argument("--test-folder", default="test", help="the test folder name")

    parser.add_argument(
        "--dg", action="store_true", help="domain generalization model"
    )

    parser.add_argument("--band-groups", default=["rgb"], nargs='+', help="List of train folders")



    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)