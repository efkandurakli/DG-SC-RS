import os
import time
import torch
import torchvision
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

import utils
import presets
from torch import nn
from dataset import BigEarthNetDataset


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    
    header = f"Epoch: [{epoch}]"
    
    for i, (image, target, domain) in enumerate(data_loader):
        print(image.shape)

def load_data(traindir, valdir, args):
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    
    interpolation = InterpolationMode(args.interpolation)
    
    tr_dataset = BigEarthNetDataset(traindir, transform=presets.ClassificationPresetTrain(
                        crop_size=train_crop_size,
                        interpolation=interpolation,
                    ))

    val_dataset = BigEarthNetDataset(valdir, transform=presets.ClassificationPresetEval(
                        crop_size=train_crop_size,
                        resize_size=val_resize_size,
                        interpolation=interpolation,
                    ))
    
    train_sampler = torch.utils.data.RandomSampler(tr_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    
    
    return tr_dataset, val_dataset, train_sampler, val_sampler

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
    
    device = torch.device(args.device)
    print(args)
    
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        
    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    
    tr_dataset, val_dataset, train_sampler, val_sampler = load_data(train_dir, val_dir, args)
    
    num_classes = len(tr_dataset.classes)
    
    collate_fn = default_collate
    
    data_loader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.workers, pin_memory=True
    )
    
    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    
    
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
    
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="BigearthNet Classification Training", add_help=add_help)
    
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    
    parser.add_argument("--data-path", default="data/bigarthnet", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "-j", "--workers", default=12, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)