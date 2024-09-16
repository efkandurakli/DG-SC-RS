import os
import time
import torch
import datetime
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import cohen_kappa_score

import utils
from torch import nn
from dataset import load_train_val_data_for_coral
from resnet import resnet18
from constants import *
from automatic_weighted_loss import AutomaticWeightedLoss
import torch.nn.functional as F
import torch.autograd as autograd

class InvariancePenaltyLoss(nn.Module):
    def __init__(self):
        super(InvariancePenaltyLoss, self).__init__()
        self.scale = torch.tensor(1.).requires_grad_()

    def forward(self, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_1 = F.cross_entropy(y[::2] * self.scale, labels[::2])
        loss_2 = F.cross_entropy(y[1::2] * self.scale, labels[1::2])
        grad_1 = autograd.grad(loss_1, [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [self.scale], create_graph=True)[0]
        penalty = torch.sum(grad_1 * grad_2)
        return penalty
    
def train_one_epoch(model, train_iter, criterion_class, criterion_irm, optimizer, device, epoch, awl, args):
    model.train()
    batch_time = utils.AverageMeter('Time', ':4.2f')
    data_time = utils.AverageMeter('Data', ':3.1f')
    losses = utils.AverageMeter('Loss', ':3.5f')
    losses_ce = utils.AverageMeter('CELoss', ':3.5f')
    losses_penalty = utils.AverageMeter('Penalty Loss', ':3.5f')
    cls_accs = utils.AverageMeter('Cls Acc', ':3.3f')

    progress = utils.ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, losses_ce, losses_penalty, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_all, labels_all, _ = next(train_iter)
        x_all = x_all.to(device)
        labels_all = labels_all.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        y_all = model(x_all)

        loss_ce = criterion_class(y_all, labels_all)

        loss_penalty = 0
        for y_per_domain, labels_per_domain in zip(y_all.chunk(args.n_domains_per_batch, dim=0),
                                                   labels_all.chunk(args.n_domains_per_batch, dim=0)):
            # normalize loss by domain num
            loss_penalty += criterion_irm(y_per_domain, labels_per_domain) / args.n_domains_per_batch

        if awl:
            loss = awl(loss_ce, loss_penalty)
        else:
            loss = loss_ce + loss_penalty * args.trade_off

        cls_acc = utils.accuracy(y_all, labels_all)[0]

        losses.update(loss.item(), x_all.size(0))
        losses_ce.update(loss_ce.item(), x_all.size(0))
        losses_penalty.update(loss_penalty.item(), x_all.size(0))
        cls_accs.update(cls_acc.item(), x_all.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    return losses.avg

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
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
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
            _, predictions = output.max(1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    # gather the stats from all processes

    kappa_score = cohen_kappa_score(all_targets, all_predictions)
    metric_logger.meters["Kappa"] = kappa_score
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}  Kappa {kappa_score:.3f}")
    return metric_logger
    

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)
    
    device = torch.device(args.device)
    print(args)
    
    if args.use_deterministic_algorithms:
        utils.set_seed(args.seed)
    else:
        torch.backends.cudnn.benchmark = True
        
        
    tr_dataset, val_dataset, train_sampler, val_sampler = load_train_val_data_for_coral(
        args.data_path,
        args.train_folders,
        args.interpolation,
        args.val_folder,
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
        args.batch_size,
        args.band_groups,
        args.n_domains_per_batch
    )
    

    num_classes = len(tr_dataset.classes)
    num_channels = len(val_dataset.bands)
    
    collate_fn = default_collate
    
    data_loader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler, num_workers=args.workers, pin_memory=True
    )

    train_iter = utils.ForeverDataIterator(data_loader)

    if args.pretrained_model:
        num_channels = args.num_channels
    
    print("Creating model")
    model = resnet18(weights=args.weights, num_classes=num_classes, num_channels=num_channels)

    if args.pretrained_model:
        checkpoint = torch.load(args.pretrained_model, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.conv1 = utils.copy_weghts(model.conv1, len(val_dataset.bands))
    model.to(device)

    criterion_class = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    
    
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    awl = None
    if args.auto_weighted_loss:
        awl = AutomaticWeightedLoss(2)
        parameters.append( {'params': awl.parameters(), 'weight_decay': args.weight_decay})
    
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
    

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    invariance_penalty_loss = InvariancePenaltyLoss().to(device)
    
    best_kappa = 0.0
    epochs_without_improvement = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        epochs_without_improvement = checkpoint["epochs_without_improvement"]

    print("Start training")
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.max_epochs):
        train_loss = train_one_epoch(model, train_iter, criterion_class, invariance_penalty_loss, optimizer, device, epoch, awl, args)
        lr_scheduler.step()
        val_metric_logger = evaluate(model, criterion_class, data_loader_test, device=device)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "epochs_without_improvement": epochs_without_improvement
        }
        
        train_losses.append(train_loss)
        
        val_loss = val_metric_logger.meters["loss"].global_avg
        val_losses.append(val_loss)
        
        kappa = val_metric_logger.meters["Kappa"]
        
        
        plot_save_path = os.path.join(args.output_dir, "train_val_loss_graph.png")
        
        utils.plot_losses(train_losses, val_losses, plot_save_path)
        
        torch.save(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        if kappa > best_kappa:
            torch.save(checkpoint, os.path.join(args.output_dir, "best_model.pth"))
            best_kappa = kappa
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early Stopping")
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="BigearthNet Classification Training", add_help=add_help)
    
    parser.add_argument("--output-dir", default="./output", type=str, help="path to save outputs")
    
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
    
    parser.add_argument("--data-path", default="data/bigearthnet", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--max-epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
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
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--patience", default=10, type=int, help='number of checks with no improvement after which training will be stopped')
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--train-folders", default=["train"], nargs='+', help="List of train folders")
    parser.add_argument("--val-folder", default="val", help="the validation folder name")
    parser.add_argument("--band-groups", default=["rgb"], nargs='+', help="List of train folders")
    parser.add_argument("--pretrained-model", type=str, default=None, help="the path of the pretrained model")
    parser.add_argument("--seed", default=2342342, type=int, help="The seed value of random")
    parser.add_argument('--trade-off', default=1, type=float, help='the trade off hyper parameter for domain adversarial loss')
    parser.add_argument(
        "--auto-weighted-loss", action="store_true", help="whether the auto weighted loss is used"
    )
    parser.add_argument('--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
    parser.add_argument('--n-domains-per-batch', default=3, type=int, help='number of domains in each mini-batch')
    parser.add_argument('--num-channels', default=4, type=int, help='The number of channels for pretrained model')


    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)