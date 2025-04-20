from models.yolo import Detect, DetectNoAnchor
import torch
import torch.nn as nn
import torch_pruning as tp
import argparse
from pathlib import Path
import sys
import os
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, check_dataset, labels_to_class_weights)
from utils.torch_utils import select_device, smart_inference_mode, torch_distributed_zero_first
from utils.dataloaders import create_dataloader
from utils.loss import ComputeLoss, v8DetectionLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="Path to the model weights")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--ratio", type=float, default=0.5, help="Pruning ratio")
    parser.add_argument("--device", default="0", help="Device to run the model on")
    parser.add_argument("--save-name", default="pruned_model", help="Name of the saved pruned model")
    parser.add_argument("--project", default="runs/prune", help="Project name")
    parser.add_argument("--name", default="exp", help="Experiment name")

    # data is used to do Hessian Taylor importance calculation
    parser.add_argument("--all-imp", action="store_true", help="Use all importance methods")
    parser.add_argument("--data", type=str, default="data/SCUT_HEAD_A_B.yaml", help="Path to the dataset yaml file")
    parser.add_argument("--anchorfree", action="store_true", help="Use anchor-free model")
    parser.add_argument("--hyp", type=str, default="data/hyps/hyp.scratch-low.yaml", help="Path to the hyperparameters yaml file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--num-batch", type=int, default=10, help="Number of batches to use for importance calculation")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    weights = args.weights
    img_size = args.img_size
    ratio = args.ratio
    save_dir = str(increment_path(Path(args.project) / args.name, exist_ok=False, mkdir=True))
    log_file = os.path.join(save_dir, 'prune.log')

    device = select_device(args.device)
    # model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    old_model = torch.load(weights, map_location=device)  # load FP32 model
    model = old_model['model'].float().fuse().eval()  # FP16 model 
    for p in model.parameters():
        p.requires_grad_(True)

    example_inputs = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    if args.all_imp:
        imp_dict = {
            'Random': tp.importance.RandomImportance(),
            'Hessian': tp.importance.HessianImportance(group_reduction='first'),
            'Taylor': tp.importance.TaylorImportance(group_reduction='first'),     
            'L1': tp.importance.MagnitudeImportance(p=1, group_reduction='first'),
            'L2': tp.importance.MagnitudeImportance(p=2, group_reduction="first"),   
        }
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = check_dataset(args.data)
        train_path = data_dict['train']
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        hyp = check_file(args.hyp) if args.hyp else None  # hyperparameters
        with open(hyp) as f:
            import yaml
            hyp = yaml.safe_load(f)
        train_loader, train_dataset = create_dataloader(
            train_path,
            imgsz=img_size,
            batch_size=args.batch_size,
            stride=gs,
            single_cls=False,
            hyp=hyp,
            augment=True,
            cache=False,
            rect=False,
            rank=LOCAL_RANK,
            workers=8,
            image_weights=False,
            quad=False,
            prefix=colorstr("train: "),
            shuffle=True,
            seed=42,
            v8loader=args.anchorfree
        )
        nl = model.model[-1].nl
        nc = len(data_dict['names'])  # number of classes
        if args.anchorfree:
            hyp["box"] *= 3 / nl
            hyp["cls"] *= nc / 80 * 3 / nl
            hyp["obj"] *= (img_size / 640) ** 2 * 3 / nl
        hyp["label_smoothing"] = False
        model.nc = nc
        model.hyp = hyp
        model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device) * nc
        model.names = data_dict['names']
        if args.anchorfree:
            model.args = model.hyp
            compute_loss = v8DetectionLoss(model)
        else:
            compute_loss = ComputeLoss(model)

    else:
        imp_dict = {'L2': tp.importance.MagnitudeImportance(p=2, group_reduction="first")}
    # imp = tp.importance.MagnitudeImportance(p=2) # L2 norm pruning

    # ignored_layers = []
    # for m in model.model.modules():
    #     if isinstance(m, (Detect, DetectNoAnchor)):
    #         ignored_layers.append(m)
    # print(ignored_layers)

    iterative_steps = 16 # progressive pruning
    
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print("Before Pruning: MACs=%f G, #Params=%f G"%(base_macs/1e9, base_nparams/1e9))

    ori_model = copy.deepcopy(model)
    for imp_name, imp in imp_dict.items():
        model = copy.deepcopy(ori_model)
        print(f"Now Pruning with {imp_name} importance")
        ignored_layers = []
        for m in model.model.modules():
            if isinstance(m, (Detect, DetectNoAnchor)):
                ignored_layers.append(m)
        print(ignored_layers)
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=ratio, 
            ignored_layers=ignored_layers,
        )
        for i in range(iterative_steps):
            print(f"Pruning step {i+1}/{iterative_steps} with {imp_name} importance and {ratio * 100}% pruning ratio:"  )
            
            if isinstance(imp, tp.importance.HessianImportance):
                # loss = F.cross_entropy(model(images), targets)
                model.train()
                for k, (imgs, targets, paths, _) in enumerate(train_loader):
                    if k>=args.num_batch: break
                    
                    train_batch = imgs
                    if targets is not None:
                        imgs = imgs.to(device, non_blocking=True).float() / 255.0
                    
                    if targets is not None:
                        pred = model(imgs)
                        loss, loss_items = compute_loss(pred, targets.to(device))
                    else:
                        train_batch["img"] = train_batch["img"].to(device, non_blocking=True).float() / 255.0
                        pred = model(train_batch["img"])
                        loss, loss_items = compute_loss(pred, train_batch)
                    imp.zero_grad() # clear accumulated gradients
                    for l in loss:
                        model.zero_grad() # clear gradients
                        l.backward(retain_graph=True) # simgle-sample gradient
                        imp.accumulate_grad(model) # accumulate g^2
            elif isinstance(imp, tp.importance.TaylorImportance):
                # loss = F.cross_entropy(model(images), targets)
                model.train()
                for k, (imgs, targets, paths, _) in enumerate(train_loader):
                    if k>=args.num_batch: break
                    train_batch = imgs
                    if targets is not None:
                        imgs = imgs.to(device, non_blocking=True).float() / 255.0
                    
                    if targets is not None:
                        pred = model(imgs)
                        loss, loss_items = compute_loss(pred, targets.to(device))
                    else:
                        train_batch["img"] = train_batch["img"].to(device, non_blocking=True).float() / 255.0
                        pred = model(train_batch["img"])
                        loss, loss_items = compute_loss(pred, train_batch)
                    loss.backward()
            for g in pruner.step(interactive=True):
                # print(g)
                g.prune()

            pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
            current_speed_up = float(base_macs) / float(pruned_macs)
            # print(model)
            # print("After Pruning: MACs=%f G, #Params=%f G"%(pruned_macs/1e9, pruned_nparams/1e9))
            print(f"{imp_name}--After pruning for {i + 1} steps: MACs={pruned_macs / 1e9:.2f} G, #Params={pruned_nparams / 1e6:.2f} M",
                f" Speedup={current_speed_up:.2f}x")
            # logging
            with open(log_file, 'a') as f:
                f.write(f"{imp_name}--After pruning for {i + 1} steps: MACs={pruned_macs / 1e9:.2f} G, #Params={pruned_nparams / 1e6:.2f} M Speedup={current_speed_up:.2f}x\n")
        

        # save pruned model
        prune_ratio = ratio
        # print(model)
        old_model['model'] = model
        # torch.save(old_model, f"weights/{weight_name}_pruned_{prune_ratio}.pt")
        torch.save(old_model, os.path.join(save_dir, f"{args.save_name}_{imp_name}_{prune_ratio}.pt"))