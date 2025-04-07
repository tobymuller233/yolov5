from models.yolo import Detect
import torch
import torch.nn as nn
import torch_pruning as tp
import argparse
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, help="Path to the model weights")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--ratio", type=int, default=0.5, help="Pruning ratio")
    parser.add_argument("--device", default="0", help="Device to run the model on")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    weights = args.weights
    img_size = args.img_size
    ratio = args.ratio

    device = select_device(args.device)
    # model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    old_model = torch.load(weights, map_location=device)  # load FP32 model
    model = old_model['model'].float().fuse().eval()  # FP16 model 
    for p in model.parameters():
        p.requires_grad_(True)

    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    imp = tp.importance.MagnitudeImportance(p=2) # L2 norm pruning

    ignored_layers = []
    for m in model.model.modules():
        if isinstance(m, Detect):
            ignored_layers.append(m)
    print(ignored_layers)

    iterative_steps = 1 # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for g in pruner.step(interactive=True):
        print(g)
        g.prune()

    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(model)
    print("Before Pruning: MACs=%f G, #Params=%f G"%(base_macs/1e9, base_nparams/1e9))
    print("After Pruning: MACs=%f G, #Params=%f G"%(pruned_macs/1e9, pruned_nparams/1e9))

    # save pruned model
    prune_ratio = 0.5
    old_model['model'] = model
    torch.save(old_model, f"weights/yolov5s_pruned_{prune_ratio}.pt")