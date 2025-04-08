import torch
import torch.nn as nn
from models.maskd import Distillation_Loss, MasKDLoss, MaskModule, MaskModules, Mask_Loss
import yaml
from utils.plots import feature_visualization

hyp = yaml.safe_load(open('data/maskd/yolov5n_visdrone_free_maskd.yaml', 'r'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mask_weight = "weights/maskmodule_yolov5l_free_visdrone.pt"
teacher_weight = "runs/thesis/yolo5n_free_visdrone/exp/weights/best.pt"
student_weight = "runs/thesis/yolo5n_free_visdrone/exp3/weights/best.pt"
tea_model = torch.load(teacher_weight, map_location='cpu')['model'].float().fuse().eval().to(device)
student_model = torch.load(student_weight, map_location='cpu')['model'].float().fuse().eval().to(device)
channels_s = [128, 64, 128, 256]
channels_t = [512, 256, 512, 1024]
# mask_hook = MasKDLoss(channels_s=channels_s, channels_t=channels_t, weight_mask=True, pretrained=mask_weight)
maskmodules = MaskModules(hyp["maskd_tea_channels"], pretrained=mask_weight).to(device)
mask_hook = Mask_Loss(tea_model, hyp, maskmodules, device=device)
mask_hook.register_hook()
