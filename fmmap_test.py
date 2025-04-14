from utils.loss import ComputeLoss
import torch

teacher_pred = torch.randn((32, 3, 80, 80, 85))
out = ComputeLoss.apply_fmnms(teacher_pred, 3)