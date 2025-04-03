from models.maskd import *
import torch
import torch.nn as nn

teacher_output = [torch.randn(1, 3 * i, 120 * i, 120 * i) for i in range(1, 5)]

maskmodules = MaskModules([3, 6, 9, 12], 6, True)
maskloss = Mask_Loss(None, {"maskd_channels": [3, 6, 9, 12], "maskd_ntokens": 6, "maskd_weightmask": True}, maskmodules=maskmodules)
maskloss.teacher_outputs = teacher_output
loss = maskloss.get_loss()