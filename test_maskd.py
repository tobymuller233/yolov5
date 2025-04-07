from models.maskd import *
import torch
import torch.nn as nn

# a simple nn.module
class SimpleModule(nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 9, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(9, 12, kernel_size=3, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

teacher_output = [torch.randn(1, 3 * i, 120 * i, 120 * i) for i in range(1, 5)]

teacher_model = SimpleModule()
for n, p in teacher_model.named_parameters():
    p.requires_grad = False
for n, m in teacher_model.named_modules():
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()
input_tensor = torch.randn(1, 3, 120, 120)
teacher_model(input_tensor)
teacher_model(input_tensor)
maskmodules = MaskModules([3, 6, 9, 12], 6, True)
maskloss = Mask_Loss(teacher_model, {"maskd_channels": [3, 6, 9, 12], "maskd_ntokens": 6, "maskd_weightmask": True}, maskmodules=maskmodules)
# maskloss.teacher_outputs = teacher_output
maskloss.register_hook()
teacher_model(input_tensor)
maskloss.reset_loss()
teacher_model(input_tensor)
loss = maskloss.get_loss()
