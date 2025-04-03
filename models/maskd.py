import math
import torch
import torch.nn as nn


def dice_coeff(inputs, eps=1e-12):
    # inputs: [B, T, H*W]
    pred = inputs[:, None, :, :]
    target = inputs[:, :, None, :]

    mask = pred.new_ones(pred.size(0), target.size(1), pred.size(2))
    mask[:, torch.arange(mask.size(1)), torch.arange(mask.size(2))] = 0

    a = torch.sum(pred * target, -1)
    b = torch.sum(pred * pred, -1) + eps
    c = torch.sum(target * target, -1) + eps
    d = (2 * a) / (b + c)
    d = (d * mask).sum() / mask.sum()
    return d


class MaskModule(nn.Module):

    def __init__(self, channels, num_tokens=6, weight_mask=True):
        super().__init__()
        self.weight_mask = weight_mask
        self.mask_token = nn.Parameter(torch.randn(num_tokens, channels).normal_(0, 0.01))
        if self.weight_mask:
            self.prob = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, num_tokens, kernel_size=1)
            )
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_mask(self, x):
        N, C, H, W = x.shape
        mask_token = self.mask_token.expand(N, -1, -1)  # [N, T, C]
        k = x.view(N, -1, H * W)
        attn = mask_token @ k  # [N, T, H * W]
        attn = attn.sigmoid()
        attn = attn.view(N, -1, H, W)   # [N, T, H, W]
        return attn

    def forward_prob(self, x):
        mask_probs = self.prob(x)  # [N, T, 1, 1]
        mask_probs = mask_probs.softmax(1).unsqueeze(2)  # [N, T, 1, 1, 1]
        return mask_probs

    def forward_train(self, x):
        mask = self.forward_mask(x)
        out = x.unsqueeze(1) * mask.unsqueeze(2)  # [N, T, C, H, W]
        # probs
        if self.weight_mask:
            mask_probs = self.forward_prob(x)
            out = out * mask_probs
        out = out.sum(1)
        # loss
        mask_loss = dice_coeff(mask.flatten(2))
        return out, mask_loss

    def forward(self, x):
        return self.forward_train(x)

class MasKDLoss(nn.Module):

    def __init__(self, channels, num_tokens=6, weight_mask=True, custom_mask=True, custom_mask_warmup=1000, pretrained=None, loss_weight=1.):
        super().__init__()
        self.loss_weight = loss_weight
        self.weight_mask = weight_mask
        self.custom_mask = custom_mask
        self.custom_mask_warmup = custom_mask_warmup

        self.mask_modules = nn.ModuleList([
            MaskModule(channels=c, num_tokens=num_tokens, weight_mask=weight_mask) for c in channels]
        )

        self.init_weights(pretrained)
        self._iter = 0

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return
        # ckpt = _load_checkpoint(pretrained, map_location='cpu')
        ckpt = torch.load(pretrained, map_location='cpu')
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if 'mask_modules' in k:
                state_dict[k[k.find('mask_modules'):]] = v
        self.load_state_dict(state_dict, strict=True)

    def forward(self, y_s_list, y_t_list):
        if not isinstance(y_s_list, (tuple, list)):
            y_s_list = (y_s_list, )
            y_t_list = (y_t_list, )
        assert len(y_s_list) == len(y_t_list) == len(self.mask_modules)

        losses = []
        for y_s, y_t, mask_module in zip(y_s_list, y_t_list, self.mask_modules):
            # predict the masks
            mask = mask_module.forward_mask(y_t)
            if self.custom_mask and self._iter >= self.custom_mask_warmup:
                if self._iter == self.custom_mask_warmup:
                    print('Start customizing masks using student\'s masks.')
                with torch.no_grad():
                    mask_s = mask_module.forward_mask(y_s)  # [N, T, H, W]
                mask = mask * mask_s

            # get the masked features
            masked_y_s = y_s.unsqueeze(1) * \
                mask.unsqueeze(2)  # [N, n_masks, C, H, W]
            masked_y_t = y_t.unsqueeze(1) * \
                mask.unsqueeze(2)  # [N, n_masks, C, H, W]

            # masked distillation
            loss = (masked_y_s - masked_y_t)**2
            loss = loss.sum((3, 4))  # [N, n_masks, C]
            loss = loss / mask.sum((2, 3)).unsqueeze(-1)
            if self.weight_mask:
                weights = mask_module.forward_prob(y_t).flatten(1)  # [N, T]
                loss = (loss.mean(-1) * weights).sum(-1)
            loss = loss.mean()
            losses.append(loss)

        loss = sum(losses)
        self._iter += 1
        return self.loss_weight * loss

class MaskModules(nn.Module):
    def __init__(self, channels=[], num_tokens=6, weight_mask=True):
        super().__init__()
        self.weight_mask = weight_mask
        self.mask_modules = nn.ModuleList([
            MaskModule(channels=c, num_tokens=num_tokens, weight_mask=weight_mask) for c in channels]
        )

    def forward(self, x):
        out = []
        assert len(x) == len(self.mask_modules)
        for mask_module, xi in zip(self.mask_modules, x):
            out.append(mask_module(xi))
        return out

class Mask_Loss:
    def __init__(self, teacher_model, hyp, maskmodules=MaskModules()):  # model must be de-paralleled

        self.teacher_module_pairs = []
        self.remove_handle = []

        self.channels = hyp["maskd_channels"]
        self.num_tokens = hyp["maskd_ntokens"]
        self.weight_mask = hyp["maskd_weightmask"]

        # self.mask_modules = nn.ModuleList([MaskModule(c, self.num_tokens, self.weight_mask)
        #                                        for c in self.channels])
        self.mask_modules = maskmodules

    def register_hook(self):
        self.teacher_outputs = []

        def make_layer_forward_hook(l):
            def forward_hook(m, input, output):
                l.append(output)

            return forward_hook

        for mt in self.teacher_module_pairs:
            self.remove_handle.append(mt.register_forward_hook(make_layer_forward_hook(self.teacher_outputs)))

    def get_loss(self):
        assert len(self.teacher_outputs) == len(self.mask_modules.mask_modules) # ensure the same number of layers

        div_losses = 0
        mask_out = self.mask_modules(self.teacher_outputs)
        for masked_out, div_loss in mask_out:
            div_losses += div_loss
        self.teacher_outputs.clear()
        return div_losses

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()