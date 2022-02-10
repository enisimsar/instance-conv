import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bn=False,
        if_bias=True,
    ):
        super(InstanceConv, self).__init__()
        self.k = kernel_size
        self.in_c = in_channel
        self.out_c = out_channel
        self.stride = stride
        self.padding = padding
        self.bn = bn
        self.if_bias = if_bias

        self.conv = nn.Parameter(torch.Tensor(self.out_c, self.in_c, *(self.k, self.k)))
        self.bias = nn.Parameter(torch.zeros(self.out_c)) if self.if_bias else 0

        # Weights and Bias initialization
        nn.init.kaiming_uniform_(self.conv, a=math.sqrt(5))
        if self.if_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp, mask):
        k = self.k
        stride = self.stride
        h_in, w_in = inp.shape[2], inp.shape[3]

        padding = self.padding
        inp = F.pad(
            input=inp,
            pad=[padding, padding, padding, padding],
            mode="constant",
            value=-1,
        )
        mask = F.pad(
            input=mask,
            pad=[padding, padding, padding, padding],
            mode="constant",
            value=-1,
        )

        batch_size = inp.shape[0]

        mask_unfolded = torch.nn.functional.unfold(mask, (k, k), stride=stride)
        mask_patched = mask_unfolded.view(
            batch_size, k, k, -1
        )  # [B, Kernel, Kernel, patches]

        inp_unfolded = torch.nn.functional.unfold(inp, (k, k), stride=stride)
        inp_patched = inp_unfolded.view(
            batch_size, self.in_c, k, k, -1
        )  # [B, C, K, K, patches]

        center = k // 2
        mask_center_equals = (
            mask_patched == mask_patched[:, center : center + 1, center : center + 1, :]
        )

        mask_center_equals = mask_center_equals.unsqueeze(
            1
        ).float()  # unsqueeze for 1 channel
        masked_input = (
            inp_patched * mask_center_equals
        )  # mask the patch pixels where pix is not equal to center [B, C, K, K, patches]

        # calculate m norm
        m_norm = mask_center_equals.view(
            batch_size, k * k, -1
        )  # [batch, kernel*kernel, patches]
        m_norm = (
            k * k / (torch.sum(m_norm, dim=1) + 1e-5)
        )  # sum over kernels, get normalization factor

        out_unfolded = torch.einsum(
            "ijklm,zjkl->izm", (masked_input, self.conv)
        )  # [B, C_out, patch_size]

        m_norm = m_norm.unsqueeze(1)
        out_unfolded_normalized = out_unfolded * m_norm

        mask_pooled = mask_patched[:, center, center, :]

        h_out = (h_in + 2 * padding - (k - 1) - 1) / stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) / stride + 1
        h_out, w_out = int(h_out), int(w_out)

        out_ = out_unfolded_normalized.view(
            batch_size, self.out_c, h_out, w_out
        )  # [b, c, h, w] back to expected size
        mask_out = mask_pooled.view(
            batch_size, 1, h_out, w_out
        )  # [b, 1, h, w] back to expected size

        if self.if_bias:
            out_ = out_ + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(out_)
        if self.bn:
            out_ = self.bn1(out_)

        return out_, mask_out


class InstanceDeconv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bn=False,
    ):
        super(InstanceDeconv, self).__init__()
        self.bn = bn
        self.sparse_conv = InstanceConv(
            in_channel, out_channel, kernel_size, stride, padding, bn=bn
        )
        self.up_sample = torch.nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x_guide, mask):
        x = self.up_sample(x_guide)
        # m = self.up_sample(mask)
        x, m = self.sparse_conv(x, mask)

        return x, m


class CenterPool(nn.Module):
    def __init__(self, kernel_size=1, stride=2, padding=0):
        super(CenterPool, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, mask):
        k = self.k
        stride = self.stride
        h_in, w_in = mask.shape[2], mask.shape[3]

        padding = self.padding  # + k//2
        mask = F.pad(
            input=mask,
            pad=[padding, padding, padding, padding],
            mode="constant",
            value=-1,
        )

        batch_size = mask.shape[0]

        mask_unfolded = torch.nn.functional.unfold(mask, (k, k), stride=stride)
        mask_patched = mask_unfolded.view(
            batch_size, k, k, -1
        )  # [B, patches, Kernel, Kernel]

        center = k // 2
        mask_pooled = mask_patched[:, center, center, :]

        h_out = (h_in + 2 * padding - (k - 1) - 1) / stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) / stride + 1
        h_out, w_out = int(h_out), int(w_out)

        mask_out = mask_pooled.view(
            batch_size, mask.size(1), h_out, w_out
        )  # [b, 1, h, w] back to expected size

        return mask_out
