
# See ./LICENSE_AND_AUTHORS for info about the authors and licensing this file.

import math
import numpy as np
import torch
import torch.nn
import torch.nn.functional
import torch.nn.init
import packaging
import packaging.version
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Set

from torch.amp import autocast

from ..train import modelconfigs

EXTRA_SCORE_DISTR_RADIUS = 60

def enumerate_tensor(tensor):
    """Iterate over (idx_tuple, value) for all values in a tensor"""
    from itertools import product

    shape = tensor.shape
    indices = [range(s) for s in shape]
    for idx in product(*indices):
        value = tensor[idx]
        yield idx, value

def debug_print_tensor(tensor):
    torch.set_printoptions(threshold=10000000,sci_mode=False)
    print(tensor)
    total1 = 0
    total2 = 0
    total3 = 0
    for (nn,cc,hh,ww), value in enumerate_tensor(tensor):
        total1 += (((cc + hh // 2 + ww // 3 + nn // 4) % 2)*2-1) * value
        total2 += (((cc + hh // 3 + ww // 1 + nn // 3) % 2)*2-1) * value
        total3 += (((cc + hh // 5 + ww // 2 + nn // 2) % 2)*2-1) * value
    print(f"TOTAL {out.shape} {total1} {total2} {total3}")


class ExtraOutputs:
    def __init__(self, requested: List[str]):
        self.requested: Set[str] = set(requested)
        self.available: List[str] = []
        self.returned: Dict[str,torch.Tensor] = {}

    def add_requested(self, requested: List[str]):
        self.requested = self.requested.union(set(requested))

    def report(self, name: str, value: torch.Tensor):
        self.available.append(name)
        if name in self.requested:
            self.returned[name] = value.detach()

def act(activation, inplace=False):
    if activation == "relu":
        return torch.nn.ReLU(inplace=inplace)
    if activation == "elu":
        return torch.nn.ELU(inplace=inplace)
    if activation == "mish":
        return torch.nn.Mish(inplace=inplace)
    if activation == "silu":
        return torch.nn.SiLU(inplace=inplace)
    if activation == "gelu":
        return torch.nn.GELU()
    if activation == "hardswish":
        if packaging.version.parse(torch.__version__) > packaging.version.parse("1.6.0"):
            return torch.nn.Hardswish(inplace=inplace)
        else:
            return torch.nn.Hardswish()
    if activation == "identity":
        return torch.nn.Identity()
    assert False, f"Unknown activation name: {activation}"

def compute_gain(activation):
    if activation == "relu" or activation == "hardswish":
        gain = math.sqrt(2.0)
    elif activation == "elu":
        gain = math.sqrt(1.55052)
    elif activation == "mish":
        gain = math.sqrt(2.210277)
    elif activation == "silu":
        gain = math.sqrt(2.0)  # Theoretically should be sqrt(2.8108), kept sqrt(2.0) for compat reasons.
    elif activation == "gelu":
        gain = math.sqrt(2.351718)
    elif activation == "identity":
        gain = 1.0
    else:
        assert False, f"Unknown activation name: {activation}"
    return gain

def init_weights(tensor, activation, scale, fan_tensor=None):
    gain = compute_gain(activation)

    if fan_tensor is not None:
        (fan_in, _) = torch.nn.init._calculate_fan_in_and_fan_out(fan_tensor)
    else:
        (fan_in, _) = torch.nn.init._calculate_fan_in_and_fan_out(tensor)

    target_std = scale * gain / math.sqrt(fan_in)
    # Multiply slightly since we use truncated normal
    std = target_std / 0.87962566103423978
    if std < 1e-10:
        tensor.fill_(0.0)
    else:
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0*std, b=2.0*std)

class SoftPlusWithGradientFloorFunction(torch.autograd.Function):
    """
    Same as softplus, except on backward pass, we never let the gradient decrease below grad_floor.
    Equivalent to having a dynamic learning rate depending on stop_grad(x) where x is the input.
    If square, then also squares the result while halving the input, and still also keeping the same gradient.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, grad_floor: float, square: bool):
        ctx.save_for_backward(x)
        ctx.grad_floor = grad_floor # grad_floor is not a tensor
        if square:
            return torch.square(torch.nn.functional.softplus(0.5 * x))
        else:
            return torch.nn.functional.softplus(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_floor = ctx.grad_floor
        grad_x = None
        grad_grad_floor = None
        grad_square = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (grad_floor + (1.0 - grad_floor) / (1.0 + torch.exp(-x)))
        return grad_x, grad_grad_floor, grad_square

class BiasMask(torch.nn.Module):
    def __init__(
        self,
        c_in,
        config: modelconfigs.ModelConfig,
        is_after_batchnorm: bool = False,
    ):
        super(BiasMask, self).__init__()
        self.c_in = c_in
        self.beta = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.is_after_batchnorm = is_after_batchnorm
        self.scale = None

    def set_scale(self, scale: Optional[float]):
        self.scale = scale

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        if self.is_after_batchnorm:
            reg_dict["output_noreg"].append(self.beta)
        else:
            reg_dict["noreg"].append(self.beta)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass

    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        if self.scale is not None:
            return (x * self.scale + self.beta) * mask
        else:
            return (x + self.beta) * mask


class RMSNormMask(torch.nn.Module):
    """RMSNorm applied per spatial position across channels, with masking for off-board positions.
    If spatial=True, computes RMS across both channels and spatial positions (masked), producing
    one scalar RMS per sample instead of per position.
    If spatial=True and cgroup_size is not None, breaks channels into groups of the given size
    and normalizes within each group across channels_in_group x H x W (like group norm but RMS only,
    no mean centering).
    """
    def __init__(self, c_in, config: modelconfigs.ModelConfig, spatial: bool, cgroup_size: Optional[int]):
        super(RMSNormMask, self).__init__()
        self.c_in = c_in
        self.spatial = spatial
        self.cgroup_size = cgroup_size
        self.eps = 1e-6
        if cgroup_size is not None:
            assert spatial, "cgroup_size requires spatial=True"
            assert c_in % cgroup_size == 0, f"c_in ({c_in}) must be divisible by cgroup_size ({cgroup_size})"
            self.num_groups = c_in // cgroup_size
        if not spatial:
            self.norm = torch.nn.RMSNorm(c_in, eps=self.eps)
        else:
            self.norm = None
            self.gamma = torch.nn.Parameter(torch.ones(c_in))
        self.beta = torch.nn.Parameter(torch.zeros(c_in))

    def set_scale(self, scale: Optional[float]):
        pass  # RMSNorm normalizes by actual magnitude, external fixup scale not needed

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        if self.norm is not None:
            reg_dict["output"].append(self.norm.weight)
        else:
            reg_dict["output"].append(self.gamma)
        reg_dict["output"].append(self.beta)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass

    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        if not self.spatial:
            # NCHW -> NHWC for RMSNorm across channels, then back
            out = x.permute(0, 2, 3, 1)
            out = self.norm(out)
            out = out.permute(0, 3, 1, 2)
            return (out + self.beta.view(1, -1, 1, 1)) * mask
        else:
            if self.cgroup_size is not None:
                # Group-wise spatial RMS: normalize within each group of channels across group_channels x H x W
                N, C, H, W = x.shape
                x_grouped = x.view(N, self.num_groups, self.cgroup_size, H, W)
                mask_grouped = mask.view(N, 1, 1, H, W)
                # mean of x^2 over group channels and masked spatial positions
                mean_sq = torch.sum(x_grouped * x_grouped * mask_grouped, dim=(2, 3, 4), keepdim=True) / (self.cgroup_size * mask_sum_hw.unsqueeze(2) + self.eps)
                rms = torch.sqrt(mean_sq + self.eps)
                out = x_grouped / rms
                out = out.view(N, C, H, W)
            else:
                # RMS across C,H,W for masked positions only, one scalar per sample
                # mean of x^2 over C and masked spatial positions
                mean_sq = torch.sum(x * x * mask, dim=(1, 2, 3), keepdim=True) / (self.c_in * mask_sum_hw + self.eps)
                rms = torch.sqrt(mean_sq + self.eps)
                out = x / rms
            return (out * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)) * mask


class NormMask(torch.nn.Module):
    def __init__(
        self,
        c_in,
        config: modelconfigs.ModelConfig,
        fixup_use_gamma: bool,
        force_use_gamma: bool = False,
        is_last_batchnorm: bool = False,
    ):
        """Various kinds of normalization.

        bnorm - batch norm
        brenorm - batch renorm
        fixup - fixup initialization https://arxiv.org/abs/1901.09321
        fixscale - fixed scaling initialization. Normalization layers simply multiply a constant scalar according
          to what batchnorm *would* do if all inputs were unit variance and all linear layers or convolutions
          preserved variance.
        fixbrenorm - fixed scaling normalization PLUS batch renorm.
        fixscaleonenorm - fixed scaling normalization PLUS only have one batch norm layer in the entire net, at the end of the residual trunk.
        """

        super(NormMask, self).__init__()
        self.norm_kind = config["norm_kind"]
        self.epsilon = config["bnorm_epsilon"]
        self.running_avg_momentum = config["bnorm_running_avg_momentum"]
        self.fixup_use_gamma = fixup_use_gamma
        self.is_last_batchnorm = is_last_batchnorm
        self.gamma_weight_decay_center_1 = config.get("gamma_weight_decay_center_1",False)
        self.use_gamma = (
            ("bnorm_use_gamma" in config and config["bnorm_use_gamma"]) or
            ((self.norm_kind == "fixup" or self.norm_kind == "fixscale" or self.norm_kind == "fixscaleonenorm") and fixup_use_gamma) or
            force_use_gamma
        )
        self.c_in = c_in

        self.scale = None
        self.gamma = None
        if self.norm_kind == "bnorm" or (self.norm_kind == "fixscaleonenorm" and self.is_last_batchnorm):
            self.is_using_batchnorm = True
            if self.use_gamma:
                if self.gamma_weight_decay_center_1:
                    self.gamma = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
                else:
                    self.gamma = torch.nn.Parameter(torch.ones(1, c_in, 1, 1))
            self.beta = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
            self.register_buffer(
                "running_mean", torch.zeros(c_in, dtype=torch.float)
            )
            self.register_buffer(
                "running_std", torch.ones(c_in, dtype=torch.float)
            )
        elif self.norm_kind == "brenorm" or self.norm_kind == "fixbrenorm":
            self.is_using_batchnorm = True
            if self.use_gamma:
                if self.gamma_weight_decay_center_1:
                    self.gamma = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
                else:
                    self.gamma = torch.nn.Parameter(torch.ones(1, c_in, 1, 1))
            self.beta = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
            self.register_buffer(
                "running_mean", torch.zeros(c_in, dtype=torch.float)
            )
            self.register_buffer(
                "running_std", torch.ones(c_in, dtype=torch.float)
            )
            self.register_buffer(
                "renorm_running_mean", torch.zeros(c_in, dtype=torch.float)
            )
            self.register_buffer(
                "renorm_running_std", torch.ones(c_in, dtype=torch.float)
            )
            self.register_buffer(
                "renorm_upper_rclippage", torch.zeros((), dtype=torch.float)
            )
            self.register_buffer(
                "renorm_lower_rclippage", torch.zeros((), dtype=torch.float)
            )
            self.register_buffer(
                "renorm_dclippage", torch.zeros((), dtype=torch.float)
            )

        elif self.norm_kind == "fixup" or self.norm_kind == "fixscale" or (self.norm_kind == "fixscaleonenorm" and not self.is_last_batchnorm):
            self.is_using_batchnorm = False
            self.beta = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
            if self.use_gamma:
                if self.gamma_weight_decay_center_1:
                    self.gamma = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
                else:
                    self.gamma = torch.nn.Parameter(torch.ones(1, c_in, 1, 1))
        else:
            assert False, f"Unimplemented norm_kind: {self.norm_kind}"

    def set_scale(self, scale: Optional[float]):
        self.scale = scale

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        if self.is_last_batchnorm:
            if self.gamma is not None:
                reg_dict["output"].append(self.gamma)
            reg_dict["output_noreg"].append(self.beta)
        else:
            if self.gamma is not None:
                reg_dict["normal_gamma"].append(self.gamma)
            reg_dict["noreg"].append(self.beta)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        self.renorm_avg_momentum = renorm_avg_momentum
        self.rmax = rmax
        self.dmax = dmax

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        upper_rclippage.append(self.renorm_upper_rclippage.cpu().item())
        lower_rclippage.append(self.renorm_lower_rclippage.cpu().item())
        dclippage.append(self.renorm_dclippage.cpu().item())

    def _compute_bnorm_values(self, x, mask, mask_sum: float):
        # This is the mean, computed only over exactly the areas of the mask, weighting each spot equally,
        # even across different elements in the batch that might have different board sizes.
        mean = torch.sum(x * mask, dim=(0,2,3),keepdim=True) / mask_sum
        zeromean_x = x - mean
        # Similarly, the variance computed exactly only over those spots
        var = torch.sum(torch.square(zeromean_x * mask),dim=(0,2,3),keepdim=True) / mask_sum
        std = torch.sqrt(var + self.epsilon)
        return zeromean_x, mean, std

    def apply_gamma_beta_scale_mask(self, x, mask):
        if self.scale is not None:
            if self.gamma is not None:
                if self.gamma_weight_decay_center_1:
                    return (x * ((self.gamma+1.0) * self.scale) + self.beta) * mask
                else:
                    return (x * (self.gamma * self.scale) + self.beta) * mask
            else:
                return (x * self.scale + self.beta) * mask
        else:
            if self.gamma is not None:
                if self.gamma_weight_decay_center_1:
                    return (x * (self.gamma+1.0) + self.beta) * mask
                else:
                    return (x * self.gamma + self.beta) * mask
            else:
                return (x + self.beta) * mask


    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """

        if self.norm_kind == "bnorm" or (self.norm_kind == "fixscaleonenorm" and self.is_last_batchnorm):
            assert x.shape[1] == self.c_in
            if self.training:
                zeromean_x, mean, std = self._compute_bnorm_values(x, mask, mask_sum)

                detached_mean = mean.view(self.c_in).detach()
                detached_std = std.view(self.c_in).detach()
                with torch.no_grad():
                    self.running_mean += self.running_avg_momentum * (detached_mean - self.running_mean)
                    self.running_std += self.running_avg_momentum * (detached_std - self.running_std)

                return self.apply_gamma_beta_scale_mask(zeromean_x / std, mask)
            else:
                return self.apply_gamma_beta_scale_mask((x - self.running_mean.view(1,self.c_in,1,1)) / self.running_std.view(1,self.c_in,1,1), mask)

        elif self.norm_kind == "brenorm" or self.norm_kind == "fixbrenorm":
            assert x.shape[1] == self.c_in
            if self.training:
                zeromean_x, mean, std = self._compute_bnorm_values(x, mask, mask_sum)

                detached_mean = mean.view(self.c_in).detach()
                detached_std = std.view(self.c_in).detach()
                with torch.no_grad():
                    unclipped_r = detached_std / self.renorm_running_std
                    unclipped_d = (detached_mean - self.renorm_running_mean) / self.renorm_running_std
                    r = unclipped_r.clamp(1.0 / self.rmax, self.rmax)
                    d = unclipped_d.clamp(-self.dmax, self.dmax)

                    self.renorm_running_mean += self.renorm_avg_momentum * (detached_mean - self.renorm_running_mean)
                    self.renorm_running_std += self.renorm_avg_momentum * (detached_std - self.renorm_running_std)
                    self.running_mean += self.running_avg_momentum * (detached_mean - self.running_mean)
                    self.running_std += self.running_avg_momentum * (detached_std - self.running_std)

                    upper_rclippage = torch.mean(torch.nn.functional.relu(torch.log(unclipped_r / r)))
                    lower_rclippage = torch.mean(torch.nn.functional.relu(-torch.log(unclipped_r / r)))
                    dclippage = torch.mean(torch.abs(unclipped_d - d))
                    self.renorm_upper_rclippage += 0.01 * (upper_rclippage - self.renorm_upper_rclippage)
                    self.renorm_lower_rclippage += 0.01 * (lower_rclippage - self.renorm_lower_rclippage)
                    self.renorm_dclippage += 0.01 * (dclippage - self.renorm_dclippage)

                if self.rmax > 1.00000001 or self.dmax > 0.00000001:
                    return self.apply_gamma_beta_scale_mask(zeromean_x / std * r.detach().view(1,self.c_in,1,1) + d.detach().view(1,self.c_in,1,1), mask)
                else:
                    return self.apply_gamma_beta_scale_mask(zeromean_x / std, mask)

            else:
                return self.apply_gamma_beta_scale_mask((x - self.running_mean.view(1,self.c_in,1,1)) / self.running_std.view(1,self.c_in,1,1), mask)

        elif self.norm_kind == "fixup" or self.norm_kind == "fixscale" or (self.norm_kind == "fixscaleonenorm" and not self.is_last_batchnorm):
            return self.apply_gamma_beta_scale_mask(x, mask)

        else:
            assert False


class KataGPool(torch.nn.Module):
    def __init__(self):
        super(KataGPool, self).__init__()

    def forward(self, x, mask, mask_sum_hw):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111

        Returns: NC11
        """
        mask_sum_hw_sqrt_offset = torch.sqrt(mask_sum_hw) - 14.0

        layer_mean = torch.sum(x, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        # All activation functions we use right now are always greater than -1.0, and map 0 -> 0.
        # So off-board areas will equal 0, and then this max is mask-safe if we assign -1.0 to off-board areas.
        (layer_max,_argmax) = torch.max((x+(mask-1.0)).view(x.shape[0],x.shape[1],-1).to(torch.float32), dim=2)
        layer_max = layer_max.view(x.shape[0],x.shape[1],1,1)

        out_pool1 = layer_mean
        out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0)
        out_pool3 = layer_max

        out = torch.cat((out_pool1, out_pool2, out_pool3), dim=1)
        return out


class KataValueHeadGPool(torch.nn.Module):
    def __init__(self):
        super(KataValueHeadGPool, self).__init__()

    def forward(self, x, mask, mask_sum_hw):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111

        Returns: NC11
        """
        mask_sum_hw_sqrt_offset = torch.sqrt(mask_sum_hw) - 14.0

        layer_mean = torch.sum(x, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw

        out_pool1 = layer_mean
        out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0)
        out_pool3 = layer_mean * ((mask_sum_hw_sqrt_offset * mask_sum_hw_sqrt_offset) / 100.0 - 0.1)

        out = torch.cat((out_pool1, out_pool2, out_pool3), dim=1)
        return out

class KataConvAndGPool(torch.nn.Module):
    def __init__(self, name, c_in, c_out, c_gpool, config, activation):
        super(KataConvAndGPool, self).__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.activation = activation
        self.conv1r = torch.nn.Conv2d(c_in, c_out, kernel_size=3, padding="same", bias=False)
        self.conv1g = torch.nn.Conv2d(c_in, c_gpool, kernel_size=3, padding="same", bias=False)
        self.normg = NormMask(
            c_gpool,
            config=config,
            fixup_use_gamma=False,
        )
        self.actg = act(self.activation, inplace=True)
        self.gpool = KataGPool()
        self.linear_g = torch.nn.Linear(3 * c_gpool, c_out, bias=False)

    def initialize(self, scale):
        # Scaling so that variance on the r and g branches adds up to 1.0
        r_scale = 0.8
        g_scale = 0.6
        if self.norm_kind == "fixup" or self.norm_kind == "fixscale" or self.norm_kind == "fixbrenorm" or self.norm_kind == "fixscaleonenorm":
            init_weights(self.conv1r.weight, self.activation, scale=scale * r_scale)
            init_weights(self.conv1g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))
            init_weights(self.linear_g.weight, self.activation, scale=math.sqrt(scale) * math.sqrt(g_scale))
        else:
            init_weights(self.conv1r.weight, self.activation, scale=scale*r_scale)
            init_weights(self.conv1g.weight, self.activation, scale=math.sqrt(scale) * 1.0)
            init_weights(self.linear_g.weight, self.activation, scale=math.sqrt(scale) * g_scale)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        reg_dict["normal"].append(self.conv1r.weight)
        reg_dict["normal"].append(self.conv1g.weight)
        self.normg.add_reg_dict(reg_dict)
        reg_dict["normal"].append(self.linear_g.weight)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        self.normg.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        self.normg.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)


    def forward(self, x, mask, mask_sum_hw, mask_sum:float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        out = x
        outr = self.conv1r(out)
        outg = self.conv1g(out)

        outg = self.normg(outg, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask=mask, mask_sum_hw=mask_sum_hw).squeeze(-1).squeeze(-1)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)

        out = outr + outg
        return out


class NormActConv(torch.nn.Module):
    def __init__(
        self,
        name: str,
        c_in: int,
        c_out: int,
        c_gpool: Optional[int],
        config: modelconfigs.ModelConfig,
        activation: str,
        kernel_size: int,
        fixup_use_gamma: bool,
    ):
        super(NormActConv, self).__init__()
        self.name = name
        self.c_in = c_in
        self.c_out = c_out
        self.c_gpool = c_gpool
        self.norm = NormMask(
            c_in,
            config=config,
            fixup_use_gamma=fixup_use_gamma,
        )
        self.activation = activation
        self.act = act(activation, inplace=True)
        self.use_repvgg_init = kernel_size > 1 and "use_repvgg_init" in config and config["use_repvgg_init"]

        if c_gpool is not None:
            self.convpool = KataConvAndGPool(name=name+".convpool",c_in=c_in, c_out=c_out, c_gpool=c_gpool, config=config, activation=activation)
            self.conv = None
        else:
            self.conv = torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding="same", bias=False)
            self.convpool = None

        self.conv1x1 = None
        if self.conv is not None and kernel_size > 1 and "use_repvgg_linear" in config and config["use_repvgg_linear"]:
            self.conv1x1 = torch.nn.Conv2d(c_in, c_out, kernel_size=1, padding="same", bias=False)

    def initialize(self, scale, norm_scale=None):
        self.norm.set_scale(norm_scale)
        if self.convpool is not None:
            self.convpool.initialize(scale=scale)
        else:
            if self.conv1x1 is not None:
                init_weights(self.conv1x1.weight, self.activation, scale=scale*0.6)
                init_weights(self.conv.weight, self.activation, scale=scale*0.8)
            else:
                if self.use_repvgg_init:
                    init_weights(self.conv.weight, self.activation, scale=scale*0.8)
                    center_bonus = self.conv.weight.new_zeros((self.conv.weight.shape[0],self.conv.weight.shape[1]),requires_grad=False)
                    init_weights(center_bonus, self.activation, scale=scale*0.6)
                    self.conv.weight[:,:,1,1] += center_bonus
                else:
                    init_weights(self.conv.weight, self.activation, scale=scale)


    def add_reg_dict(self, reg_dict:Dict[str,List]):
        self.norm.add_reg_dict(reg_dict)
        if self.convpool is not None:
            self.convpool.add_reg_dict(reg_dict)
        else:
            if self.conv1x1 is not None:
                reg_dict["normal"].append(self.conv1x1.weight)
            reg_dict["normal"].append(self.conv.weight)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        self.norm.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        if self.convpool is not None:
            self.convpool.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        self.norm.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        if self.convpool is not None:
            self.convpool.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        out = x
        out = self.norm(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
        out = self.act(out)
        # print("TENSOR AFTER NORMACT")
        # print(out)
        if self.convpool is not None:
            out = self.convpool(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        else:
            if self.conv1x1 is not None:
                out = self.conv(out) + self.conv1x1(out)
            else:
                out = self.conv(out)
        if extra_outputs is not None:
            extra_outputs.report(self.name+".out", out)
        return out


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        name: str,
        c_main: int,
        c_mid: int,
        c_gpool: Optional[int],
        config: modelconfigs.ModelConfig,
        activation: str,
    ):
        super(ResBlock, self).__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.normactconv1 = NormActConv(
            name=name+".normactconv1",
            c_in=c_main,
            c_out=c_mid - (0 if c_gpool is None else c_gpool),
            c_gpool=c_gpool,
            config=config,
            activation=activation,
            kernel_size=3,
            fixup_use_gamma=False,
        )
        self.normactconv2 = NormActConv(
            name=name+".normactconv2",
            c_in=c_mid - (0 if c_gpool is None else c_gpool),
            c_out=c_main,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=3,
            fixup_use_gamma=True,
        )

    def initialize(self, fixup_scale):
        if self.norm_kind == "fixup":
            self.normactconv1.initialize(scale=fixup_scale)
            self.normactconv2.initialize(scale=0.0)
        elif self.norm_kind == "fixscale" or self.norm_kind == "fixbrenorm" or self.norm_kind == "fixscaleonenorm":
            self.normactconv1.initialize(scale=1.0, norm_scale=fixup_scale)
            self.normactconv2.initialize(scale=1.0)
        else:
            self.normactconv1.initialize(scale=1.0)
            self.normactconv2.initialize(scale=1.0)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        self.normactconv1.add_reg_dict(reg_dict)
        self.normactconv2.add_reg_dict(reg_dict)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        self.normactconv1.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        self.normactconv2.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        self.normactconv1.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        self.normactconv2.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW (residual only, caller is responsible for adding to trunk)
        """
        out = x
        out = self.normactconv1(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        out = self.normactconv2(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        if extra_outputs is not None:
            extra_outputs.report(self.name+".out", out)
        return out


class BottleneckResBlock(torch.nn.Module):
    def __init__(
        self,
        name: str,
        internal_length: int,
        c_main: int,
        c_mid: int,
        c_gpool: Optional[int],
        config: modelconfigs.ModelConfig,
        activation: str,
    ):
        super(BottleneckResBlock, self).__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.internal_length = internal_length
        assert internal_length >= 1

        self.normactconvp = NormActConv(
            name=name+".normactconvp",
            c_in=c_main,
            c_out=c_mid,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=1,
            fixup_use_gamma=False,
        )

        self.normactconvstack = torch.nn.ModuleList()
        self.normactconvstack.append(NormActConv(
            name=name+".normactconvstack."+str(0),
            c_in=c_mid,
            c_out=c_mid - (0 if c_gpool is None else c_gpool),
            c_gpool=c_gpool,
            config=config,
            activation=activation,
            kernel_size=3,
            fixup_use_gamma=False,
        ))
        for i in range(self.internal_length-1):
            self.normactconvstack.append(NormActConv(
                name=name+".normactconvstack."+str(i+1),
                c_in=self.normactconvstack[-1].c_out,
                c_out=c_mid,
                c_gpool=None,
                config=config,
                activation=activation,
                kernel_size=3,
                fixup_use_gamma=False,
            ))

        self.normactconvq = NormActConv(
            name=name+".normactconvq",
            c_in=self.normactconvstack[-1].c_out,
            c_out=c_main,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=1,
            fixup_use_gamma=True,
        )

    def initialize(self, fixup_scale):
        if self.norm_kind == "fixup":
            self.normactconvp.initialize(scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)))
            for i in range(self.internal_length):
                self.normactconvstack[i].initialize(scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)))
            self.normactconvq.initialize(scale=0.0)
        elif self.norm_kind == "fixscale" or self.norm_kind == "fixbrenorm" or self.norm_kind == "fixscaleonenorm":
            self.normactconvp.initialize(scale=1.0, norm_scale=fixup_scale)
            for i in range(self.internal_length):
                self.normactconvstack[i].initialize(scale=1.0)
            self.normactconvq.initialize(scale=1.0)
        else:
            self.normactconvp.initialize(scale=1.0)
            for i in range(self.internal_length):
                self.normactconvstack[i].initialize(scale=1.0)
            self.normactconvq.initialize(scale=1.0)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        self.normactconvp.add_reg_dict(reg_dict)
        for i in range(self.internal_length):
            self.normactconvstack[i].add_reg_dict(reg_dict)
        self.normactconvq.add_reg_dict(reg_dict)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        self.normactconvp.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        for i in range(self.internal_length):
            self.normactconvstack[i].set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        self.normactconvq.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        self.normactconvp.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        for i in range(self.internal_length):
            self.normactconvstack[i].add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        self.normactconvq.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW (residual only, caller is responsible for adding to trunk)
        """
        out = x
        out = self.normactconvp(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        for i in range(self.internal_length):
            out = self.normactconvstack[i](out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        out = self.normactconvq(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        if extra_outputs is not None:
            extra_outputs.report(self.name+".out", out)
        return out


class NestedBottleneckResBlock(torch.nn.Module):
    def __init__(
        self,
        name: str,
        internal_length: int,
        c_main: int,
        c_mid: int,
        c_gpool: Optional[int],
        config: modelconfigs.ModelConfig,
        activation: str,
    ):
        super(NestedBottleneckResBlock, self).__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.internal_length = internal_length
        assert internal_length >= 1

        self.normactconvp = NormActConv(
            name=name+".normactconvp",
            c_in=c_main,
            c_out=c_mid,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=1,
            fixup_use_gamma=False,
        )

        self.blockstack = torch.nn.ModuleList()
        for i in range(self.internal_length):
            self.blockstack.append(ResBlock(
                name=name+".blockstack."+str(i),
                c_main=c_mid,
                c_mid=c_mid,
                c_gpool=(c_gpool if i == 0 else None),
                config=config,
                activation=activation,
            ))

        self.normactconvq = NormActConv(
            name=name+".normactconvq",
            c_in=c_mid,
            c_out=c_main,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=1,
            fixup_use_gamma=True,
        )

    def initialize(self, fixup_scale):
        if self.norm_kind == "fixup":
            self.normactconvp.initialize(scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)))
            for i in range(self.internal_length):
                self.blockstack[i].initialize(fixup_scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)))
            self.normactconvq.initialize(scale=0.0)
        elif self.norm_kind == "fixscale" or self.norm_kind == "fixbrenorm" or self.norm_kind == "fixscaleonenorm":
            self.normactconvp.initialize(scale=1.0, norm_scale=fixup_scale)
            for i in range(self.internal_length):
                self.blockstack[i].initialize(fixup_scale=1.0 / math.sqrt(i+1.0))
            self.normactconvq.initialize(scale=1.0, norm_scale=1.0 / math.sqrt(self.internal_length+1.0))
        else:
            self.normactconvp.initialize(scale=1.0)
            for i in range(self.internal_length):
                self.blockstack[i].initialize(fixup_scale=1.0)
            self.normactconvq.initialize(scale=1.0)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        self.normactconvp.add_reg_dict(reg_dict)
        for i in range(self.internal_length):
            self.blockstack[i].add_reg_dict(reg_dict)
        self.normactconvq.add_reg_dict(reg_dict)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        self.normactconvp.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        for i in range(self.internal_length):
            self.blockstack[i].set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        self.normactconvq.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        self.normactconvp.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        for i in range(self.internal_length):
            self.blockstack[i].add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        self.normactconvq.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW (residual only, caller is responsible for adding to trunk)
        """
        out = x
        out = self.normactconvp(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        for i in range(self.internal_length):
            out = out + self.blockstack[i](out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        out = self.normactconvq(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        if extra_outputs is not None:
            extra_outputs.report(self.name+".out", out)
        return out


class DilationNestedBottleneckResBlock(torch.nn.Module):
    def __init__(
        self,
        name: str,
        internal_length: int,
        c_main: int,
        c_mid: int,
        config: modelconfigs.ModelConfig,
        activation: str,
    ):
        super(DilationNestedBottleneckResBlock, self).__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.internal_length = internal_length
        assert internal_length >= 1

        self.normactconvp = NormActConv(
            name=name+".normactconvp",
            c_in=c_main,
            c_out=c_mid,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=1,
            fixup_use_gamma=False,
        )

        self.blockstack = torch.nn.ModuleList()
        for i in range(self.internal_length):
            self.blockstack.append(ResBlock(
                name=name+".blockstack."+str(i),
                c_main=c_mid,
                c_mid=c_mid,
                c_gpool=None,
                config=config,
                activation=activation,
            ))

        self.normactconvq = NormActConv(
            name=name+".normactconvq",
            c_in=c_mid,
            c_out=c_main,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=1,
            fixup_use_gamma=True,
        )

    def initialize(self, fixup_scale):
        if self.norm_kind == "fixup":
            self.normactconvp.initialize(scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)))
            for i in range(self.internal_length):
                self.blockstack[i].initialize(fixup_scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)))
            self.normactconvq.initialize(scale=0.0)
        elif self.norm_kind == "fixscale" or self.norm_kind == "fixbrenorm" or self.norm_kind == "fixscaleonenorm":
            self.normactconvp.initialize(scale=1.0, norm_scale=fixup_scale)
            for i in range(self.internal_length):
                self.blockstack[i].initialize(fixup_scale=1.0 / math.sqrt(i+1.0))
            self.normactconvq.initialize(scale=1.0, norm_scale=1.0 / math.sqrt(self.internal_length+1.0))
        else:
            self.normactconvp.initialize(scale=1.0)
            for i in range(self.internal_length):
                self.blockstack[i].initialize(fixup_scale=1.0)
            self.normactconvq.initialize(scale=1.0)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        self.normactconvp.add_reg_dict(reg_dict)
        for i in range(self.internal_length):
            self.blockstack[i].add_reg_dict(reg_dict)
        self.normactconvq.add_reg_dict(reg_dict)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        self.normactconvp.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        for i in range(self.internal_length):
            self.blockstack[i].set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        self.normactconvq.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        self.normactconvp.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        for i in range(self.internal_length):
            self.blockstack[i].add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        self.normactconvq.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW (residual only, caller is responsible for adding to trunk)
        """
        out = x
        # mask_sum_hw and mask_sum are None because the dilation rearrangement changes the spatial
        # dimensions, invalidating the original values. Only fixup/fixscale norms are supported here.
        out = self.normactconvp(out, mask=mask, mask_sum_hw=None, mask_sum=None, extra_outputs=extra_outputs)

        assert len(out.shape) == 4
        assert len(mask.shape) == 4
        n,c,h,w = out.shape
        # pad to multiple of 3
        padding_h = (3 - h % 3) % 3
        padding_w = (3 - w % 3) % 3
        if padding_w != 0 or padding_h != 0:
            out = torch.nn.functional.pad(out, (0, padding_w, 0, padding_h))
            mask_t = torch.nn.functional.pad(mask, (0, padding_w, 0, padding_h))
        else:
            mask_t = mask
        padded_h = h + padding_h
        padded_w = w + padding_w
        padded_h_div3 = padded_h // 3
        padded_w_div3 = padded_w // 3
        # transpose!
        out = out.reshape((n,c,padded_h_div3,3,padded_w_div3,3))
        out = out.permute(0,3,5,1,2,4)
        assert tuple(out.shape) == (n,3,3,c,padded_h_div3,padded_w_div3), f"{tuple(out.shape)=}"
        out = out.reshape((n*3*3,c,padded_h_div3,padded_w_div3))
        mask_t = mask_t.reshape((n,1,padded_h_div3,3,padded_w_div3,3))
        mask_t = mask_t.permute(0,3,5,1,2,4)
        mask_t = mask_t.reshape((n*3*3,1,padded_h_div3,padded_w_div3))
        mask_t = mask_t.detach()

        # mask_sum_hw and mask_sum are None - see comment above on normactconvp.
        for i in range(self.internal_length):
            out = out + self.blockstack[i](out, mask=mask_t, mask_sum_hw=None, mask_sum=None, extra_outputs=extra_outputs)

        # untranspose!
        out = out.reshape((n,3,3,c,padded_h_div3,padded_w_div3))
        out = out.permute(0,3,4,1,5,2)
        assert tuple(out.shape) == (n,c,padded_h_div3,3,padded_w_div3,3), f"{tuple(out.shape)=}"
        out = out.reshape((n,c,padded_h,padded_w))
        out = out[:,:,:h,:w].contiguous()

        # mask_sum_hw and mask_sum are None - see comment above on normactconvp.
        out = self.normactconvq(out, mask=mask, mask_sum_hw=None, mask_sum=None, extra_outputs=extra_outputs)
        if extra_outputs is not None:
            extra_outputs.report(self.name+".out", out)
        return out


# =============================================================================
# Positional encoding and attention bias for transformers
# =============================================================================
# TransformerBlock supports several positional encoding / attention bias methods.
# All are translationally equivariant so that the position of a masked small board
# within a larger tensor does not matter, up to float precision, so that the
# net can generalize across board sizes.
#
# RoPE (Rotary Position Embeddings):
# Static positional encoding. Precomputes sin/cos tables for the full
# pos_len x pos_len grid and rotates Q/K vectors so that their dot product depends
# on relative position based on fixed frequencies regardless of board content.
# Requires head_dim % 4 == 0 for 2D interleaved layout.
# Only axis-aligned frequencies are included, cannot express diagonal attention
# except as a product of axis attention.
#
# Learnable RoPE (config "learnable_rope": True):
# Replaces fixed axis-aligned frequencies with learnable 2D frequencies,
# learnable per head. Enables heads adaptively choosing what they are sensitive to
# and to also attend to diagonal offsets or patterns.
#
# GAB (Geometric Attention Bias):
# Similar to the Chessformer paper from 2026 https://openreview.net/forum?id=2ltBRzEHyd
# Produces per-head (board area x board area) attention bias matrices that are
# added to QK^T logits before softmax. Materializes the full S x S bias.
# Steps:
# Shared template generation (GABTemplateMLP, computed once): A small MLP maps relative
# offsets (dr, dc) between all grid position pairs to gab_num_templates many templates
# using Fourier features, including diagonal cross-terms (dr+dc) and (dr-dc)
# so the MLP can easily represent spatial patterns like rows, columns, and diagonals.
# Each template is e.g. a (19x19)x(19x19) bias with this internal MLP-originated
# translational structure. This is different than Chessformer which used hardcoded
# 64x64 templates for Chess, since in Go we want to generalize across board size.
#
# TAB (Topological Attention Bias):
#
# TODO: (lightvector) I think I was stupid here, I think all of this might actually
# be implemented significantly more cheaply by dropping all of the rotate/unrotate
# operations EXCEPT for the last one. Why? Because convolutions can simply learn to
# fold the rotate/unrotate into their own weights, if that's what's needed.
# Worth testing later - but for now we can leave this here as a record of the
# conceptual path we walked getting here.
#
# In games like Go, a chain of stones can make spatially distant squares effectively
# "near" each other. We try to address this by TAB, which produces templates that
# react to board state.
# We do this by having a preliminary module that computes some complex convolutions
# of the board state. We initialize every square of the board via a learnable function
# of the input to some complex values, and then perform a RoPE-like rotation by learnable
# frequencies, so that considering all frequencies together, every position on the board
# can be uniquely keyed by its given combination of fourier frequencies, while also
# carrying information of what's on the board at that spot (e.g. stones).
# We then perform a series of dilated and regular 3x3 complex-valued convolutions, to
# rapidly mix and allow locations on the board to compute things like "what's the frequency
# signature of the space 3 spaces east of me" as a function of the stones on the board.
# If the net wanted to do things like try to assign different "groups" different frequency
# signatures that are common across the stones in the group, in theory it should
# be able to do that here.
# We also apply nonlinear activations, but those activations are performed in "unrotated"
# space (i.e. we undo the RoPE-like rotations) - equivalently, you can think of us as
# twisting the axis on which each activation acts on to align with the RoPE rotations.
# This ensures that the entire module is equivariant to phase, and thus produces
# attention biases that are invariant to translation. You could translate the
# board within a tensor by any amount, causing all the different frequencies to phase-shift,
# and the module would still compute the same result. Any fixed rotation by a given phase
# commutes through the complex-valued convolutions - the only thing that matters is
# the *relative* difference in phase between different spatial locations.
# There is also a frequency-mixing TAB variant that rather than having each frequency have
# its own bundle of channels, every channel has a different frequency, and convs are
# restricted to be depthwise, and there is a separate channelwise frequency mixing 1x1 conv
# that happens in unrotated space. (3x3 convs can never mix frequencies since that breaks
# equivariance).
# The final output of TAB is a set of bias templates just like GAB, except that we avoid
# materializing them because due to input dependence, we'd get different templates per
# every batch element, and N * 19 * 19 * 19 * 19 * num_templates is too much memory.
# Instead, we keep them in factored key-query form and append them on to the keys and
# queries in dot product attention. In theory, these keys and queries should enable
# the net to encode things like "attend to all stones in this group" for reasonably
# sized groups, or "attend to all vulnerable opposing groups in a capturing race",
# or "attend to all nearby liberties of a given group".
#
# Per-layer template selection: All bias mechanisms (GAB, TAB) share the same
# pathway in _compute_gab_bias: we do global pooling and allow the net to pick an
# arbitrary linear combination of templates for each head based on the global state.
# Materialized biases (GAB) are added to the attention mask; factored biases (TAB)
# produce extra K/Q dims that are concatenated onto the main attention keys/queries.
#
# Without RoPE, all other operations in the transformer block (Q/K/V projections,
# attention, FFN, norms) are per-token and permutation-equivariant - the attention
# bias mechanisms or RoPE are the only source of positional information within the block.
#
# Masking: Off-board key positions receive -inf attention bias (from the position mask),
# which dominates any finite bias, so softmax zeros them out. The compression pathway's
# mean pooling also masks off-board positions when summarizing the board state.
# =============================================================================

def precompute_freqs_cos_sin_2d(dim, pos_len, theta=100.0):
    """Precompute cos and sin tables of 2D frequencies for RoPE (real-valued, interleaved layout).
    Returns shape: (pos_len * pos_len, dim)
    """
    assert dim % 4 == 0
    dim_half = dim // 2

    freqs = 1.0 / (theta ** (torch.arange(0, dim_half, 2).float() / dim_half))

    t = torch.arange(pos_len, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(t, t, indexing='ij')

    emb_h = grid_h.unsqueeze(-1) * freqs
    emb_w = grid_w.unsqueeze(-1) * freqs

    emb = torch.cat([emb_h, emb_w], dim=-1)
    emb = emb.flatten(0, 1)
    emb = emb.repeat_interleave(2, dim=-1)

    return emb.cos(), emb.sin()

def apply_rotary_emb(xq, xk, cos, sin):
    """Apply rotary position embeddings to Q and K tensors.
    xq, xk: (Batch, Seq, Heads, Dim)
    cos, sin: (Seq, Dim)
    """
    def rotate_every_two(x):
        x = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = x.unbind(dim=-1)
        x_rotated = torch.stack([-x1, x0], dim=-1)
        return x_rotated.flatten(-2)

    cos = cos.view(1, xq.shape[1], 1, xq.shape[-1])
    sin = sin.view(1, xq.shape[1], 1, xq.shape[-1])

    xq_out = xq * cos + rotate_every_two(xq) * sin
    xk_out = xk * cos + rotate_every_two(xk) * sin

    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_learnable_rotary_emb(xq, xk, cos_q, sin_q, cos_k, sin_k):
    """Apply learnable rotary position embeddings to Q and K tensors.
    xq: (Batch, Seq, num_heads, Dim)
    xk: (Batch, Seq, num_kv_heads, Dim)
    cos_q, sin_q: (Seq, num_heads, Dim/2) - per-head, per-pair
    cos_k, sin_k: (Seq, num_kv_heads, Dim/2) - per-kv-head, per-pair
    """
    def _rotate(x, cos, sin):
        B, S, H, D = x.shape
        P = D // 2
        x_pairs = x.view(B, S, H, P, 2)
        x0, x1 = x_pairs.unbind(dim=-1)  # each (B, S, H, P)
        cos = cos.unsqueeze(0)  # (1, S, H, P)
        sin = sin.unsqueeze(0)
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return out.reshape(B, S, H, D).type_as(x)

    return _rotate(xq, cos_q, sin_q), _rotate(xk, cos_k, sin_k)


def compute_gab_fourier_features(dr, dc, freqs):
    """Compute Fourier features for relative (dr, dc) offsets.
    dr: (...) float tensor of row offsets
    dc: (...) float tensor of col offsets
    freqs: (num_freqs,) tensor of learnable frequencies
    Returns: (..., 8*num_freqs)
    """
    features = []
    dr_plus_dc = dr + dc
    dr_minus_dc = dr - dc
    for f in freqs:
        features.append(torch.sin(f * dr).unsqueeze(-1))
        features.append(torch.cos(f * dr).unsqueeze(-1))
        features.append(torch.sin(f * dc).unsqueeze(-1))
        features.append(torch.cos(f * dc).unsqueeze(-1))
        features.append(torch.sin(f * dr_plus_dc).unsqueeze(-1))
        features.append(torch.cos(f * dr_plus_dc).unsqueeze(-1))
        features.append(torch.sin(f * dr_minus_dc).unsqueeze(-1))
        features.append(torch.cos(f * dr_minus_dc).unsqueeze(-1))
    return torch.cat(features, dim=-1)


GAB_TEMPLATES = "gab_templates"
TAB_KQ = "tab_kq"

@dataclass
class GABTemplateData:
    """Precomputed GAB template values, shared across all blocks in a forward pass.
    By convention, templates are pre-scaled by 1/sqrt of the appropriate quantity
    so that a weighted combination does not need further scaling.
    """
    templates: torch.Tensor  # (S, S, T) template values for all position pairs

@dataclass
class TABKeyQueryData:
    """Precomputed factored TAB keys and queries, shared across all blocks in a forward pass.
    Instead of materializing (N, T, S, S) templates, stores the factored keys/queries
    so they can be concatenated onto the main attention K/Q.
    By convention, keys and/or queries are pre-scaled by 1/sqrt of the appropriate quantity
    so that a weighted combination does not need further scaling.
    """
    keys: torch.Tensor    # (N, 2*F, 1, S) - single complex key shared across templates
    queries: torch.Tensor # (N, 2*F, T, S) - complex query vectors per template


class GABTemplateMLP(torch.nn.Module):
    """Shared module that maps relative (dr, dc) offsets to T template values.
    Computed once and shared across all GAB-enabled transformer blocks.
    """
    def __init__(self, gab_num_templates, gab_num_fourier_features, gab_mlp_hidden, pos_len, activation):
        # Let F = gab_num_fourier_features, H = gab_mlp_hidden, T = gab_num_templates
        # S = pos_len * pos_len (max spatial positions)
        super().__init__()
        self.gab_num_templates = gab_num_templates
        self.activation = activation
        self.act = act(activation)
        assert gab_num_fourier_features >= 2, "gab_num_fourier_features must be >= 2"
        fourier_input_dim = 8 * gab_num_fourier_features  # 8*F

        # Geometric initialization from 1 rad/square to 1/50 rad/square
        init_freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1.0 / 50.0), gab_num_fourier_features))
        self.gab_freqs = torch.nn.Parameter(init_freqs)  # (F,)

        self.linear1 = torch.nn.Linear(fourier_input_dim, gab_mlp_hidden)  # (8*F) -> (H)
        self.linear2 = torch.nn.Linear(gab_mlp_hidden, gab_num_templates)  # (H) -> (T)

        S = pos_len * pos_len
        s_idx = torch.arange(S)
        s_r, s_c = s_idx // pos_len, s_idx % pos_len
        offset_dr = (s_r.unsqueeze(1) - s_r.unsqueeze(0)).float()  # (S, S)
        offset_dc = (s_c.unsqueeze(1) - s_c.unsqueeze(0)).float()  # (S, S)
        self.register_buffer("offset_dr", offset_dr, persistent=False)
        self.register_buffer("offset_dc", offset_dc, persistent=False)

    def forward(self, seq_len):
        """Compute templates for all position pairs up to seq_len.
        Returns: (seq_len, seq_len, T)
        """
        dr = self.offset_dr[:seq_len, :seq_len]              # (S, S)
        dc = self.offset_dc[:seq_len, :seq_len]              # (S, S)
        fourier_feats = compute_gab_fourier_features(dr, dc, self.gab_freqs)  # (S, S, 8*F)
        x = self.act(self.linear1(fourier_feats))            # (S, S, H)
        x = self.linear2(x)                                  # (S, S, T)
        scale = 1.0 / math.sqrt(self.gab_num_templates)
        return x * scale


    def initialize(self):
        init_weights(self.linear1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, "identity", scale=1.0)

    def add_reg_dict(self, reg_dict):
        reg_dict["noreg"].append(self.gab_freqs)
        reg_dict["gab_mlp"].append(self.linear1.weight)
        reg_dict["noreg"].append(self.linear1.bias)
        reg_dict["gab_mlp"].append(self.linear2.weight)
        reg_dict["noreg"].append(self.linear2.bias)


def tab_rotate(z, cos_a, sin_a):
    """Apply complex rotation to z.
    z: (*, 2, c_z, H, W) where dim -4 is [real, imag]
    cos_a, sin_a: broadcastable to (*, 1, c_z, H, W)
    Returns: same shape as z
    """
    r = z[:, 0:1, :, :, :]  # (*, 1, c_z, H, W)
    i = z[:, 1:2, :, :, :]
    new_r = r * cos_a - i * sin_a
    new_i = r * sin_a + i * cos_a
    return torch.cat([new_r, new_i], dim=-4)


class ComplexConv2d(torch.nn.Module):
    """A 2D convolution that enforces complex multiplication structure.

    Stores real_kernel and imag_kernel of shape (c_out, c_in, K, K).
    Builds the (2*c_out, 2*c_in, K, K) block-structured kernel:
        [[real_kernel, -imag_kernel],
         [imag_kernel,  real_kernel]]
    and applies F.conv2d.

    Input: (*, 2*c_in, H, W), Output: (*, 2*c_out, H, W).
    """
    def __init__(self, c_in, c_out=None, kernel_size=1, dilation=1):
        super().__init__()
        if c_out is None:
            c_out = c_in
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.real_kernel = torch.nn.Parameter(torch.empty(c_out, c_in, kernel_size, kernel_size))
        self.imag_kernel = torch.nn.Parameter(torch.empty(c_out, c_in, kernel_size, kernel_size))

    def forward(self, x):
        # We encode c_in x c_in complex convolution as a 2*c_in x 2*c_in real convolution
        # where the kernel is constrained to have the appropriate structure.
        top = torch.cat([self.real_kernel, -self.imag_kernel], dim=1)  # (c_out, 2*c_in, K, K)
        bot = torch.cat([self.imag_kernel, self.real_kernel], dim=1)   # (c_out, 2*c_in, K, K)
        kernel = torch.cat([top, bot], dim=0)  # (2*c_out, 2*c_in, K, K)
        padding = self.dilation * (self.kernel_size // 2)
        return torch.nn.functional.conv2d(x, kernel, padding=padding, dilation=self.dilation)

    def initialize(self, activation, scale=1.0):
        init_weights(self.real_kernel, activation, scale=scale / math.sqrt(2.0))
        init_weights(self.imag_kernel, activation, scale=scale / math.sqrt(2.0))


class TABEquivariantBlock(torch.nn.Module):
    """One equivariant residual block for TAB.

    Contains two complex convolutions (first with dilation, second without)
    with activations and RoPE-style rotations for equivariance.
    """
    def __init__(self, c_z, activation, dilation):
        super().__init__()
        self.act1 = act(activation)
        self.conv1 = ComplexConv2d(c_z, kernel_size=3, dilation=dilation)
        self.act2 = act(activation)
        self.conv2 = ComplexConv2d(c_z, kernel_size=3, dilation=1)
        self.c_z = c_z

    def forward(self, z, cos_a, sin_a, block_idx):
        """
        z: (NF, 2, c_z, H, W)
        cos_a, sin_a: (NF, 1, 1, H, W)
        block_idx: int, for variance normalization
        """
        zskip = z
        # Normalize - variance after block_idx prior blocks is proportional to block_idx + 1
        # (if we model the input as variance 1 and each block as contributing variance 1)
        z = z * (1.0 / math.sqrt(block_idx + 1))
        z = self.act1(z)
        z = tab_rotate(z, cos_a, sin_a)
        z = z.reshape(z.shape[0], 2 * self.c_z, z.shape[3], z.shape[4])
        z = self.conv1(z)
        z = z.reshape(z.shape[0], 2, self.c_z, z.shape[2], z.shape[3])
        z = tab_rotate(z, cos_a, -sin_a)
        z = self.act2(z)
        z = tab_rotate(z, cos_a, sin_a)
        z = z.reshape(z.shape[0], 2 * self.c_z, z.shape[3], z.shape[4])
        z = self.conv2(z)
        z = z.reshape(z.shape[0], 2, self.c_z, z.shape[2], z.shape[3])
        z = tab_rotate(z, cos_a, -sin_a)
        z = z + zskip
        return z

    def initialize(self, activation):
        self.conv1.initialize(activation, scale=1.0)
        self.conv2.initialize(activation, scale=1.0)


class TABModule(torch.nn.Module):
    """Shared module that generates factored input-dependent attention bias.

    Uses a stack of rotationally-equivariant complex convolutional blocks
    with learnable 2D RoPE-style frequencies. Produces factored keys and queries
    via complex key-query projections.

    Uses a single shared key projection and T query projections,
    returning factored (keys, queries) that are concatenated onto
    the main attention K/Q in each transformer block.

    Computed once and shared across all transformer blocks.
    """
    def __init__(self, trunk_channels, tab_c_z, tab_num_templates, tab_num_freqs, tab_num_blocks, tab_dilation, activation, pos_len):
        super().__init__()
        self.tab_c_z = tab_c_z
        self.tab_num_freqs = tab_num_freqs
        self.tab_num_templates = tab_num_templates
        self.tab_num_blocks = tab_num_blocks
        self.activation = activation

        # 1x1 conv to project trunk channels -> 2*F*c_z (interpreted as F*c_z complex values)
        self.input_proj = torch.nn.Conv2d(trunk_channels, 2 * tab_num_freqs * tab_c_z, kernel_size=1, bias=False)

        # Learnable 2D RoPE frequencies: (F, 2) for (omega_X, omega_Y)
        # Geometric initialization from 1 rad/square to 1/50 rad/square
        log_lo = math.log(1.0 / 50.0)
        log_hi = math.log(1.0)
        init_freqs = torch.exp(torch.empty(tab_num_freqs, 2).uniform_(log_lo, log_hi))
        init_freqs = init_freqs * (torch.randint(0, 2, (tab_num_freqs, 2)) * 2 - 1).float()
        self.rope_freqs = torch.nn.Parameter(init_freqs)

        self.blocks = torch.nn.ModuleList()
        for _ in range(tab_num_blocks):
            self.blocks.append(TABEquivariantBlock(tab_c_z, activation, tab_dilation))

        self.final_act = act(activation)
        self.key_proj = ComplexConv2d(tab_c_z, 1, kernel_size=1)
        self.query_proj = ComplexConv2d(tab_c_z, tab_num_templates, kernel_size=1)

    def forward(self, x, mask):
        """
        x: (N, C, H, W) trunk output
        mask: (N, 1, H, W) or None
        Returns: (keys, queries) with keys (N, 2*F, 1, S) and queries (N, 2*F, T, S), pre-scaled
        """
        N, C, H, W = x.shape
        S = H * W
        F = self.tab_num_freqs
        T = self.tab_num_templates
        c_z = self.tab_c_z

        z = self.input_proj(x)  # (N, 2*F*c_z, H, W)
        z = z.view(N, F, 2, c_z, H, W)

        # Precompute angles from learnable frequencies and grid coordinates
        gy = torch.arange(H, device=x.device, dtype=x.dtype)
        gx = torch.arange(W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H, W)
        # angles[f, y, x] = omega_f_X * x + omega_f_Y * y
        angles = self.rope_freqs[:, 0:1].unsqueeze(-1) * grid_x.unsqueeze(0) + \
                 self.rope_freqs[:, 1:2].unsqueeze(-1) * grid_y.unsqueeze(0)  # (F, H, W)
        cos_a = torch.cos(angles).view(1, F, 1, 1, H, W)  # (1, F, 1, 1, H, W)
        sin_a = torch.sin(angles).view(1, F, 1, 1, H, W)

        # Apply mask to zero off-board positions
        if mask is not None:
            z = z * mask.view(N, 1, 1, 1, H, W)

        # Fold N*F into batch dimension for batched processing
        z = z.reshape(N * F, 2, c_z, H, W)
        cos_a_batched = cos_a.expand(N, F, 1, 1, H, W).reshape(N * F, 1, 1, H, W)
        sin_a_batched = sin_a.expand(N, F, 1, 1, H, W).reshape(N * F, 1, 1, H, W)

        # Equivariant blocks
        block_idx = 0
        for block in self.blocks:
            z = block(z, cos_a_batched, sin_a_batched, block_idx)
            block_idx += 1

        # Normalize to variance 1 - variance after block_idx prior blocks is proportional to block_idx + 1
        # (if we model the input as variance 1 and each block as contributing variance 1)
        z = z * (1.0 / math.sqrt(block_idx + 1))

        # Final projection: activate, rotate into RoPE space, project keys/queries
        z = self.final_act(z)
        z = tab_rotate(z, cos_a_batched, sin_a_batched)

        z_flat = z.reshape(N * F, 2 * c_z, H, W)

        keys = self.key_proj(z_flat)      # (N*F, 2, H, W)
        queries = self.query_proj(z_flat)  # (N*F, 2*T, H, W)
        # Reshape: (N*F, 2*(T or 1), H, W) -> (N, 2*F, (T or 1), S)
        keys = keys.view(N, 2 * F, 1, S)
        queries = queries.view(N, 2 * F, T, S)
        return keys / math.sqrt(F), queries / math.sqrt(self.tab_num_templates)

    def initialize(self):
        init_weights(self.input_proj.weight, self.activation, scale=1.0)
        for block in self.blocks:
            block.initialize(self.activation)
        self.key_proj.initialize(self.activation, scale=1.0)
        self.query_proj.initialize(self.activation, scale=1.0)

    def add_reg_dict(self, reg_dict):
        reg_dict["tab_module"].append(self.input_proj.weight)
        reg_dict["noreg"].append(self.rope_freqs)
        for block in self.blocks:
            reg_dict["tab_module"].append(block.conv1.real_kernel)
            reg_dict["tab_module"].append(block.conv1.imag_kernel)
            reg_dict["tab_module"].append(block.conv2.real_kernel)
            reg_dict["tab_module"].append(block.conv2.imag_kernel)
        reg_dict["tab_module"].append(self.key_proj.real_kernel)
        reg_dict["tab_module"].append(self.key_proj.imag_kernel)
        reg_dict["tab_module"].append(self.query_proj.real_kernel)
        reg_dict["tab_module"].append(self.query_proj.imag_kernel)


class ComplexDepthwiseConv2d(torch.nn.Module):
    """Depthwise 2D complex convolution.

    Each of the c channels gets its own K x K complex kernel (no cross-channel mixing).
    Stores real_kernel and imag_kernel of shape (c, 1, K, K).

    Computes complex multiplication via two separate depthwise convolutions (groups=c):
        out_real = real_kernel * in_real - imag_kernel * in_imag
        out_imag = imag_kernel * in_real + real_kernel * in_imag

    Input: (*, 2*c, H, W) where channels are [re_0..re_{c-1}, im_0..im_{c-1}].
    Output: same layout.
    """
    def __init__(self, c, kernel_size=3, dilation=1):
        super().__init__()
        self.c = c
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.real_kernel = torch.nn.Parameter(torch.empty(c, 1, kernel_size, kernel_size))
        self.imag_kernel = torch.nn.Parameter(torch.empty(c, 1, kernel_size, kernel_size))

    def forward(self, x):
        # x: (*, 2*c, H, W) laid out as [re_0, ..., re_{c-1}, im_0, ..., im_{c-1}]
        padding = self.dilation * (self.kernel_size // 2)
        x_re = x[..., :self.c, :, :]   # (*, c, H, W)
        x_im = x[..., self.c:, :, :]   # (*, c, H, W)

        # Conv 1: convolve [re; im] with [rk; ik], fully depthwise (groups=2c)
        x_ri = torch.cat([x_re, x_im], dim=-3)              # (*, 2c, H, W)
        k_ri = torch.cat([self.real_kernel, self.imag_kernel], dim=0)  # (2c, 1, K, K)
        conv1 = torch.nn.functional.conv2d(x_ri, k_ri, padding=padding, dilation=self.dilation, groups=2 * self.c)
        # conv1: (*, 2c, H, W) = [rk*re; ik*im]

        # Conv 2: convolve [re; im] with [-ik; rk], fully depthwise (groups=2c)
        k_neg_ir = torch.cat([-self.imag_kernel, self.real_kernel], dim=0)  # (2c, 1, K, K)
        conv2 = torch.nn.functional.conv2d(x_ri, k_neg_ir, padding=padding, dilation=self.dilation, groups=2 * self.c)
        # conv2: (*, 2c, H, W) = [-ik*re; rk*im]

        # out_re = rk*re - ik*im = conv1[:c] - conv1[c:]
        # out_im = ik*re + rk*im = -conv2[:c] + conv2[c:]
        out_re = conv1[..., :self.c, :, :] - conv1[..., self.c:, :, :]
        out_im = conv2[..., self.c:, :, :] - conv2[..., :self.c, :, :]
        return torch.cat([out_re, out_im], dim=-3)

    def initialize(self, activation, scale=1.0):
        init_weights(self.real_kernel, activation, scale=scale / math.sqrt(2.0))
        init_weights(self.imag_kernel, activation, scale=scale / math.sqrt(2.0))


class FrequencyMixingTABBlock(torch.nn.Module):
    """One residual block for frequency-mixing TAB.

    Depthwise convs are per-frequency in the rotated frame (equivariant).
    1x1 convs mix freely across all 2*c_z channels in the unrotated frame (equivariant).
    """
    def __init__(self, c_z, activation, dilation):
        super().__init__()
        self.c_z = c_z
        self.act1 = act(activation)
        self.dw_conv1 = ComplexDepthwiseConv2d(c_z, kernel_size=3, dilation=dilation)
        self.mix1 = torch.nn.Conv2d(2 * c_z, 2 * c_z, kernel_size=1, bias=False)
        self.act2 = act(activation)
        self.dw_conv2 = ComplexDepthwiseConv2d(c_z, kernel_size=3, dilation=1)
        self.mix2 = torch.nn.Conv2d(2 * c_z, 2 * c_z, kernel_size=1, bias=False)

    def forward(self, z, cos_a, sin_a, block_idx):
        """
        z: (N, 2, c_z, H, W) - [real, imag] x c_z frequency channels
        cos_a, sin_a: (1, 1, c_z, H, W) - per-frequency angles, broadcastable
        block_idx: int, for variance normalization
        """
        N, _, c_z, H, W = z.shape
        zskip = z

        # Normalize variance (same logic as TABEquivariantBlock)
        z = z * (1.0 / math.sqrt(block_idx + 1))

        z = self.act1(z)

        # Depthwise conv in rotated frame
        z = tab_rotate(z, cos_a, sin_a)
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.dw_conv1(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)
        z = tab_rotate(z, cos_a, -sin_a)

        # 1x1 channel mixing in unrotated frame
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.mix1(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)

        z = self.act2(z)

        # Depthwise conv in rotated frame
        z = tab_rotate(z, cos_a, sin_a)
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.dw_conv2(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)
        z = tab_rotate(z, cos_a, -sin_a)

        # 1x1 channel mixing in unrotated frame
        z_flat = z.reshape(N, 2 * c_z, H, W)
        z_flat = self.mix2(z_flat)
        z = z_flat.view(N, 2, c_z, H, W)

        z = z + zskip
        return z

    def initialize(self, activation):
        self.dw_conv1.initialize(activation, scale=1.0)
        self.dw_conv2.initialize(activation, scale=1.0)
        init_weights(self.mix1.weight, activation, scale=1.0)
        init_weights(self.mix2.weight, "identity", scale=1.0)

    def add_reg_dict(self, reg_dict):
        reg_dict["tab_module"].append(self.dw_conv1.real_kernel)
        reg_dict["tab_module"].append(self.dw_conv1.imag_kernel)
        reg_dict["tab_module"].append(self.mix1.weight)
        reg_dict["tab_module"].append(self.dw_conv2.real_kernel)
        reg_dict["tab_module"].append(self.dw_conv2.imag_kernel)
        reg_dict["tab_module"].append(self.mix2.weight)


class FrequencyMixingTABModule(torch.nn.Module):
    """TAB module with frequency mixing.

    Unlike TABModule where each frequency has an independent c_z-channel convnet,
    here c_z IS the number of frequencies. Frequencies interact via pointwise (1x1)
    convs in the unrotated frame, spatial mixing happens via depthwise convs in the
    rotated frame. This preserves translational equivariance.
    """
    def __init__(self, trunk_channels, tab_c_z, tab_num_templates, tab_num_blocks, tab_dilation, activation, pos_len):
        super().__init__()
        self.tab_c_z = tab_c_z  # = number of frequencies
        self.tab_num_templates = tab_num_templates
        self.tab_num_blocks = tab_num_blocks
        self.activation = activation

        # 1x1 conv to project trunk channels -> 2*c_z (interpreted as c_z complex values)
        self.input_proj = torch.nn.Conv2d(trunk_channels, 2 * tab_c_z, kernel_size=1, bias=False)

        # Learnable 2D RoPE frequencies: (c_z, 2) for (omega_X, omega_Y)
        # Geometric initialization from 1 rad/square to 1/50 rad/square
        log_lo = math.log(1.0 / 50.0)
        log_hi = math.log(1.0)
        init_freqs = torch.exp(torch.empty(tab_c_z, 2).uniform_(log_lo, log_hi))
        init_freqs = init_freqs * (torch.randint(0, 2, (tab_c_z, 2)) * 2 - 1).float()
        self.rope_freqs = torch.nn.Parameter(init_freqs)

        self.blocks = torch.nn.ModuleList()
        for _ in range(tab_num_blocks):
            self.blocks.append(FrequencyMixingTABBlock(tab_c_z, activation, tab_dilation))

        self.final_act = act(activation)
        self.key_proj = torch.nn.Conv2d(2 * tab_c_z, 2 * tab_c_z, kernel_size=1, bias=False)
        self.query_proj = torch.nn.Conv2d(2 * tab_c_z, 2 * tab_c_z * tab_num_templates, kernel_size=1, bias=False)

    def forward(self, x, mask):
        """
        x: (N, C, H, W) trunk output
        mask: (N, 1, H, W) or None
        Returns: (keys, queries) with keys (N, 2*c_z, 1, S) and queries (N, 2*c_z, T, S), pre-scaled
        """
        N, C, H, W = x.shape
        S = H * W
        c_z = self.tab_c_z
        T = self.tab_num_templates

        z = self.input_proj(x)

        # Apply mask to zero off-board positions
        if mask is not None:
            z = z * mask

        z = z.view(N, 2, c_z, H, W)

        # Precompute angles
        gy = torch.arange(H, device=x.device, dtype=x.dtype)
        gx = torch.arange(W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H, W)
        angles = self.rope_freqs[:, 0:1].unsqueeze(-1) * grid_x.unsqueeze(0) + \
                 self.rope_freqs[:, 1:2].unsqueeze(-1) * grid_y.unsqueeze(0)  # (c_z, H, W)
        # Shape (1, 1, c_z, H, W) to broadcast with (N, 2, c_z, H, W) in tab_rotate
        cos_a = torch.cos(angles).view(1, 1, c_z, H, W)
        sin_a = torch.sin(angles).view(1, 1, c_z, H, W)

        block_idx = 0
        for block in self.blocks:
            z = block(z, cos_a, sin_a, block_idx)
            block_idx += 1

        # Normalize variance
        z = z * (1.0 / math.sqrt(block_idx + 1))

        # Final: activate, project keys/queries in unrotated space, then rotate
        z = self.final_act(z)
        z_flat = z.reshape(N, 2 * c_z, H, W)

        # cos/sin for final rotation: (1, c_z, 1, 1, H, W) -> tile across N samples
        # After folding N*c_z into batch, need (N*c_z, 1, 1, H, W)
        cos_a_out = cos_a.view(1, c_z, 1, 1, H, W).expand(N, c_z, 1, 1, H, W).reshape(N * c_z, 1, 1, H, W)
        sin_a_out = sin_a.view(1, c_z, 1, 1, H, W).expand(N, c_z, 1, 1, H, W).reshape(N * c_z, 1, 1, H, W)

        # Keys: mix in unrotated space first, reshape per-frequency, then rotate
        keys = self.key_proj(z_flat)  # (N, 2*c_z, H, W)
        keys = keys.view(N * c_z, 2, 1, H, W)
        keys = tab_rotate(keys, cos_a_out, sin_a_out)

        # Queries: mix in unrotated space first, reshape per-frequency, then rotate
        queries = self.query_proj(z_flat)  # (N, 2*c_z*T, H, W)
        queries = queries.view(N * c_z, 2, T, H, W)
        queries = tab_rotate(queries, cos_a_out, sin_a_out)

        keys = keys.reshape(N, 2 * c_z, 1, S)
        queries = queries.reshape(N, 2 * c_z, T, S)
        return keys / math.sqrt(c_z), queries / math.sqrt(T)

    def initialize(self):
        init_weights(self.input_proj.weight, self.activation, scale=1.0)
        for block in self.blocks:
            block.initialize(self.activation)
        init_weights(self.key_proj.weight, self.activation, scale=1.0)
        init_weights(self.query_proj.weight, self.activation, scale=1.0)

    def add_reg_dict(self, reg_dict):
        reg_dict["tab_module"].append(self.input_proj.weight)
        reg_dict["noreg"].append(self.rope_freqs)
        for block in self.blocks:
            block.add_reg_dict(reg_dict)
        reg_dict["tab_module"].append(self.key_proj.weight)
        reg_dict["tab_module"].append(self.query_proj.weight)


class NestedBottleneckTransformerBlock(torch.nn.Module):
    """A bottleneck residual block that uses transformer blocks internally.

    Structure: 1x1 conv (c_main -> c_mid) -> N transformer blocks at c_mid -> 1x1 conv (c_mid -> c_main) + residual
    This mirrors NestedBottleneckResBlock but replaces the inner conv ResBlocks with TransformerBlocks.
    """
    def __init__(
        self,
        name: str,
        internal_length: int,
        c_main: int,
        c_mid: int,
        config: modelconfigs.ModelConfig,
        activation: str,
        pos_len: int,
        use_swiglu: bool,
        use_rope: bool = True,
        use_gab: bool = False,
        use_tab: bool = False,
    ):
        super(NestedBottleneckTransformerBlock, self).__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.internal_length = internal_length
        assert internal_length >= 1

        self.normactconvp = NormActConv(
            name=name+".normactconvp",
            c_in=c_main,
            c_out=c_mid,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=1,
            fixup_use_gamma=False,
        )

        self.blockstack = torch.nn.ModuleList()
        for i in range(self.internal_length):
            self.blockstack.append(TransformerAttentionBlock(
                name=name+".blockstack.attn"+str(i+1),
                c_main=c_mid,
                config=config,
                activation=activation,
                pos_len=pos_len,
                use_rope=use_rope,
                use_gab=use_gab,
                use_tab=use_tab,
            ))
            self.blockstack.append(TransformerFFNBlock(
                name=name+".blockstack.ffn"+str(i+1),
                c_main=c_mid,
                config=config,
                activation=activation,
                use_swiglu=use_swiglu,
            ))

        self.normactconvq = NormActConv(
            name=name+".normactconvq",
            c_in=c_mid,
            c_out=c_main,
            c_gpool=None,
            config=config,
            activation=activation,
            kernel_size=1,
            fixup_use_gamma=True,
        )

    def initialize(self, fixup_scale):
        num_internal_blocks = 2 * self.internal_length
        if self.norm_kind == "fixup":
            self.normactconvp.initialize(scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)))
            for i in range(num_internal_blocks):
                self.blockstack[i].initialize(fixup_scale=math.pow(fixup_scale, 1.0 / (1.0 + self.internal_length)))
            self.normactconvq.initialize(scale=0.0)
        elif self.norm_kind == "fixscale" or self.norm_kind == "fixbrenorm" or self.norm_kind == "fixscaleonenorm":
            self.normactconvp.initialize(scale=1.0, norm_scale=fixup_scale)
            for i in range(num_internal_blocks):
                # Scale based on logical transformer block index (i//2), not the doubled block object index,
                # since splitting into attention+FFN blocks should not change initialization scaling.
                self.blockstack[i].initialize(fixup_scale=1.0 / math.sqrt(i//2+1.0))
            self.normactconvq.initialize(scale=1.0, norm_scale=1.0 / math.sqrt(self.internal_length+1.0))
        else:
            self.normactconvp.initialize(scale=1.0)
            for i in range(num_internal_blocks):
                self.blockstack[i].initialize(fixup_scale=1.0)
            self.normactconvq.initialize(scale=1.0)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        self.normactconvp.add_reg_dict(reg_dict)
        for block in self.blockstack:
            block.add_reg_dict(reg_dict)
        self.normactconvq.add_reg_dict(reg_dict)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        self.normactconvp.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        for block in self.blockstack:
            block.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        self.normactconvq.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        self.normactconvp.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        for block in self.blockstack:
            block.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        self.normactconvq.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW (residual only, caller is responsible for adding to trunk)
        """
        out = x
        out = self.normactconvp(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        for block in self.blockstack:
            out = out + block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs, block_shared_data=block_shared_data)
        out = self.normactconvq(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs)
        if extra_outputs is not None:
            extra_outputs.report(self.name+".out", out)
        return out


class TransformerAttentionBlock(torch.nn.Module):
    """Self-attention half of a transformer block, with its own residual connection.

    Contains: RMSNorm -> Q/K/V projections -> (optional RoPE) -> attention -> output projection.
    Returns residual only; caller is responsible for adding to trunk.
    """
    def __init__(
        self,
        name,
        c_main,
        config,
        activation,
        pos_len,
        use_rope=True,
        use_gab=False,
        use_tab=False,
    ):
        super(TransformerAttentionBlock, self).__init__()
        self.name = name
        self.norm_kind = config.get("norm_kind", "layer")
        self.use_rope = use_rope
        self.use_gab = use_gab
        self.use_tab = use_tab

        self.num_heads = config["transformer_heads"]
        self.num_kv_heads = config.get("transformer_kv_heads", self.num_heads)
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_head_dim = config.get("attention_query_head_dim", c_main // self.num_heads)
        self.v_head_dim = config.get("attention_value_head_dim", c_main // self.num_heads)

        if self.use_rope:
            assert self.q_head_dim % 4 == 0, f"Query head dim must be divisible by 4 for 2D RoPE"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"Query heads ({self.num_heads}) must be divisible by KV heads ({self.num_kv_heads})"

        self.q_proj = torch.nn.Linear(c_main, self.num_heads * self.q_head_dim, bias=False)
        self.k_proj = torch.nn.Linear(c_main, self.num_kv_heads * self.q_head_dim, bias=False)
        self.v_proj = torch.nn.Linear(c_main, self.num_kv_heads * self.v_head_dim, bias=False)
        self.out_proj = torch.nn.Linear(self.num_heads * self.v_head_dim, c_main, bias=False)

        # QK-norm: RMSNorm on Q and K per-head before the attention dot product.
        # See ViT-22B, etc.
        self.use_qk_norm = config.get("attention_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = torch.nn.RMSNorm(self.q_head_dim, eps=1e-6)
            self.k_norm = torch.nn.RMSNorm(self.q_head_dim, eps=1e-6)

        self.learnable_rope = config.get("learnable_rope", False) if self.use_rope else False
        if self.use_rope:
            if self.learnable_rope:
                assert self.q_head_dim % 2 == 0, f"Head dim must be even for learnable RoPE, got {self.q_head_dim}"
                num_pairs = self.q_head_dim // 2
                # Learnable 2D RoPE frequencies.
                # Geometric initialization from 1 rad/square to 1/50 rad/square
                log_lo = math.log(1.0 / 50.0)
                log_hi = math.log(1.0)
                init_freqs = torch.exp(torch.empty(self.num_kv_heads, num_pairs, 2).uniform_(log_lo, log_hi))
                init_freqs = init_freqs * (torch.randint(0, 2, (self.num_kv_heads, num_pairs, 2)) * 2 - 1).float()
                self.rope_freqs = torch.nn.Parameter(init_freqs)  # (num_kv_heads, P, 2)
                self.pos_len = pos_len
                self.cos_cached = None
                self.sin_cached = None
            else:
                self.rope_theta = config.get("rope_theta", 100.0)
                assert self.rope_theta > pos_len * 2.0, f"theta={self.rope_theta} of RoPE may be too small for pos_len={pos_len}"
                cos_cached, sin_cached = precompute_freqs_cos_sin_2d(self.q_head_dim, pos_len, self.rope_theta)
                self.register_buffer("cos_cached", cos_cached, persistent=False)
                self.register_buffer("sin_cached", sin_cached, persistent=False)
        else:
            self.cos_cached = None
            self.sin_cached = None

        if self.use_gab or self.use_tab:
            gab_d1 = config["gab_d1"]
            gab_d2 = config["gab_d2"]
            self.gab_num_templates = config["gab_num_templates"] if self.use_gab else 0
            self.tab_num_templates = config["tab_num_templates"] if self.use_tab else 0
            # Per-head weights: one per GAB template, one per TAB template.
            # TAB weights are per-template (shared across 2*F real/imag freq channels).
            self.total_num_weights = self.gab_num_templates + self.tab_num_templates
            self.gab_proj1 = torch.nn.Linear(c_main, gab_d1, bias=False)
            self.gab_proj2 = torch.nn.Linear(gab_d1, gab_d2, bias=False)
            self.gab_norm1 = torch.nn.RMSNorm(gab_d2, eps=1e-6)
            self.gab_proj3 = torch.nn.Linear(gab_d2, self.num_heads * self.total_num_weights, bias=False)
            self.gab_norm2 = torch.nn.RMSNorm(self.num_heads * self.total_num_weights, eps=1e-6)
            self.gab_act1 = act(activation, inplace=False)
            self.gab_act2 = act(activation, inplace=False)

        self.norm1 = torch.nn.RMSNorm(c_main, eps=1e-6)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        for name, param in self.named_parameters():
            if "norm" in name or "cached" in name:
                reg_dict["noreg"].append(param)
                continue
            if "weight" in name:
                if any(x in name for x in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                    reg_dict["normal_attn"].append(param)
                elif "gab_proj" in name:
                    reg_dict["normal_gab"].append(param)
                else:
                    reg_dict["normal"].append(param)
            else:
                reg_dict["noreg"].append(param)

    def initialize(self, fixup_scale):
        # Relies on torch initialization, nothing to do here.
        # Since we have active normalization layers, initial scaling doesn't matter so much.
        pass

    def set_brenorm_params(self, renorm_avg_momentum, rmax, dmax):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass

    def _compute_gab_bias(self, x_norm, mask, mask_sum_hw, block_shared_data):
        """Compute attention bias from GAB templates and/or TAB factored keys/queries.
        x_norm: (B, S, C) normalized token representations
        mask: (N, 1, H, W) or None
        mask_sum_hw: (N, 1, 1, 1) or None
        block_shared_data: dict with precomputed template/key-query data
        Returns: (template_bias, extra_kq) where
            template_bias: (B, H, S, S) materialized attention bias, or None
            extra_kq: (extra_k, extra_q) to concatenate onto main K/Q, or None
        """
        batch_size, seq_len, _ = x_norm.shape

        # Per-token projection
        y = self.gab_proj1(x_norm) # (B, S, d1)

        # Masked mean pooling over valid positions
        if mask is not None:
            mask_flat = mask.view(batch_size, seq_len, 1)  # (B, S, 1)
            y = y * mask_flat
            pooled = y.sum(dim=1) / mask_sum_hw.view(batch_size, 1)  # (B, d1)
        else:
            pooled = y.mean(dim=1)                       # (B, d1)

        # Compress + activation + norm
        z = self.gab_act1(self.gab_proj2(pooled))         # (B, d2)
        z = self.gab_norm1(z)

        # Generate per-head weights for all bias mechanisms
        z = self.gab_act2(self.gab_proj3(z))              # (B, H*total_num_weights)
        z = self.gab_norm2(z)
        z = z.view(batch_size, self.num_heads, self.total_num_weights)  # (B, H, W_total)

        bias = None
        extra_k_parts = []
        extra_q_parts = []
        idx = 0

        # GAB contribution: input-independent templates (S, S, T_gab)
        if self.use_gab:
            z_gab = z[:, :, idx:idx + self.gab_num_templates]
            idx += self.gab_num_templates
            gab_data = block_shared_data[GAB_TEMPLATES]
            gab_templates = gab_data.templates
            bias = torch.einsum("bhd,std->bhst", z_gab, gab_templates)

        # TAB contribution: mix templates in K/Q space, then append 2*F_tab dims.
        # Instead of keeping T templates separate (which would need 2*F*T extra dims),
        # we contract over templates before the dot product, yielding one mixed
        # key/query per frequency per head - only 2*F_tab extra dims.
        if self.use_tab:
            z_tab = z[:, :, idx:idx + self.tab_num_templates]  # (B, H, T)
            idx += self.tab_num_templates
            tab_data = block_shared_data[TAB_KQ]
            tab_keys = tab_data.keys         # (N, 2*F_tab, 1, S)
            tab_queries = tab_data.queries   # (N, 2*F_tab, T, S)
            # Mix queries across templates: einsum "bht, bfts -> bhfs"
            # z_tab: (B, H, T), tab_queries: (B, 2*F_tab, T, S) -> mixed_q: (B, H, 2*F_tab, S)
            mixed_q = torch.einsum("bht,bfts->bhfs", z_tab, tab_queries)  # (B, H, 2*F_tab, S)
            extra_q_parts.append(mixed_q.permute(0, 1, 3, 2))   # (B, H, S, 2*F_tab)

            tab_keys = tab_keys.squeeze(2).permute(0, 2, 1)       # (B, S, 2*F_tab)
            tab_keys = tab_keys.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            extra_k_parts.append(tab_keys)  # (B, H, S, 2*F_tab)

        assert idx == self.total_num_weights

        extra_kq = None
        if extra_k_parts:
            extra_k = torch.cat(extra_k_parts, dim=-1)  # (B, H, S, D_extra)
            extra_q = torch.cat(extra_q_parts, dim=-1)  # (B, H, S, D_extra)
            extra_kq = (extra_k, extra_q)

        return bias, extra_kq

    def forward(self, x, mask, mask_sum_hw, mask_sum:float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW (residual only, caller is responsible for adding to trunk)
        """
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        x_in = x.view(batch_size, channels, -1).permute(0, 2, 1)

        x_norm = self.norm1(x_in)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.q_head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.v_head_dim)

        if self.use_rope:
            if self.learnable_rope:
                # Compute per-head, per-pair angles from learnable 2D frequencies.
                # rope_freqs: (num_kv_heads, P, 2) = (H_kv, P, [omega_x, omega_y])
                s_idx = torch.arange(seq_len, device=q.device)
                s_y = (s_idx // self.pos_len).float()  # row
                s_x = (s_idx % self.pos_len).float()   # col
                # angles: (S, H_kv, P) = omega_x * x + omega_y * y
                angles = s_x.view(-1, 1, 1) * self.rope_freqs[:, :, 0] + s_y.view(-1, 1, 1) * self.rope_freqs[:, :, 1]
                cos_k = torch.cos(angles)  # (S, H_kv, P)
                sin_k = torch.sin(angles)
                # For Q: expand kv head freqs to match num_heads if using multi-query attention
                if self.n_rep > 1:
                    cos_q = cos_k.unsqueeze(2).expand(-1, -1, self.n_rep, -1).reshape(seq_len, self.num_heads, -1)
                    sin_q = sin_k.unsqueeze(2).expand(-1, -1, self.n_rep, -1).reshape(seq_len, self.num_heads, -1)
                else:
                    cos_q = cos_k
                    sin_q = sin_k
                q, k = apply_learnable_rotary_emb(q, k, cos_q, sin_q, cos_k, sin_k)
            else:
                q, k = apply_rotary_emb(q, k, self.cos_cached, self.sin_cached)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.n_rep, seq_len, self.q_head_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.q_head_dim)
            v = v.unsqueeze(2).expand(batch_size, self.num_kv_heads, self.n_rep, seq_len, self.v_head_dim)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.v_head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        template_bias = None
        extra_kq = None
        if self.use_gab or self.use_tab:
            template_bias, extra_kq = self._compute_gab_bias(x_norm, mask, mask_sum_hw, block_shared_data)

        if mask is not None:
            mask_flat = mask.view(batch_size, 1, 1, seq_len)
            attn_mask = torch.zeros_like(mask_flat, dtype=q.dtype)
            attn_mask.masked_fill_(mask_flat == 0, float('-inf'))
        else:
            attn_mask = None

        if template_bias is not None:
            if attn_mask is not None:
                attn_mask = attn_mask + template_bias
            else:
                attn_mask = template_bias

        # Default scaling for q/k dot product, 1/sqrt(query head dim)
        scale = 1.0 / math.sqrt(self.q_head_dim)

        if extra_kq is not None:
            # Concatenate extra keys/queries (from TAB) onto main K/Q.
            # q, k: (B, H, S, d_head), extra_k, extra_q: (B, H, S, D_extra)
            extra_k, extra_q = extra_kq

            # Pre-scale q and disable the overall scale passed to scaled_dot_product_attention
            # since the different extra q and extra k will have their own scaling.
            # The convention is that their scaling, if any, is already pre-multiplied in.
            q = q * scale
            scale = 1.0

            q = torch.cat([q, extra_q], dim=-1)  # (B, H, S, d_head + D_extra)
            k = torch.cat([k, extra_k], dim=-1)  # (B, H, S, d_head + D_extra)
            # v stays (B, H, S, d_head), scaled_dot_product_attention supports differing channels for v than q/k

        # If attention weights are requested, force the manual path so we can capture them.
        wants_attn_weights = (
            extra_outputs is not None
            and self.name+".attn_weights" in extra_outputs.requested
        )

        if not wants_attn_weights:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                scale=scale,
            )
        else:
            # Manual attention path to capture weights.
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S, S)

            if attn_mask is not None:
                logits = logits + attn_mask

            attn_weights = torch.softmax(logits, dim=-1)

            if extra_outputs is not None:
                extra_outputs.report(self.name+".attn_weights", attn_weights)

            attn_output = torch.matmul(attn_weights, v)  # (B, H, S, Dv)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        attn_output = self.out_proj(attn_output)

        result = attn_output.permute(0, 2, 1).view(batch_size, channels, height, width)
        if extra_outputs is not None:
            extra_outputs.report(self.name+".out", result)
        return result


class TransformerFFNBlock(torch.nn.Module):
    """Feed-forward half of a transformer block, with its own residual connection.

    Contains: RMSNorm -> FFN (optionally SwiGLU) -> optional depthwise conv.
    Returns residual only; caller is responsible for adding to trunk.
    """
    def __init__(
        self,
        name,
        c_main,
        config,
        activation,
        use_swiglu,
    ):
        super(TransformerFFNBlock, self).__init__()
        self.name = name
        self.norm_kind = config.get("norm_kind", "layer")
        self.ffn_dim = config["transformer_ffn_channels"]
        self.use_swiglu = use_swiglu

        self.use_depthwise_conv = config.get("transformer_ffn_depthwise_conv", False)

        self.ffn_linear1 = torch.nn.Linear(c_main, self.ffn_dim, bias=False)
        if self.use_swiglu:
            self.ffn_linear_gate = torch.nn.Linear(c_main, self.ffn_dim, bias=False)
            self.ffn_act = torch.nn.SiLU(inplace=False)
        else:
            self.ffn_act = act(activation, inplace=False)
        if self.use_depthwise_conv:
            self.ffn_dwconv = torch.nn.Conv2d(self.ffn_dim, self.ffn_dim, kernel_size=3, padding=1, groups=self.ffn_dim, bias=False)
        self.ffn_linear2 = torch.nn.Linear(self.ffn_dim, c_main, bias=False)

        self.norm = torch.nn.RMSNorm(c_main, eps=1e-6)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        for name, param in self.named_parameters():
            if "norm" in name:
                reg_dict["noreg"].append(param)
                continue
            if "weight" in name:
                reg_dict["normal"].append(param)
            else:
                reg_dict["noreg"].append(param)

    def initialize(self, fixup_scale):
        pass

    def set_brenorm_params(self, renorm_avg_momentum, rmax, dmax):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass

    def forward(self, x, mask, mask_sum_hw, mask_sum:float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW (residual only, caller is responsible for adding to trunk)
        """
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        x_in = x.view(batch_size, channels, -1).permute(0, 2, 1)

        xn = self.norm(x_in)

        if self.use_swiglu:
            x1 = self.ffn_linear1(xn)
            x1 = self.ffn_act(x1)
            x_gate = self.ffn_linear_gate(xn)
            x1 = x1 * x_gate
        else:
            x1 = self.ffn_linear1(xn)
            x1 = self.ffn_act(x1)
        if self.use_depthwise_conv:
            # Reshape to NCHW for depthwise conv, apply mask, reshape back
            x1_spatial = x1.permute(0, 2, 1).view(batch_size, self.ffn_dim, height, width)
            x1_spatial = self.ffn_dwconv(x1_spatial) * mask
            x1 = x1_spatial.view(batch_size, self.ffn_dim, -1).permute(0, 2, 1)
        x1 = self.ffn_linear2(x1)

        result = x1.permute(0, 2, 1).view(batch_size, channels, height, width)

        if extra_outputs is not None:
            extra_outputs.report(self.name+".out", result)
        return result


class PolicyHead(torch.nn.Module):
    def __init__(self, c_in, c_p1, c_g1, config, activation):
        super(PolicyHead, self).__init__()
        self.config = config
        self.activation = activation

        if config["version"] <= 11:
            self.num_policy_outputs = 4
        elif config["version"] <= 15:
            self.num_policy_outputs = 6
        else:
            self.num_policy_outputs = 8
        # Output 0: policy prediction
        # Output 1: opponent reply policy prediction
        # Output 2: soft policy prediction
        # Output 3: soft opponent reply policy prediction
        # Output 4: long-term-optimistic policy prediction
        # Output 5: short-term-optimistic policy prediction
        # Output 6: q value winloss prediction, pre-tanh
        # Output 7: q value score prediction

        self.conv1p = torch.nn.Conv2d(c_in, c_p1, kernel_size=1, padding="same", bias=False)
        self.conv1g = torch.nn.Conv2d(c_in, c_g1, kernel_size=1, padding="same", bias=False)

        self.biasg = BiasMask(
            c_g1,
            config=config,
            is_after_batchnorm=True,
        )
        self.actg = act(self.activation)
        self.gpool = KataGPool()

        self.linear_g = torch.nn.Linear(3 * c_g1, c_p1, bias=False)
        if config["version"] <= 14:
            self.linear_pass = torch.nn.Linear(3 * c_g1, self.num_policy_outputs, bias=False)
        else:
            self.linear_pass = torch.nn.Linear(3 * c_g1, c_p1, bias=True)
            self.act_pass = act(self.activation)
            self.linear_pass2 = torch.nn.Linear(c_p1, self.num_policy_outputs, bias=False)

        self.bias2 = BiasMask(
            c_p1,
            config=config,
            is_after_batchnorm=True,
        )
        self.act2 = act(activation)
        self.conv2p = torch.nn.Conv2d(c_p1, self.num_policy_outputs, kernel_size=1, padding="same", bias=False)


    def initialize(self):
        # Scaling so that variance on the p and g branches adds up to 1.0
        p_scale = 0.8
        g_scale = 0.6
        bias_scale = 0.2
        # Extra scaling for outputs
        scale_output = 0.3
        init_weights(self.conv1p.weight, self.activation, scale=p_scale)
        init_weights(self.conv1g.weight, self.activation, scale=1.0)
        init_weights(self.linear_g.weight, self.activation, scale=g_scale)
        if self.config["version"] <= 14:
            init_weights(self.linear_pass.weight, "identity", scale=scale_output)
        else:
            init_weights(self.linear_pass.weight, self.activation, scale=1.0)
            init_weights(self.linear_pass.bias, self.activation, scale=bias_scale, fan_tensor=self.linear_pass.weight)
            init_weights(self.linear_pass2.weight, "identity", scale=scale_output)
        init_weights(self.conv2p.weight, "identity", scale=scale_output)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        reg_dict["output"].append(self.conv1p.weight)
        reg_dict["output"].append(self.conv1g.weight)
        reg_dict["output"].append(self.linear_g.weight)
        if self.config["version"] <= 14:
            reg_dict["output"].append(self.linear_pass.weight)
        else:
            reg_dict["output"].append(self.linear_pass.weight)
            reg_dict["output_noreg"].append(self.linear_pass.bias)
            reg_dict["output"].append(self.linear_pass2.weight)

        reg_dict["output"].append(self.conv2p.weight)
        self.biasg.add_reg_dict(reg_dict)
        self.bias2.add_reg_dict(reg_dict)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass

    def forward(self, x, mask, mask_sum_hw, mask_sum:float, extra_outputs: Optional[ExtraOutputs], block_shared_data: Optional[Dict[str, Any]] = None):
        outp = self.conv1p(x)
        outg = self.conv1g(x)

        outg = self.biasg(outg, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask=mask, mask_sum_hw=mask_sum_hw).squeeze(-1).squeeze(-1) # NC

        if self.config["version"] <= 14:
            outpass = self.linear_pass(outg) # NC
        else:
            outpass = self.linear_pass(outg) # NC
            outpass = self.act_pass(outpass) # NC
            outpass = self.linear_pass2(outpass) # NC

        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1) # NCHW

        outp = outp + outg
        outp = self.bias2(outp, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
        outp = self.act2(outp)
        outp = self.conv2p(outp)
        outpolicy = outp

        # mask out parts outside the board by making them a huge neg number, so that they're 0 after softmax
        outpolicy = outpolicy - (1.0 - mask) * 5000.0
        # NC(HW) concat with NC1
        return torch.cat((outpolicy.view(outpolicy.shape[0],outpolicy.shape[1],-1), outpass.unsqueeze(-1)),dim=2)


class ValueHead(torch.nn.Module):
    def __init__(self, c_in, c_v1, c_v2, c_sv2, num_scorebeliefs, config, activation, pos_len):
        super(ValueHead, self).__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(c_in, c_v1, kernel_size=1, padding="same", bias=False)
        self.bias1 = BiasMask(
            c_v1,
            config=config,
            is_after_batchnorm=True,
        )
        self.act1 = act(activation)
        self.gpool = KataValueHeadGPool()

        self.linear2 = torch.nn.Linear(3 * c_v1, c_v2, bias=True)
        self.act2 = act(activation)

        self.linear_valuehead = torch.nn.Linear(c_v2, 3, bias=True)
        self.linear_miscvaluehead = torch.nn.Linear(c_v2, 10, bias=True)
        self.linear_moremiscvaluehead = torch.nn.Linear(c_v2, 8, bias=True)
        self.conv_ownership = torch.nn.Conv2d(c_v1, 1, kernel_size=1, padding="same", bias=False)
        self.conv_scoring = torch.nn.Conv2d(c_v1, 1, kernel_size=1, padding="same", bias=False)
        self.conv_futurepos = torch.nn.Conv2d(c_in, 2, kernel_size=1, padding="same", bias=False)
        self.conv_seki = torch.nn.Conv2d(c_in, 4, kernel_size=1, padding="same", bias=False)

        self.pos_len = pos_len
        self.scorebelief_mid = self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS
        self.scorebelief_len = self.scorebelief_mid * 2
        self.num_scorebeliefs = num_scorebeliefs
        self.c_sv2 = c_sv2

        self.linear_s2 = torch.nn.Linear(3 * c_v1, c_sv2, bias=True)
        self.linear_s2off = torch.nn.Linear(1, c_sv2, bias=False)
        self.linear_s2par = torch.nn.Linear(1, c_sv2, bias=False)
        self.linear_s3 = torch.nn.Linear(c_sv2, num_scorebeliefs, bias=True)
        self.linear_smix = torch.nn.Linear(3 * c_v1, num_scorebeliefs, bias=True)

        self.register_buffer("score_belief_offset_vector", torch.tensor(
            data=[(float(i-self.scorebelief_mid)+0.5) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
            requires_grad=False,
        ), persistent=False)
        self.register_buffer("score_belief_offset_bias_vector", torch.tensor(
            data=[0.05 * (float(i-self.scorebelief_mid)+0.5) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
            requires_grad=False,
        ), persistent=False)
        self.register_buffer("score_belief_parity_vector", torch.tensor(
            [0.5-float((i-self.scorebelief_mid) % 2) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
            requires_grad=False,
        ), persistent=False)


    def initialize(self):
        bias_scale = 0.2
        init_weights(self.conv1.weight, self.activation, scale=1.0)
        init_weights(self.linear2.weight, self.activation, scale=1.0)
        init_weights(self.linear2.bias, self.activation, scale=bias_scale, fan_tensor=self.linear2.weight)

        init_weights(self.linear_valuehead.weight, "identity", scale=1.0)
        init_weights(self.linear_valuehead.bias, "identity", scale=bias_scale, fan_tensor=self.linear_valuehead.weight)

        init_weights(self.linear_miscvaluehead.weight, "identity", scale=1.0)
        init_weights(self.linear_miscvaluehead.bias, "identity", scale=bias_scale, fan_tensor=self.linear_miscvaluehead.weight)

        init_weights(self.linear_moremiscvaluehead.weight, "identity", scale=1.0)
        init_weights(self.linear_moremiscvaluehead.bias, "identity", scale=bias_scale, fan_tensor=self.linear_moremiscvaluehead.weight)

        aux_spatial_output_scale = 0.2
        init_weights(self.conv_ownership.weight, "identity", scale=aux_spatial_output_scale)
        init_weights(self.conv_scoring.weight, "identity", scale=aux_spatial_output_scale)
        init_weights(self.conv_futurepos.weight, "identity", scale=aux_spatial_output_scale)
        init_weights(self.conv_seki.weight, "identity", scale=aux_spatial_output_scale)

        init_weights(self.linear_s2.weight, self.activation, scale=1.0)
        init_weights(self.linear_s2.bias, self.activation, scale=1.0, fan_tensor=self.linear_s2.weight)
        init_weights(self.linear_s2off.weight, self.activation, scale=1.0, fan_tensor=self.linear_s2.weight)
        init_weights(self.linear_s2par.weight, self.activation, scale=1.0, fan_tensor=self.linear_s2.weight)

        scorebelief_output_scale = 0.5
        init_weights(self.linear_s3.weight, "identity", scale=scorebelief_output_scale)
        init_weights(self.linear_s3.bias, "identity", scale=scorebelief_output_scale*bias_scale, fan_tensor=self.linear_s3.weight)
        init_weights(self.linear_smix.weight, "identity", scale=1.0)
        init_weights(self.linear_smix.bias, "identity", scale=bias_scale, fan_tensor=self.linear_smix.weight)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        reg_dict["output"].append(self.conv1.weight)
        reg_dict["output"].append(self.linear2.weight)
        reg_dict["output_noreg"].append(self.linear2.bias)
        reg_dict["output"].append(self.linear_valuehead.weight)
        reg_dict["output_noreg"].append(self.linear_valuehead.bias)
        reg_dict["output"].append(self.linear_miscvaluehead.weight)
        reg_dict["output_noreg"].append(self.linear_miscvaluehead.bias)
        reg_dict["output"].append(self.linear_moremiscvaluehead.weight)
        reg_dict["output_noreg"].append(self.linear_moremiscvaluehead.bias)
        reg_dict["output"].append(self.conv_ownership.weight)
        reg_dict["output"].append(self.conv_scoring.weight)
        reg_dict["output"].append(self.conv_futurepos.weight)
        reg_dict["output"].append(self.conv_seki.weight)
        reg_dict["output"].append(self.linear_s2.weight)
        reg_dict["output_noreg"].append(self.linear_s2.bias)
        reg_dict["output"].append(self.linear_s2off.weight)
        reg_dict["output"].append(self.linear_s2par.weight)
        reg_dict["output"].append(self.linear_s3.weight)
        reg_dict["output_noreg"].append(self.linear_s3.bias)
        reg_dict["output"].append(self.linear_smix.weight)
        reg_dict["output_noreg"].append(self.linear_smix.bias)
        self.bias1.add_reg_dict(reg_dict)

    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass

    def forward(self, x, mask, mask_sum_hw, mask_sum:float, input_global, extra_outputs: Optional[ExtraOutputs]):
        outv1 = x
        outv1 = self.conv1(outv1)
        outv1 = self.bias1(outv1, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
        outv1 = self.act1(outv1)

        outpooled = self.gpool(outv1, mask=mask, mask_sum_hw=mask_sum_hw).squeeze(-1).squeeze(-1)

        outv2 = self.linear2(outpooled)
        outv2 = self.act2(outv2)

        # Different subheads
        out_value = self.linear_valuehead(outv2)
        out_miscvalue = self.linear_miscvaluehead(outv2)
        out_moremiscvalue = self.linear_moremiscvaluehead(outv2)
        out_ownership = self.conv_ownership(outv1) * mask
        out_scoring = self.conv_scoring(outv1) * mask
        out_futurepos = self.conv_futurepos(x) * mask
        out_seki = self.conv_seki(x) * mask

        # Score belief head
        batch_size = x.shape[0]
        outsv2 = (
            self.linear_s2(outpooled).view(batch_size,1,self.c_sv2) +
            self.linear_s2off(self.score_belief_offset_bias_vector.view(1,self.scorebelief_len,1)) +
            self.linear_s2par((self.score_belief_parity_vector.view(1,self.scorebelief_len) * input_global[:,-1:]).view(batch_size,self.scorebelief_len,1))
        ) # N,scorebelief_len,c_sv2

        outsv2 = self.act2(outsv2)
        outsv3 = self.linear_s3(outsv2) # N, scorebelief_len, num_scorebeliefs

        outsmix = self.linear_smix(outpooled) # N, num_scorebeliefs
        outsmix_logweights = torch.nn.functional.log_softmax(outsmix, dim=1)
        # For each of num_scorebeliefs, compute softmax to make it into probability distribution
        out_scorebelief_logprobs = torch.nn.functional.log_softmax(outsv3, dim=1)
        # Take the mixture distribution weighted by outsmix_weights
        out_scorebelief_logprobs = torch.logsumexp(out_scorebelief_logprobs + outsmix_logweights.view(-1, 1, self.num_scorebeliefs), dim=2)

        return (
            out_value,
            out_miscvalue,
            out_moremiscvalue,
            out_ownership,
            out_scoring,
            out_futurepos,
            out_seki,
            out_scorebelief_logprobs,
        )

class MetadataEncoder(torch.nn.Module):
    def __init__(self, config: modelconfigs.ModelConfig):
        super(MetadataEncoder, self).__init__()

        self.config = config
        self.activation = config["activation"]
        self.meta_encoder_version = 1 if "meta_encoder_version" not in config["metadata_encoder"] else config["metadata_encoder"]["meta_encoder_version"]

        self.c_input = modelconfigs.get_num_meta_encoder_input_features(self.meta_encoder_version)
        assert self.c_input == 192

        self.c_internal = self.config["metadata_encoder"]["internal_num_channels"]
        self.c_trunk = self.config["trunk_num_channels"]
        self.out_scale = 0.5

        self.register_buffer("feature_mask", torch.tensor(
            # 86 is board area
            data=[(0.0 if i == 86 else 1.0) for i in range(self.c_input)],
            dtype=torch.float32,
            requires_grad=False,
        ), persistent=True)

        self.linear1 = torch.nn.Linear(self.c_input, self.c_internal, bias=True)
        self.act1 = act(self.activation, inplace=True)
        self.linear2 = torch.nn.Linear(self.c_internal, self.c_internal, bias=True)
        self.act2 = act(self.activation, inplace=True)
        self.linear_output_to_trunk = torch.nn.Linear(self.c_internal, self.c_trunk, bias=False)

    def initialize(self):
        weight_scale = 0.8
        bias_scale = 0.2
        with torch.no_grad():
            init_weights(self.linear1.weight, self.activation, scale=weight_scale)
            init_weights(self.linear1.bias, self.activation, scale=bias_scale, fan_tensor=self.linear1.weight)
            init_weights(self.linear2.weight, self.activation, scale=weight_scale)
            init_weights(self.linear2.bias, self.activation, scale=bias_scale, fan_tensor=self.linear2.weight)
            init_weights(self.linear_output_to_trunk.weight, self.activation, scale=weight_scale)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        reg_dict["input"].append(self.linear1.weight)
        reg_dict["input_noreg"].append(self.linear1.bias)
        reg_dict["input"].append(self.linear2.weight)
        reg_dict["input_noreg"].append(self.linear2.bias)
        reg_dict["input"].append(self.linear_output_to_trunk.weight)

    def forward(self, input_meta, extra_outputs: Optional[ExtraOutputs]):
        x = input_meta
        x = x * self.feature_mask.reshape((1,-1))
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return self.out_scale * self.linear_output_to_trunk(x)


# Exhaustive mapping of block kinds to whether they use GAB and/or TAB.
# If a new block kind is added without updating this dict, the lookup will raise
# NotImplementedError so the omission is caught immediately.
_BLOCK_KIND_FLAGS = {
    # (uses_gab, uses_tab)
    "regular":                              (False, False),
    "bottle1":                              (False, False),
    "bottle":                               (False, False),
    "bottle2":                              (False, False),
    "bottle3":                              (False, False),
    "bottlenest2":                          (False, False),
    "dilatedbottlenest2":                   (False, False),
    "bottlenest3":                          (False, False),
    "attnrope":                             (False, False),
    "attngab":                              (True,  False),
    "attnropegab":                          (True,  False),
    "attnropetab":                          (False, True),
    "ffnsg":                                (False, False),
    "ffng":                                 (False, False),
    "bottlenest2transformerropesg":         (False, False),
    "bottlenest2transformergabsg":          (True,  False),
    "bottlenest2transformerropegabsg":      (True,  False),
    "bottlenest2transformertabsg":         (False, True),
    "bottlenest2transformerropetabsg":     (False, True),
}

def _block_kind_base(block_kind: str) -> str:
    """Strip trailing 'gpool' suffix if present."""
    return block_kind[:-5] if block_kind.endswith("gpool") else block_kind

def _block_kind_uses_gab(block_kind: str) -> bool:
    base = _block_kind_base(block_kind)
    if base not in _BLOCK_KIND_FLAGS:
        raise NotImplementedError(f"Unknown block kind {block_kind!r}, add it to _BLOCK_KIND_FLAGS")
    return _BLOCK_KIND_FLAGS[base][0]

def _block_kind_uses_tab(block_kind: str) -> bool:
    base = _block_kind_base(block_kind)
    if base not in _BLOCK_KIND_FLAGS:
        raise NotImplementedError(f"Unknown block kind {block_kind!r}, add it to _BLOCK_KIND_FLAGS")
    return _BLOCK_KIND_FLAGS[base][1]


class Model(torch.nn.Module):
    def __init__(self, config: modelconfigs.ModelConfig, pos_len: int):
        super(Model, self).__init__()

        self.config = config
        self.norm_kind = config["norm_kind"]
        self.block_kind = config["block_kind"]
        self.c_trunk = config["trunk_num_channels"]
        self.c_mid = config["mid_num_channels"]
        self.c_gpool = config["gpool_num_channels"]
        self.c_outermid = config["outermid_num_channels"] if "outermid_num_channels" in config else self.c_mid
        self.c_p1 = config["p1_num_channels"]
        self.c_g1 = config["g1_num_channels"]
        self.c_v1 = config["v1_num_channels"]
        self.c_v2 = config["v2_size"]
        self.c_sv2 = config["sbv2_num_channels"]
        self.num_scorebeliefs = config["num_scorebeliefs"]
        self.num_total_blocks = len(self.block_kind)
        self.pos_len = pos_len

        if config["version"] <= 12:
            self.td_score_multiplier = 20.0
            self.scoremean_multiplier = 20.0
            self.scorestdev_multiplier = 20.0
            self.lead_multiplier = 20.0
            self.variance_time_multiplier = 40.0
            self.shortterm_value_error_multiplier = 0.25
            self.shortterm_score_error_multiplier = 30.0
        else:
            self.td_score_multiplier = 20.0
            self.scoremean_multiplier = 20.0
            self.scorestdev_multiplier = 20.0
            self.lead_multiplier = 20.0
            self.variance_time_multiplier = 40.0
            self.shortterm_value_error_multiplier = 0.25
            self.shortterm_score_error_multiplier = 150.0

        self.trunk_normless = "trunk_normless" in config and config["trunk_normless"]
        self.trunk_final_rmsnorm = "trunk_final_rmsnorm" in config and config["trunk_final_rmsnorm"]

        if "has_intermediate_head" in config and config["has_intermediate_head"]:
            self.has_intermediate_head = True
            self.intermediate_head_blocks = config["intermediate_head_blocks"]
        else:
            self.has_intermediate_head = False
            self.intermediate_head_blocks = 0

        self.activation = "relu" if "activation" not in config else config["activation"]

        if config["initial_conv_1x1"]:
            self.conv_spatial = torch.nn.Conv2d(22, self.c_trunk, kernel_size=1, padding="same", bias=False)
        else:
            self.conv_spatial = torch.nn.Conv2d(22, self.c_trunk, kernel_size=3, padding="same", bias=False)
        self.linear_global = torch.nn.Linear(19, self.c_trunk, bias=False)

        if "metadata_encoder" in config and config["metadata_encoder"] is not None:
            self.metadata_encoder = MetadataEncoder(config)
        else:
            self.metadata_encoder = None

        self.bin_input_shape = [22, pos_len, pos_len]
        self.global_input_shape = [19]

        # Create shared GAB template MLP if any block uses GAB
        has_gab = any(_block_kind_uses_gab(bk[1]) for bk in self.block_kind)
        if has_gab:
            self.gab_template_mlp = GABTemplateMLP(
                gab_num_templates=config["gab_num_templates"],
                gab_num_fourier_features=config["gab_num_fourier_features"],
                gab_mlp_hidden=config["gab_mlp_hidden"],
                pos_len=pos_len,
                activation=self.activation,
            )
        else:
            self.gab_template_mlp = None

        # Create shared TAB module if any block uses TAB
        has_tab = any(_block_kind_uses_tab(bk[1]) for bk in self.block_kind)
        if has_tab:
            if config.get("tab_use_frequency_mixing", False):
                self.tab_module = FrequencyMixingTABModule(
                    trunk_channels=self.c_trunk,
                    tab_c_z=config["tab_c_z"],
                    tab_num_templates=config["tab_num_templates"],
                    tab_num_blocks=config["tab_num_blocks"],
                    tab_dilation=config["tab_dilation"],
                    activation=self.activation,
                    pos_len=pos_len,
                )
            else:
                self.tab_module = TABModule(
                    trunk_channels=self.c_trunk,
                    tab_c_z=config["tab_c_z"],
                    tab_num_templates=config["tab_num_templates"],
                    tab_num_freqs=config["tab_num_freqs"],
                    tab_num_blocks=config["tab_num_blocks"],
                    tab_dilation=config["tab_dilation"],
                    activation=self.activation,
                    pos_len=pos_len,
                )
        else:
            self.tab_module = None

        self.blocks = torch.nn.ModuleList()
        for block_config in self.block_kind:
            block_name = block_config[0]
            block_kind = block_config[1]
            use_gpool_this_block = False
            if block_kind.endswith("gpool"):
                use_gpool_this_block = True
                block_kind = block_kind[:-5]

            if block_kind == "regular":
                self.blocks.append(ResBlock(
                    name=block_name,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    c_gpool=(self.c_gpool if use_gpool_this_block else None),
                    config=self.config,
                    activation=self.activation,
                ))
            elif block_kind == "bottle1" or block_kind == "bottle":
                self.blocks.append(BottleneckResBlock(
                    name=block_name,
                    internal_length=1,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    c_gpool=(self.c_gpool if use_gpool_this_block else None),
                    config=self.config,
                    activation=self.activation,
                ))
            elif block_kind == "bottle2":
                self.blocks.append(BottleneckResBlock(
                    name=block_name,
                    internal_length=2,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    c_gpool=(self.c_gpool if use_gpool_this_block else None),
                    config=self.config,
                    activation=self.activation,
                ))
            elif block_kind == "bottle3":
                self.blocks.append(BottleneckResBlock(
                    name=block_name,
                    internal_length=3,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    c_gpool=(self.c_gpool if use_gpool_this_block else None),
                    config=self.config,
                    activation=self.activation,
                ))
            elif block_kind == "bottlenest2":
                self.blocks.append(NestedBottleneckResBlock(
                    name=block_name,
                    internal_length=2,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    c_gpool=(self.c_gpool if use_gpool_this_block else None),
                    config=self.config,
                    activation=self.activation,
                ))
            elif block_kind == "dilatedbottlenest2":
                assert not use_gpool_this_block
                self.blocks.append(DilationNestedBottleneckResBlock(
                    name=block_name,
                    internal_length=2,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    config=self.config,
                    activation=self.activation,
                ))
            elif block_kind == "bottlenest3":
                self.blocks.append(NestedBottleneckResBlock(
                    name=block_name,
                    internal_length=3,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    c_gpool=(self.c_gpool if use_gpool_this_block else None),
                    config=self.config,
                    activation=self.activation,
                ))
            elif block_kind == "attnrope":
                self.blocks.append(TransformerAttentionBlock(
                    name=block_name,
                    c_main=self.c_trunk,
                    config=self.config,
                    activation=self.activation,
                    pos_len=pos_len,
                    use_rope=True,
                ))
            elif block_kind == "attngab":
                self.blocks.append(TransformerAttentionBlock(
                    name=block_name,
                    c_main=self.c_trunk,
                    config=self.config,
                    activation=self.activation,
                    pos_len=pos_len,
                    use_rope=False,
                    use_gab=True,
                ))
            elif block_kind == "attnropegab":
                self.blocks.append(TransformerAttentionBlock(
                    name=block_name,
                    c_main=self.c_trunk,
                    config=self.config,
                    activation=self.activation,
                    pos_len=pos_len,
                    use_rope=True,
                    use_gab=True,
                ))
            elif block_kind == "attnropetab":
                self.blocks.append(TransformerAttentionBlock(
                    name=block_name,
                    c_main=self.c_trunk,
                    config=self.config,
                    activation=self.activation,
                    pos_len=pos_len,
                    use_rope=True,
                    use_tab=True,
                ))
            elif block_kind == "ffnsg":
                self.blocks.append(TransformerFFNBlock(
                    name=block_name,
                    c_main=self.c_trunk,
                    config=self.config,
                    activation=self.activation,
                    use_swiglu=True,
                ))
            elif block_kind == "ffng":
                self.blocks.append(TransformerFFNBlock(
                    name=block_name,
                    c_main=self.c_trunk,
                    config=self.config,
                    activation=self.activation,
                    use_swiglu=False,
                ))
            elif block_kind == "bottlenest2transformerropesg":
                self.blocks.append(NestedBottleneckTransformerBlock(
                    name=block_name,
                    internal_length=2,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    config=self.config,
                    activation=self.activation,
                    pos_len=pos_len,
                    use_swiglu=True,
                    use_rope=True,
                ))
            elif block_kind == "bottlenest2transformergabsg":
                self.blocks.append(NestedBottleneckTransformerBlock(
                    name=block_name,
                    internal_length=2,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    config=self.config,
                    activation=self.activation,
                    pos_len=pos_len,
                    use_swiglu=True,
                    use_rope=False,
                    use_gab=True,
                ))
            elif block_kind == "bottlenest2transformerropegabsg":
                self.blocks.append(NestedBottleneckTransformerBlock(
                    name=block_name,
                    internal_length=2,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    config=self.config,
                    activation=self.activation,
                    pos_len=pos_len,
                    use_swiglu=True,
                    use_rope=True,
                    use_gab=True,
                ))
            elif block_kind == "bottlenest2transformertabsg":
                self.blocks.append(NestedBottleneckTransformerBlock(
                    name=block_name,
                    internal_length=2,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    config=self.config,
                    activation=self.activation,
                    pos_len=pos_len,
                    use_swiglu=True,
                    use_rope=False,
                    use_tab=True,
                ))
            elif block_kind == "bottlenest2transformerropetabsg":
                self.blocks.append(NestedBottleneckTransformerBlock(
                    name=block_name,
                    internal_length=2,
                    c_main=self.c_trunk,
                    c_mid=self.c_mid,
                    config=self.config,
                    activation=self.activation,
                    pos_len=pos_len,
                    use_swiglu=True,
                    use_rope=True,
                    use_tab=True,
                ))
            else:
                assert False, f"Unknown block kind: {block_config[1]}"

        # Trunk channel gating: per-channel learned gate that interpolates between
        # trunk and residual at each block.
        self.use_trunk_channel_gate = config.get("use_trunk_channel_gate", False)
        if self.use_trunk_channel_gate:
            num_blocks = len(self.blocks)
            self.trunk_channel_gate_logits = torch.nn.ParameterList()
            for k in range(num_blocks):
                self.trunk_channel_gate_logits.append(torch.nn.Parameter(torch.zeros(1, self.c_trunk, 1, 1)))

        if self.trunk_final_rmsnorm:
            spatial = config.get("trunk_rmsnorm_spatial", False)
            cgroup_size = config.get("rmsnorm_spatial_cgroup_size", None) if spatial else None
            self.norm_trunkfinal = RMSNormMask(self.c_trunk, self.config, spatial=spatial, cgroup_size=cgroup_size)
        elif self.trunk_normless:
            self.norm_trunkfinal = BiasMask(self.c_trunk, self.config, is_after_batchnorm=True)
        else:
            self.norm_trunkfinal = NormMask(self.c_trunk, self.config, fixup_use_gamma=False, is_last_batchnorm=True)
        self.act_trunkfinal = act(self.activation)

        self.policy_head = PolicyHead(
            self.c_trunk,
            self.c_p1,
            self.c_g1,
            self.config,
            self.activation,
        )
        self.value_head = ValueHead(
            self.c_trunk,
            self.c_v1,
            self.c_v2,
            self.c_sv2,
            self.num_scorebeliefs,
            self.config,
            self.activation,
            self.pos_len,
        )
        if self.has_intermediate_head:
            self.norm_intermediate_trunkfinal = NormMask(self.c_trunk, self.config, fixup_use_gamma=False, is_last_batchnorm=True)
            self.act_intermediate_trunkfinal = act(self.activation)
            self.intermediate_policy_head = PolicyHead(
                self.c_trunk,
                self.c_p1,
                self.c_g1,
                self.config,
                self.activation,
            )
            self.intermediate_value_head = ValueHead(
                self.c_trunk,
                self.c_v1,
                self.c_v2,
                self.c_sv2,
                self.num_scorebeliefs,
                self.config,
                self.activation,
                self.pos_len,
            )

    @property
    def device(self):
        return self.linear_global.weight.device

    def initialize(self):
        with torch.no_grad():
            spatial_scale = 0.8
            global_scale = 0.6
            init_weights(self.conv_spatial.weight, self.activation, scale=spatial_scale)
            init_weights(self.linear_global.weight, self.activation, scale=global_scale)

            if self.metadata_encoder is not None:
                self.metadata_encoder.initialize()
            if self.gab_template_mlp is not None:
                self.gab_template_mlp.initialize()
            if self.tab_module is not None:
                self.tab_module.initialize()

            if self.norm_kind == "fixup":
                fixup_scale = 1.0 / math.sqrt(self.num_total_blocks)
                for block in self.blocks:
                    block.initialize(fixup_scale=fixup_scale)
            elif self.norm_kind == "fixscale" or self.norm_kind == "fixbrenorm" or self.norm_kind == "fixscaleonenorm":
                for i, block in enumerate(self.blocks):
                    block.initialize(fixup_scale=1.0 / math.sqrt(i+1.0))
                self.norm_trunkfinal.set_scale(1.0 / math.sqrt(self.num_total_blocks+1.0))
            else:
                for block in self.blocks:
                    block.initialize(fixup_scale=1.0)

            self.policy_head.initialize()
            self.value_head.initialize()
            if self.has_intermediate_head:
                self.intermediate_policy_head.initialize()
                self.intermediate_value_head.initialize()

    def get_norm_kind(self) -> bool:
        return self.norm_kind

    def get_has_intermediate_head(self) -> bool:
        return self.has_intermediate_head
    def get_has_metadata_encoder(self) -> bool:
        return self.metadata_encoder is not None

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        reg_dict["input"] = []
        reg_dict["input_noreg"] = []
        reg_dict["normal"] = []
        reg_dict["normal_attn"] = []
        reg_dict["normal_gab"] = []
        reg_dict["gab_mlp"] = []
        reg_dict["tab_module"] = []
        reg_dict["normal_gamma"] = []
        reg_dict["noreg"] = []
        reg_dict["output"] = []
        reg_dict["output_noreg"] = []

        reg_dict["input"].append(self.conv_spatial.weight)
        reg_dict["input"].append(self.linear_global.weight)
        if self.metadata_encoder is not None:
            self.metadata_encoder.add_reg_dict(reg_dict)
        for block in self.blocks:
            block.add_reg_dict(reg_dict)
        if self.gab_template_mlp is not None:
            self.gab_template_mlp.add_reg_dict(reg_dict)
        if self.tab_module is not None:
            self.tab_module.add_reg_dict(reg_dict)
        if self.use_trunk_channel_gate:
            for gate_logit in self.trunk_channel_gate_logits:
                reg_dict["normal_gamma"].append(gate_logit)
        self.norm_trunkfinal.add_reg_dict(reg_dict)
        self.policy_head.add_reg_dict(reg_dict)
        self.value_head.add_reg_dict(reg_dict)
        if self.has_intermediate_head:
            self.norm_intermediate_trunkfinal.add_reg_dict(reg_dict)
            self.intermediate_policy_head.add_reg_dict(reg_dict)
            self.intermediate_value_head.add_reg_dict(reg_dict)


    def set_brenorm_params(self, renorm_avg_momentum: float, rmax: float, dmax: float):
        for block in self.blocks:
            block.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        self.norm_trunkfinal.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        self.policy_head.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        self.value_head.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
        if self.has_intermediate_head:
            self.norm_intermediate_trunkfinal.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
            self.intermediate_policy_head.set_brenorm_params(renorm_avg_momentum, rmax, dmax)
            self.intermediate_value_head.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        for block in self.blocks:
            block.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        self.norm_trunkfinal.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        self.policy_head.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        self.value_head.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
        if self.has_intermediate_head:
            self.norm_intermediate_trunkfinal.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
            self.intermediate_policy_head.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
            self.intermediate_value_head.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)

    def _channel_gated_add(self, trunk, residual, block_idx, mask, mask_sum_hw):
        """Add residual to trunk with per-channel gate.

        The gate logits are static (1, C, 1, 1) learned params initialized to zero.
        """
        gate_logit = 0.5 * self.trunk_channel_gate_logits[block_idx]
        w = ((block_idx+2) / (block_idx+1)) / ((1.0 / (block_idx+1)) + torch.exp(-gate_logit))
        trunk_factor = (1.0/(block_idx+1)) * ((block_idx+2) - w)
        residual_factor = w
        return trunk_factor * trunk + residual_factor * residual

    # Returns a tuple of tuples of outputs
    # The outer tuple indexes different sets of heads, such as if the net also computes intermediate heads.
    #   0 is the main output, 1 is intermediate.
    # The inner tuple ranges over the outputs of a set of heads (policy, value, etc).
    def forward(
        self,
        input_spatial,
        input_global,
        input_meta = None,
        extra_outputs: Optional[ExtraOutputs] = None,
    ):
        # float_formatter = "{:.3f}".format
        # np.set_printoptions(formatter={'float_kind':float_formatter}, threshold=1000000, linewidth=10000)

        mask = input_spatial[:, 0:1, :, :].contiguous()
        mask_sum_hw = torch.sum(mask,dim=(2,3),keepdim=True)
        mask_sum = torch.sum(mask)

        x_spatial = self.conv_spatial(input_spatial)
        x_global = self.linear_global(input_global).unsqueeze(-1).unsqueeze(-1)

        out = x_spatial + x_global

        if self.metadata_encoder is not None:
            assert input_meta is not None
            x_meta = self.metadata_encoder.forward(input_meta,extra_outputs)
            out = out + x_meta.unsqueeze(-1).unsqueeze(-1)

        # print("TENSOR BEFORE TRUNK")
        # print(out)

        # Compute shared block data
        block_shared_data = {}
        if self.gab_template_mlp is not None:
            seq_len = mask.shape[2] * mask.shape[3]  # H * W
            templates = self.gab_template_mlp(seq_len)
            block_shared_data[GAB_TEMPLATES] = GABTemplateData(templates=templates)
        if self.tab_module is not None:
            tab_keys, tab_queries = self.tab_module(out, mask)
            block_shared_data[TAB_KQ] = TABKeyQueryData(keys=tab_keys, queries=tab_queries)

        if self.has_intermediate_head:
            count = 0
            for i, block in enumerate(self.blocks[:self.intermediate_head_blocks]):
                residual = block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs, block_shared_data=block_shared_data)
                if self.use_trunk_channel_gate:
                    out = self._channel_gated_add(out, residual, i, mask, mask_sum_hw)
                else:
                    out = out + residual
                count += 1

            iout = out
            iout = self.norm_intermediate_trunkfinal(iout, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
            iout = self.act_intermediate_trunkfinal(iout)
            # Use fp32 for output heads to handle potentially large values
            with autocast("cuda", enabled=False):
                iout_fp32 = iout.float()
                mask_fp32 = mask.float()
                mask_sum_hw_fp32 = mask_sum_hw.float()
                mask_sum_fp32 = mask_sum.float() if isinstance(mask_sum, torch.Tensor) else mask_sum
                input_global_fp32 = input_global.float()

                iout_policy = self.intermediate_policy_head(
                    iout_fp32,
                    mask=mask_fp32,
                    mask_sum_hw=mask_sum_hw_fp32,
                    mask_sum=mask_sum_fp32,
                    extra_outputs=extra_outputs
                )
                (
                    iout_value,
                    iout_miscvalue,
                    iout_moremiscvalue,
                    iout_ownership,
                    iout_scoring,
                    iout_futurepos,
                    iout_seki,
                    iout_scorebelief_logprobs,
                ) = self.intermediate_value_head(
                    iout_fp32,
                    mask=mask_fp32,
                    mask_sum_hw=mask_sum_hw_fp32,
                    mask_sum=mask_sum_fp32,
                    input_global=input_global_fp32,
                    extra_outputs=extra_outputs
                )

            for i, block in enumerate(self.blocks[self.intermediate_head_blocks:], start=self.intermediate_head_blocks):
                residual = block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs, block_shared_data=block_shared_data)
                if self.use_trunk_channel_gate:
                    out = self._channel_gated_add(out, residual, i, mask, mask_sum_hw)
                else:
                    out = out + residual
                count += 1

        else:
            for i, block in enumerate(self.blocks):
                residual = block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=extra_outputs, block_shared_data=block_shared_data)
                if self.use_trunk_channel_gate:
                    out = self._channel_gated_add(out, residual, i, mask, mask_sum_hw)
                else:
                    out = out + residual

        out = self.norm_trunkfinal(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)
        out = self.act_trunkfinal(out)

        if extra_outputs is not None:
            extra_outputs.report("trunkfinal", out)

        # print("MAIN")
        # Disable autocast for main output heads - compute in fp32
        with autocast("cuda", enabled=False):
            out = out.float()
            mask_fp32 = mask.float()
            mask_sum_hw_fp32 = mask_sum_hw.float()
            mask_sum_fp32 = mask_sum.float() if isinstance(mask_sum, torch.Tensor) else mask_sum
            input_global_fp32 = input_global.float()

            out_policy = self.policy_head(
                out,
                mask=mask_fp32,
                mask_sum_hw=mask_sum_hw_fp32,
                mask_sum=mask_sum_fp32,
                extra_outputs=extra_outputs
            )
            (
                out_value,
                out_miscvalue,
                out_moremiscvalue,
                out_ownership,
                out_scoring,
                out_futurepos,
                out_seki,
                out_scorebelief_logprobs,
            ) = self.value_head(
                out,
                mask=mask_fp32,
                mask_sum_hw=mask_sum_hw_fp32,
                mask_sum=mask_sum_fp32,
                input_global=input_global_fp32,
                extra_outputs=extra_outputs
            )

        if self.has_intermediate_head:
            return (
                (
                    out_policy,
                    out_value,
                    out_miscvalue,
                    out_moremiscvalue,
                    out_ownership,
                    out_scoring,
                    out_futurepos,
                    out_seki,
                    out_scorebelief_logprobs,
                ),
                (
                    iout_policy,
                    iout_value,
                    iout_miscvalue,
                    iout_moremiscvalue,
                    iout_ownership,
                    iout_scoring,
                    iout_futurepos,
                    iout_seki,
                    iout_scorebelief_logprobs,
                ),
            )
        else:
            return ((
                out_policy,
                out_value,
                out_miscvalue,
                out_moremiscvalue,
                out_ownership,
                out_scoring,
                out_futurepos,
                out_seki,
                out_scorebelief_logprobs,
            ),)

    def float32ify_output(self, outputs_byheads):
        return tuple(self.float32ify_single_heads_output(outputs) for outputs in outputs_byheads)

    def float32ify_single_heads_output(self, outputs):
        (
            out_policy,
            out_value,
            out_miscvalue,
            out_moremiscvalue,
            out_ownership,
            out_scoring,
            out_futurepos,
            out_seki,
            out_scorebelief_logprobs,
        ) = outputs
        return (
            out_policy.to(torch.float32),
            out_value.to(torch.float32),
            out_miscvalue.to(torch.float32),
            out_moremiscvalue.to(torch.float32),
            out_ownership.to(torch.float32),
            out_scoring.to(torch.float32),
            out_futurepos.to(torch.float32),
            out_seki.to(torch.float32),
            out_scorebelief_logprobs.to(torch.float32),
        )

    def postprocess_output(self, outputs_byheads):
        return tuple(self.postprocess_single_heads_output(outputs) for outputs in outputs_byheads)

    def postprocess_single_heads_output(self, outputs):
        (
            out_policy,
            out_value,
            out_miscvalue,
            out_moremiscvalue,
            out_ownership,
            out_scoring,
            out_futurepos,
            out_seki,
            out_scorebelief_logprobs,
        ) = outputs

        policy_logits = out_policy
        value_logits = out_value
        td_value_logits = torch.stack((out_miscvalue[:,4:7], out_miscvalue[:,7:10], out_moremiscvalue[:,2:5]), dim=1)
        pred_td_score = out_moremiscvalue[:,5:8] * self.td_score_multiplier
        ownership_pretanh = out_ownership
        pred_scoring = out_scoring
        futurepos_pretanh = out_futurepos
        seki_logits = out_seki
        pred_scoremean = out_miscvalue[:, 0] * self.scoremean_multiplier
        pred_scorestdev = SoftPlusWithGradientFloorFunction.apply(out_miscvalue[:, 1], 0.05, False) * self.scorestdev_multiplier
        pred_lead = out_miscvalue[:, 2] * self.lead_multiplier
        pred_variance_time = SoftPlusWithGradientFloorFunction.apply(out_miscvalue[:, 3], 0.05, False) * self.variance_time_multiplier
        if self.config["version"] < 14:
            pred_shortterm_value_error = SoftPlusWithGradientFloorFunction.apply(out_moremiscvalue[:,0], 0.05, False) * self.shortterm_value_error_multiplier
            pred_shortterm_score_error = SoftPlusWithGradientFloorFunction.apply(out_moremiscvalue[:,1], 0.05, False) * self.shortterm_score_error_multiplier
        else:
            pred_shortterm_value_error = SoftPlusWithGradientFloorFunction.apply(out_moremiscvalue[:,0], 0.05, True) * self.shortterm_value_error_multiplier
            pred_shortterm_score_error = SoftPlusWithGradientFloorFunction.apply(out_moremiscvalue[:,1], 0.05, True) * self.shortterm_score_error_multiplier
        scorebelief_logits = out_scorebelief_logprobs

        return (
            policy_logits,      # N, num_policy_outputs, move
            value_logits,       # N, {win,loss,noresult}
            td_value_logits,    # N, {long, mid, short} {win,loss,noresult}
            pred_td_score,      # N, {long, mid, short}
            ownership_pretanh,  # N, 1, y, x
            pred_scoring,       # N, 1, y, x
            futurepos_pretanh,  # N, 2, y, x
            seki_logits,        # N, 4, y, x
            pred_scoremean,     # N
            pred_scorestdev,    # N
            pred_lead,          # N
            pred_variance_time, # N
            pred_shortterm_value_error, # N
            pred_shortterm_score_error, # N
            scorebelief_logits, # N, 2 * (self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS)
        )
