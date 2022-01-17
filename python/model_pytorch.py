import math
import torch
import torch.nn
import torch.nn.functional
import torch.nn.init
import packaging
from typing import List, Dict

import modelconfigs

EXTRA_SCORE_DISTR_RADIUS = 60

def act(activation, inplace=False):
    if activation == "relu":
        return torch.nn.ReLU(inplace=inplace)
    if activation == "hardswish":
        if packaging.version.parse(torch.__version__) > packaging.version.parse("1.6.0"):
            return torch.nn.Hardswish(inplace=inplace)
        else:
            return torch.nn.Hardswish()
    if activation == "identity":
        return torch.nn.Identity()
    assert False, f"Unknown activation name: {activation}"

def init_weights(tensor, activation, scale, fan_tensor=None):
    if activation == "relu" or activation == "hardswish":
        gain = math.sqrt(2.0)
    elif activation == "identity":
        gain = 1.0
    else:
        assert False, f"Unknown activation name: {activation}"

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


class NormMask(torch.nn.Module):
    def __init__(
        self,
        c_in,
        config: modelconfigs.ModelConfig,
        fixup_use_gamma,
    ):
        super(NormMask, self).__init__()
        self.norm_kind = config["norm_kind"]
        self.epsilon = config["bnorm_epsilon"]
        self.running_avg_momentum = config["bnorm_running_avg_momentum"]
        self.fixup_use_gamma = fixup_use_gamma
        self.c_in = c_in

        if norm_kind == "bnorm":
            self.beta = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
            self.register_buffer(
                "running_mean", torch.zeros(c_in, dtype=torch.float)
            )
            self.register_buffer(
                "running_std", torch.ones(c_in, dtype=torch.float)
            )
        elif norm_kind == "fixup":
            self.beta = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
            if fixup_use_gamma:
                self.gamma = torch.nn.Parameter(torch.ones(1, c_in, 1, 1))
        else:
            assert False, f"Unimplemented norm_kind: {norm_kind}"

    def add_reg_dict(self, reg_dict:Dict[str,List], is_last_batchnorm=False):
        if self.norm_kind == "fixup" and self.fixup_use_gamma:
            reg_dict["normal"].append(self.gamma)
        if is_last_batchnorm:
            reg_dict["output_noreg"].append(self.beta)
        else:
            reg_dict["noreg"].append(self.beta)

    def forward(self, x, mask, mask_sum: float):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum: scalar

        Returns: NCHW
        """

        if self.norm_kind == "bnorm":
            assert x.shape[1] == self.c_in
            if self.training:
                # This is the mean, computed only over exactly the areas of the mask, weighting each spot equally,
                # even across different elements in the batch that might have different board sizes.
                mean = torch.sum(x * mask, dim=(0,2,3),keepdim=True) / mask_sum
                zeromean_x = x - mean
                # Similarly, the variance computed exactly only over those spots
                var = torch.sum(torch.square(zeromean_x * mask),dim=(0,2,3),keepdim=True) / mask_sum
                std = torch.sqrt(var + self.epsilon)

                self.running_mean += self.running_avg_momentum * (mean.view(self.c_in).detach() - self.running_mean)
                self.running_std += self.running_avg_momentum * (std.view(self.c_in).detach() - self.running_std)

                return (zeromean_x / std + self.beta) * mask
            else:
                return ((x - self.running_mean.view(1,self.c_in,1,1)) / self.running_std.view(1,self.c_in,1,1) + self.beta) * mask

        elif self.norm_kind == "fixup":
            return (x + self.beta) * mask

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

        layer_mean = torch.sum(x, dim=(2, 3), keepdim=True) / mask_sum_hw
        (layer_max,_argmax) = torch.max(x.view(x.shape[0],x.shape[1],-1), dim=2)
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

        layer_mean = torch.sum(x, dim=(2, 3), keepdim=True) / mask_sum_hw

        out_pool1 = layer_mean
        out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0)
        out_pool3 = layer_mean * ((mask_sum_hw_sqrt_offset * mask_sum_hw_sqrt_offset) / 100.0 - 0.1)

        out = torch.cat((out_pool1, out_pool2, out_pool3), dim=1)
        return out

class ResBlock(torch.nn.Module):
    def __init__(self, name, c_in, c_mid, config, activation, num_total_blocks):
        super(ResBlock, self).__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.activation = activation
        self.num_total_blocks = num_total_blocks
        self.norm1 = NormMask(
            c_in,
            config=config,
            fixup_use_gamma=False,
        )
        self.act1 = act(activation, inplace=True)
        self.conv1 = torch.nn.Conv2d(c_in, c_mid, kernel_size=3, padding="same", bias=False)
        self.norm2 = NormMask(
            c_mid,
            config=config,
            fixup_use_gamma=True,
        )
        self.act2 = act(activation, inplace=True)
        self.conv2 = torch.nn.Conv2d(c_mid, c_in, kernel_size=3, padding="same", bias=False)

    def initialize(self):
        if self.norm_kind == "fixup":
            init_weights(self.conv1.weight, self.activation, scale=1.0/math.sqrt(self.num_total_blocks))
            init_weights(self.conv2.weight, self.activation, 0.0)
        else:
            init_weights(self.conv1.weight, self.activation, scale=1.0)
            init_weights(self.conv2.weight, self.activation, scale=1.0)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        reg_dict["normal"].append(self.conv1.weight)
        reg_dict["normal"].append(self.conv2.weight)
        self.norm1.add_reg_dict(reg_dict)
        self.norm2.add_reg_dict(reg_dict)

    def forward(self, x, mask, mask_sum_hw, mask_sum: float):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        out = x
        out = self.norm1(out, mask=mask, mask_sum=mask_sum)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.norm2(out, mask=mask, mask_sum=mask_sum)
        out = self.act2(out)
        out = self.conv2(out)
        return x + out


class GPoolResBlock(torch.nn.Module):
    def __init__(self, name, c_in, c_regular, c_gpool, config, activation, num_total_blocks):
        super(GPoolResBlock, self).__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.activation = activation
        self.num_total_blocks = num_total_blocks
        self.norm1 = NormMask(
            c_in,
            config=config,
            fixup_use_gamma=False,
        )
        self.act1 = act(activation, inplace=True)
        self.conv1r = torch.nn.Conv2d(c_in, c_regular, kernel_size=3, padding="same", bias=False)
        self.conv1g = torch.nn.Conv2d(c_in, c_gpool, kernel_size=3, padding="same", bias=False)
        self.normg = NormMask(
            c_gpool,
            config=config,
            fixup_use_gamma=False,
        )
        self.actg = act(activation, inplace=True)
        self.gpool = KataGPool()
        self.linear_g = torch.nn.Linear(3 * c_gpool, c_regular, bias=False)
        self.norm2 = NormMask(
            c_regular,
            config=config,
            fixup_use_gamma=True,
        )
        self.act2 = act(activation, inplace=True)
        self.conv2 = torch.nn.Conv2d(c_regular, c_in, kernel_size=3, padding="same", bias=False)

    def initialize(self):
        # Scaling so that variance on the r and g branches adds up to 1.0
        r_scale = 0.8
        g_scale = 0.6
        if self.norm_kind == "fixup":
            init_weights(self.conv1r.weight, self.activation, scale=1.0/math.sqrt(self.num_total_blocks) * r_scale)
            init_weights(self.conv1g.weight, self.activation, scale=1.0/math.sqrt(math.sqrt(self.num_total_blocks)) * math.sqrt(g_scale))
            init_weights(self.linear_g.weight, self.activation, scale=1.0/math.sqrt(math.sqrt(self.num_total_blocks)) * math.sqrt(g_scale))
            init_weights(self.conv2.weight, self.activation, 0.0)
        else:
            init_weights(self.conv1r.weight, self.activation, scale=r_scale)
            init_weights(self.conv1g.weight, self.activation, scale=1.0)
            init_weights(self.linear_g.weight, self.activation, scale=g_scale)
            init_weights(self.conv2.weight, self.activation, scale=1.0)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        reg_dict["normal"].append(self.conv1r.weight)
        reg_dict["normal"].append(self.conv1g.weight)
        reg_dict["normal"].append(self.linear_g.weight)
        reg_dict["normal"].append(self.conv2.weight)
        self.norm1.add_reg_dict(reg_dict)
        self.normg.add_reg_dict(reg_dict)
        self.norm2.add_reg_dict(reg_dict)

    def forward(self, x, mask, mask_sum_hw, mask_sum:float):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        out = x
        out = self.norm1(out, mask=mask, mask_sum=mask_sum)
        out = self.act1(out)

        outr = self.conv1r(out)
        outg = self.conv1g(out)

        outg = self.normg(outg, mask=mask, mask_sum=mask_sum)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask=mask, mask_sum_hw=mask_sum_hw).squeeze(-1).squeeze(-1)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)

        out = outr + outg

        out = self.norm2(out, mask=mask, mask_sum=mask_sum)
        out = self.act2(out)
        out = self.conv2(out)
        return x + out


class PolicyHead(torch.nn.Module):
    def __init__(self, c_in, c_p1, c_g1, config, activation):
        super(PolicyHead, self).__init__()
        self.norm_kind = config["norm_kind"]
        self.activation = activation

        self.conv1p = torch.nn.Conv2d(c_in, c_p1, kernel_size=1, padding="same", bias=False)
        self.conv1g = torch.nn.Conv2d(c_in, c_g1, kernel_size=1, padding="same", bias=False)

        self.normg = NormMask(
            c_g1,
            config=config,
            fixup_use_gamma=False,
        )
        self.actg = act(activation)
        self.gpool = KataGPool()

        self.linear_g = torch.nn.Linear(3 * c_g1, c_p1, bias=False)
        self.linear_pass = torch.nn.Linear(3 * c_g1, 2, bias=False)

        self.norm2 = NormMask(
            c_p1,
            config=config,
            fixup_use_gamma=False,
        )
        self.act2 = act(activation)
        self.conv2p = torch.nn.Conv2d(c_p1, 2, kernel_size=1, padding="same", bias=False)


    def initialize(self):
        # Scaling so that variance on the p and g branches adds up to 1.0
        p_scale = 0.8
        g_scale = 0.6
        # Extra scaling for outputs
        scale_output = 0.3
        init_weights(self.conv1p.weight, self.activation, scale=p_scale)
        init_weights(self.conv1g.weight, self.activation, scale=1.0)
        init_weights(self.linear_g.weight, self.activation, scale=g_scale)
        init_weights(self.linear_pass.weight, "identity", scale=scale_output)
        init_weights(self.conv2p.weight, "identity", scale=scale_output)

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        reg_dict["normal"].append(self.conv1p.weight)
        reg_dict["normal"].append(self.conv1g.weight)
        reg_dict["normal"].append(self.linear_g.weight)
        reg_dict["output"].append(self.linear_pass.weight)
        reg_dict["output"].append(self.conv2p.weight)
        self.normg.add_reg_dict(reg_dict,is_last_batchnorm=True)
        self.norm2.add_reg_dict(reg_dict,is_last_batchnorm=True)

    def forward(self, x, mask, mask_sum_hw, mask_sum:float):
        outp = self.conv1p(x)
        outg = self.conv1g(x)

        outg = self.normg(outg, mask=mask, mask_sum=mask_sum)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask=mask, mask_sum_hw=mask_sum_hw).squeeze(-1).squeeze(-1) # NC

        outpass = self.linear_pass(outg) # NC
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1) # NCHW

        outp = outp + outg
        outp = self.norm2(outp, mask=mask, mask_sum=mask_sum)
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
        self.norm1 = NormMask(
            c_v1,
            config=config,
            fixup_use_gamma=False,
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
        ))
        self.register_buffer("score_belief_offset_bias_vector", torch.tensor(
            data=[0.05 * (float(i-self.scorebelief_mid)+0.5) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
            requires_grad=False,
        ))
        self.register_buffer("score_belief_parity_vector", torch.tensor(
            [0.5-float((i-self.scorebelief_mid) % 2) for i in range(self.scorebelief_len)],
            dtype=torch.float32,
            requires_grad=False,
        ))


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
        reg_dict["normal"].append(self.conv1.weight)
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
        self.norm1.add_reg_dict(reg_dict,is_last_batchnorm=True)

    def forward(self, x, mask, mask_sum_hw, mask_sum:float, input_global):
        outv1 = x
        outv1 = self.conv1(outv1)
        outv1 = self.norm1(outv1, mask=mask, mask_sum=mask_sum)
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

class Model(torch.nn.Module):
    def __init__(self, config: modelconfigs.ModelConfig, pos_len: int):
        super(Model, self).__init__()

        self.config = config
        self.norm_kind = config["norm_kind"]
        self.block_kind = config["block_kind"]
        self.c_trunk = config["trunk_num_channels"]
        self.c_mid = config["mid_num_channels"]
        self.c_regular = config["regular_num_channels"]
        self.c_dilated = config["dilated_num_channels"]
        self.c_gpool = config["gpool_num_channels"]
        self.c_p1 = config["p1_num_channels"]
        self.c_g1 = config["g1_num_channels"]
        self.c_v1 = config["v1_num_channels"]
        self.c_v2 = config["v2_size"]
        self.c_sv2 = config["sbv2_num_channels"]
        self.num_scorebeliefs = config["num_scorebeliefs"]
        self.num_total_blocks = len(self.block_kind)
        self.pos_len = pos_len

        assert config["use_initial_conv_3"], "use_initial_conv_3 must be true"
        assert config["support_japanese_rules"], "support_jp_rules must be true"

        self.activation = "relu"

        self.conv_spatial = torch.nn.Conv2d(22, self.c_trunk, kernel_size=3, padding="same", bias=False)
        self.linear_global = torch.nn.Linear(19, self.c_trunk, bias=False)

        self.blocks = torch.nn.ModuleList()
        for block_config in self.block_kind:
            block_name = block_config[0]
            block_kind = block_config[1]
            if block_config[1] == "regular":
                self.blocks.append(ResBlock(
                    name=block_name,
                    c_in=self.c_trunk,
                    c_mid=self.c_mid,
                    config=self.config,
                    activation=self.activation,
                    num_total_blocks=self.num_total_blocks,
                ))
            elif block_config[1] == "gpool":
                self.blocks.append(GPoolResBlock(
                    block_name,
                    c_in=self.c_trunk,
                    c_regular=self.c_regular,
                    c_gpool=self.c_gpool,
                    config=self.config,
                    activation=self.activation,
                    num_total_blocks=self.num_total_blocks,
                ))
            else:
                assert False

        self.norm_trunkfinal = NormMask(self.c_trunk, self.config, fixup_use_gamma=False)
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

    def initialize(self):
        with torch.no_grad():
            spatial_scale = 0.8
            global_scale = 0.6
            init_weights(self.conv_spatial.weight, self.activation, scale=spatial_scale)
            init_weights(self.linear_global.weight, self.activation, scale=global_scale)

            for block in self.blocks:
                block.initialize()
            self.policy_head.initialize()
            self.value_head.initialize()

    def get_norm_kind(self) -> bool:
        return self.norm_kind

    def add_reg_dict(self, reg_dict:Dict[str,List]):
        reg_dict["normal"] = []
        reg_dict["output"] = []
        reg_dict["noreg"] = []
        reg_dict["output_noreg"] = []

        reg_dict["normal"].append(self.conv_spatial.weight)
        reg_dict["normal"].append(self.linear_global.weight)
        for block in self.blocks:
            block.add_reg_dict(reg_dict)
        self.norm_trunkfinal.add_reg_dict(reg_dict)
        self.policy_head.add_reg_dict(reg_dict)
        self.value_head.add_reg_dict(reg_dict)

    def forward(self, input_spatial, input_global):
        mask = input_spatial[:, 0:1, :, :].contiguous()
        mask_sum_hw = torch.sum(mask,dim=(2,3),keepdim=True)
        mask_sum = torch.sum(mask)

        x_spatial = self.conv_spatial(input_spatial)
        x_global = self.linear_global(input_global).unsqueeze(-1).unsqueeze(-1)
        out = x_spatial + x_global

        for block in self.blocks:
            out = block(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)

        out = self.norm_trunkfinal(out, mask=mask, mask_sum=mask_sum)
        out = self.act_trunkfinal(out)

        out_policy = self.policy_head(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum)

        (
            out_value,
            out_miscvalue,
            out_moremiscvalue,
            out_ownership,
            out_scoring,
            out_futurepos,
            out_seki,
            out_scorebelief_logprobs,
        ) = self.value_head(out, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, input_global=input_global)

        return (
            out_policy,
            out_value,
            out_miscvalue,
            out_moremiscvalue,
            out_ownership,
            out_scoring,
            out_futurepos,
            out_seki,
            out_scorebelief_logprobs,
        )


    def postprocess_output(self, outputs):
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
        pred_td_score = out_moremiscvalue[:,5:8] * 20.0
        ownership_pretanh = out_ownership
        pred_scoring = out_scoring
        futurepos_pretanh = out_futurepos
        seki_logits = out_seki
        pred_scoremean = out_miscvalue[:, 0] * 20.0
        pred_scorestdev = torch.nn.functional.softplus(out_miscvalue[:, 1]) * 20.0
        pred_lead = out_miscvalue[:, 2] * 20.0
        pred_variance_time = torch.nn.functional.softplus(out_miscvalue[:, 3]) * 40.0
        pred_shortterm_value_error = torch.nn.functional.softplus(out_moremiscvalue[:,0]) * 0.25
        pred_shortterm_score_error = torch.nn.functional.softplus(out_moremiscvalue[:,1]) * 30.0
        scorebelief_logits = out_scorebelief_logprobs

        return (
            policy_logits,
            value_logits,
            td_value_logits,
            pred_td_score,
            ownership_pretanh,
            pred_scoring,
            futurepos_pretanh,
            seki_logits,
            pred_scoremean,
            pred_scorestdev,
            pred_lead,
            pred_variance_time,
            pred_shortterm_value_error,
            pred_shortterm_score_error,
            scorebelief_logits,
        )

