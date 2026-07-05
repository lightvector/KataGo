#!/usr/bin/python3
import sys
import os
import argparse
import traceback
import random
import math
import time
import struct
import json
import datetime
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn
from torch.optim.swa_utils import AveragedModel
from katago.train.model_pytorch import RMSNormMask

from katago.train import modelconfigs
from katago.train.model_pytorch import Model, ResBlock, NestedBottleneckResBlock, TransformerAttentionBlock, TransformerFFNBlock, NestedBottleneckTransformerBlock
from katago.train.load_model import load_model

#Command and args-------------------------------------------------------------------

description = """
Export neural net weights to file for KataGo engine.
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('-checkpoint', help='Checkpoint to test', required=False)
parser.add_argument('-export-random-initialized-model', help='Instead of loading a checkpoint, export a freshly random-initialized model of the given model config name (e.g. b15c512h8nbttflrs-fson-silu-rsnh)', required=False)
parser.add_argument('-export-dir', help='model file dir to save to', required=True)
parser.add_argument('-model-name', help='name to record in model file', required=True)
parser.add_argument('-filename-prefix', help='filename prefix to save to within dir', required=True)
parser.add_argument('-use-swa', help='Use SWA model', action="store_true", required=False)
parser.add_argument('-export-14-as-15', help='Export model version 14 as 15', action="store_true", required=False)
parser.add_argument('-export-15-or-16-as-17', help='Export model version 15 or 16 as 17', action="store_true", required=False)
args = vars(parser.parse_args())


def main(args):
    checkpoint_file = args["checkpoint"]
    export_random_initialized_model = args["export_random_initialized_model"]
    export_dir = args["export_dir"]
    model_name = args["model_name"]
    filename_prefix = args["filename_prefix"]
    use_swa = args["use_swa"]
    export_14_as_15 = args["export_14_as_15"]
    export_15_or_16_as_17 = args["export_15_or_16_as_17"]

    if (checkpoint_file is None) == (export_random_initialized_model is None):
        raise Exception("Exactly one of -checkpoint or -export-random-initialized-model must be specified")
    if export_random_initialized_model is not None and use_swa:
        raise Exception("-use-swa cannot be used with -export-random-initialized-model")

    os.makedirs(export_dir,exist_ok=True)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(export_dir + "/log.txt"),
        ],
    )
    np.set_printoptions(linewidth=150)

    logging.info(str(sys.argv))

    # LOAD MODEL ---------------------------------------------------------------------
    if export_random_initialized_model is not None:
        if export_random_initialized_model not in modelconfigs.config_of_name:
            raise Exception(f"Unknown model config name: {export_random_initialized_model}")
        model_config = modelconfigs.config_of_name[export_random_initialized_model]
        logging.info(f"Exporting freshly random-initialized model with config: {export_random_initialized_model}")
        logging.info(str(model_config))
        model = Model(model_config, pos_len=19)
        model.initialize()
        model.to("cpu")
        swa_model = None
        other_state_dict = {}
    else:
        model, swa_model, other_state_dict = load_model(checkpoint_file, use_swa, device="cpu", verbose=True)
    model_config = model.config

    # WRITING MODEL ----------------------------------------------------------------
    extension = ".bin"
    mode = "wb"
    f = open(export_dir + "/" + filename_prefix + extension, mode)
    def writeln(s):
        f.write((str(s)+"\n").encode(encoding="ascii",errors="backslashreplace"))
    def writestr(s):
        f.write(s.encode(encoding="ascii",errors="backslashreplace"))

    # Ignore what's in the config if less than 11 since a lot of testing models
    # are on old version but actually have various new architectures.
    version = max(model_config["version"],11)
    true_version = version
    # Hack to be able to export version 14 as version 15
    if version == 14 and export_14_as_15:
        version = 15
    elif export_14_as_15:
        raise Exception("export_14_as_15 specified but version was " + str(version))

    # Configs that use rmsnorm for trunk tip that claim version 15 or 16 automatically get upgraded
    # to version 17. Version 17 is a model format change that supports a strict superset of version 15 and 16,
    # representing q value predictions by outputting the expected policy outputs (2 or 4), having a few more
    # slots for unused params to allow for flexibility without future version bumps
    # Version 15 and/or 16 was used to train some experimental transformer with trunk tip rmsnorm
    # that we want to still be able to export and run, so we autoupgrade to version 17 for exporting them.
    if (version == 15 or version == 16) and export_15_or_16_as_17:
        version = 17
    elif export_15_or_16_as_17:
        raise Exception("export_15_or_16_as_17 specified but version was " + str(version))
    elif version == 15 or version == 16:
        if model.trunk_final_rmsnorm:
            logging.warn("Autoupgrading v15 or v16 model to v17 due to trunk rmsnorm")
            version = 17

    writeln(model_name)
    writeln(version)
    writeln(modelconfigs.get_num_bin_input_features(model_config))
    writeln(modelconfigs.get_num_global_input_features(model_config))

    if version <= 12:
        assert model.td_score_multiplier == 20.0
        assert model.scoremean_multiplier == 20.0
        assert model.scorestdev_multiplier == 20.0
        assert model.lead_multiplier == 20.0
        assert model.variance_time_multiplier == 40.0
        assert model.shortterm_value_error_multiplier == 0.25
        assert model.shortterm_score_error_multiplier == 30.0
    else:
        writeln(model.td_score_multiplier)
        writeln(model.scoremean_multiplier)
        writeln(model.scorestdev_multiplier)
        writeln(model.lead_multiplier)
        writeln(model.variance_time_multiplier)
        writeln(model.shortterm_value_error_multiplier)
        writeln(model.shortterm_score_error_multiplier)

    if version >= 15:
        if model.metadata_encoder is not None:
            writeln(model.metadata_encoder.meta_encoder_version)
        else:
            writeln(0)

        # Write some dummy placeholders for future features
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)
        writeln(0)


    def write_weights(weights):
        # Little endian
        reshaped = np.reshape(weights.detach().numpy(),[-1])
        num_weights = len(reshaped)
        writestr("@BIN@")
        f.write(struct.pack(f'<{num_weights}f',*reshaped))
        writestr("\n")

    def write_conv_weight(name,convweight):
        (out_channels, in_channels, diamy, diamx) = convweight.shape
        dilation = 1
        writeln(name)
        writeln(diamy) #y
        writeln(diamx) #x
        writeln(in_channels)
        writeln(out_channels)
        writeln(dilation) #y
        writeln(dilation) #x
        # Torch order is oc,ic,y,x
        # Desired output order is y,x,ic,oc
        write_weights(torch.permute(convweight,(2,3,1,0)))

    def write_conv(name,conv):
        assert conv.bias is None
        write_conv_weight(name, conv.weight)

    def write_bn(name,normmask):
        writeln(name)

        writeln(normmask.c_in)
        epsilon = 1e-20
        writeln(epsilon)
        has_gamma_or_scale = normmask.scale is not None or normmask.gamma is not None
        has_beta = True
        writeln(1 if has_gamma_or_scale else 0)
        writeln(1 if has_beta else 0)

        if hasattr(normmask,"running_mean") and normmask.running_mean is not None:
            assert normmask.is_using_batchnorm
            assert normmask.running_mean.shape == (normmask.c_in,)
            write_weights(normmask.running_mean)
        else:
            assert not normmask.is_using_batchnorm
            write_weights(torch.zeros(normmask.c_in, dtype=torch.float))

        if hasattr(normmask,"running_std") and normmask.running_std is not None:
            assert normmask.is_using_batchnorm
            assert normmask.running_std.shape == (normmask.c_in,)
            write_weights(torch.maximum(torch.tensor(1e-20), normmask.running_std * normmask.running_std - epsilon))
        else:
            assert not normmask.is_using_batchnorm
            write_weights((1.0-epsilon) * torch.ones(normmask.c_in, dtype=torch.float))

        if normmask.scale is not None:
            if normmask.gamma is not None:
                assert normmask.gamma.shape == (1, normmask.c_in, 1, 1)
                assert has_gamma_or_scale
                if normmask.gamma_weight_decay_center_1:
                    write_weights(normmask.scale * (normmask.gamma+1.0))
                else:
                    write_weights(normmask.scale * normmask.gamma)
            else:
                assert has_gamma_or_scale
                write_weights(normmask.scale * torch.ones(normmask.c_in, dtype=torch.float, device="cpu"))
        else:
            if normmask.gamma is not None:
                assert normmask.gamma.shape == (1, normmask.c_in, 1, 1)
                assert has_gamma_or_scale
                if normmask.gamma_weight_decay_center_1:
                    write_weights(normmask.gamma+1.0)
                else:
                    write_weights(normmask.gamma)
            else:
                assert not has_gamma_or_scale
                pass

        assert normmask.beta.shape == (1, normmask.c_in, 1, 1)
        write_weights(normmask.beta)

    def write_biasmask(name,biasmask):
        writeln(name)

        writeln(biasmask.c_in)
        epsilon = 1e-20
        writeln(epsilon)
        has_gamma_or_scale = biasmask.scale is not None
        has_beta = True
        writeln(1 if has_gamma_or_scale else 0)
        writeln(1 if has_beta else 0)

        write_weights(torch.zeros(biasmask.c_in, dtype=torch.float))
        write_weights((1.0-epsilon) * torch.ones(biasmask.c_in, dtype=torch.float))

        if biasmask.scale is not None:
            write_weights(biasmask.scale * torch.ones(biasmask.c_in, dtype=torch.float, device="cpu"))

        assert biasmask.beta.shape == (1, biasmask.c_in, 1, 1)
        write_weights(biasmask.beta)

    def write_activation(name, activation):
        writeln(name)
        if isinstance(activation,torch.nn.ReLU):
            writeln("ACTIVATION_RELU")
        elif isinstance(activation,torch.nn.Mish):
            writeln("ACTIVATION_MISH")
        elif isinstance(activation,torch.nn.SiLU):
            writeln("ACTIVATION_SILU")
        elif isinstance(activation,torch.nn.Identity):
            writeln("ACTIVATION_IDENTITY")
        else:
            assert False, f"Activation not supported for export: {activation}"


    def write_matmul(name,linearweight):
        writeln(name)
        (out_channels,in_channels) = linearweight.shape
        writeln(in_channels)
        writeln(out_channels)
        # Torch order is oc,ic
        # Desired output order is ic,oc
        write_weights(torch.permute(linearweight,(1,0)))

    def write_matbias(name,linearbias):
        writeln(name)
        (out_channels,) = linearbias.shape
        writeln(out_channels)
        write_weights(linearbias)

    def write_rmsnorm(name,rmsnormmask):
        writeln(name)
        writeln(rmsnormmask.c_in)
        writeln(rmsnormmask.eps)
        writeln(1 if rmsnormmask.spatial else 0)
        cgroup_size = rmsnormmask.cgroup_size if rmsnormmask.cgroup_size is not None else 0
        writeln(cgroup_size)
        if not rmsnormmask.spatial:
            # Non-spatial: weight comes from torch.nn.RMSNorm
            assert rmsnormmask.norm is not None
            assert rmsnormmask.norm.weight.shape == (rmsnormmask.c_in,)
            write_weights(rmsnormmask.norm.weight)
        else:
            # Spatial: weight comes from self.gamma
            assert rmsnormmask.gamma.shape == (rmsnormmask.c_in,)
            write_weights(rmsnormmask.gamma)
        assert rmsnormmask.beta.shape == (rmsnormmask.c_in,)
        write_weights(rmsnormmask.beta)

    def write_normactconv(name,normactconv):
        if normactconv.c_gpool is None:
            assert normactconv.convpool is None
            if normactconv.conv1x1 is None:
                write_bn(name+".norm", normactconv.norm)
                write_activation(name+".act", normactconv.act)
                write_conv(name+".conv", normactconv.conv)
            else:
                write_bn(name+".norm", normactconv.norm)
                write_activation(name+".act", normactconv.act)
                # Torch conv order is oc,ic,h,w
                # We want to add the 1x1 conv to the center of the h,w
                h,w = (normactconv.conv.weight.shape[2],normactconv.conv.weight.shape[3])
                assert h % 2 == 1, "Conv1x1 can't be merged with even-sized convolution kernel"
                assert w % 2 == 1, "Conv1x1 can't be merged with even-sized convolution kernel"
                combined_conv = normactconv.conv.weight.detach().clone()
                combined_conv[:,:,h//2:h//2+1,w//2:w//2+1] += normactconv.conv1x1.weight
                assert normactconv.conv.bias is None
                assert normactconv.conv1x1.bias is None
                write_conv_weight(name+".conv", combined_conv)
        else:
            assert normactconv.convpool is not None
            assert normactconv.conv1x1 is None
            write_bn(name+".norm", normactconv.norm)
            write_activation(name+".act", normactconv.act)
            write_conv(name+".convpool.conv1r", normactconv.convpool.conv1r)
            write_conv(name+".convpool.conv1g", normactconv.convpool.conv1g)
            write_bn(name+".convpool.normg", normactconv.convpool.normg)
            write_activation(name+".convpool.actg", normactconv.convpool.actg)
            write_matmul(name+".convpool.linear_g", normactconv.convpool.linear_g.weight)
            assert normactconv.convpool.linear_g.bias is None

    def write_transformer_norm(name,rmsnorm):
        """Write an inline RMSNorm (torch.nn.RMSNorm) used inside transformer blocks.
        These are simpler than the trunk-final RMSNormMask: just a weight vector, no bias, no spatial modes."""
        writeln(name)
        num_channels = rmsnorm.weight.shape[0]
        writeln(num_channels)
        writeln(rmsnorm.eps)
        write_weights(rmsnorm.weight)

    def write_transformer_attention_block(name,block):
        assert not getattr(block, 'use_qk_norm', False), \
            f"{name}: QK normalization is not yet supported for export"
        assert not getattr(block, 'use_gab', False), \
            f"{name}: GAB attention bias is not yet supported for export"
        assert not getattr(block, 'use_tab', False), \
            f"{name}: TAB attention bias is not yet supported for export"
        assert not getattr(block, 'inline_registers', False), \
            f"{name}: Inline registers are not yet supported for export"
        assert getattr(block, 'num_rw_registers', 0) == 0, \
            f"{name}: RW registers are not yet supported for export"

        writeln("transformer_attention_block")
        writeln(name)
        writeln(block.num_heads)
        writeln(block.num_kv_heads)
        writeln(block.q_head_dim)
        writeln(block.v_head_dim)
        writeln(1 if block.use_rope else 0)
        writeln(1 if block.learnable_rope else 0)

        write_transformer_norm(name+".norm1", block.norm1)
        write_matmul(name+".q_proj", block.q_proj.weight)
        write_matmul(name+".k_proj", block.k_proj.weight)
        write_matmul(name+".v_proj", block.v_proj.weight)
        write_matmul(name+".out_proj", block.out_proj.weight)

        if block.use_rope:
            if block.learnable_rope:
                # rope_freqs: (num_kv_heads, q_head_dim//2, 2)
                writeln(name+".rope_freqs")
                freqs = block.rope_freqs.detach()
                writeln(freqs.shape[0])  # num_kv_heads
                writeln(freqs.shape[1])  # num_pairs
                writeln(freqs.shape[2])  # 2 (omega_x, omega_y)
                write_weights(freqs)
            else:
                # Non-learnable: write theta so C++ can recompute
                writeln(name+".rope_theta")
                writeln(block.rope_theta)

    def write_transformer_ffn_block(name,block):
        assert not getattr(block, 'use_depthwise_conv', False), \
            f"{name}: FFN depthwise conv is not yet supported for export"
        assert not getattr(block, 'inline_registers', False), \
            f"{name}: Inline registers are not yet supported for export"
        assert getattr(block, 'num_rw_registers', 0) == 0, \
            f"{name}: RW registers are not yet supported for export"

        writeln("transformer_ffn_block")
        writeln(name)
        writeln(block.c_main)
        writeln(block.ffn_dim)
        writeln(1 if block.use_swiglu else 0)

        write_transformer_norm(name+".norm", block.norm)
        write_matmul(name+".ffn_linear1", block.ffn_linear1.weight)
        if block.use_swiglu:
            write_matmul(name+".ffn_linear_gate", block.ffn_linear_gate.weight)
        write_matmul(name+".ffn_linear2", block.ffn_linear2.weight)

    def write_block(name,block):
        if isinstance(block,ResBlock) and block.normactconv1.c_gpool is None:
            assert block.normactconv2.c_gpool is None
            writeln("ordinary_block")
            writeln(name)
            write_normactconv(name+".normactconv1", block.normactconv1)
            write_normactconv(name+".normactconv2", block.normactconv2)
        elif isinstance(block,ResBlock) and block.normactconv1.c_gpool is not None:
            assert block.normactconv2.c_gpool is None
            writeln("gpool_block")
            writeln(name)
            write_normactconv(name+".normactconv1", block.normactconv1)
            write_normactconv(name+".normactconv2", block.normactconv2)
        elif isinstance(block,NestedBottleneckResBlock):
            writeln("nested_bottleneck_block")
            writeln(name)
            writeln(block.internal_length)
            assert block.internal_length == len(block.blockstack)
            write_normactconv(name+".normactconvp", block.normactconvp)
            for i,subblock in enumerate(block.blockstack):
                write_block(name+".blockstack."+str(i),subblock)
            write_normactconv(name+".normactconvq", block.normactconvq)
        elif isinstance(block,TransformerAttentionBlock):
            write_transformer_attention_block(name, block)
        elif isinstance(block,TransformerFFNBlock):
            write_transformer_ffn_block(name, block)
        elif isinstance(block,NestedBottleneckTransformerBlock):
            writeln("nested_bottleneck_block")
            writeln(name)
            writeln(2 * block.internal_length)
            assert len(block.blockstack) == 2 * block.internal_length
            write_normactconv(name+".normactconvp", block.normactconvp)
            for i,subblock in enumerate(block.blockstack):
                write_block(name+".blockstack."+str(i), subblock)
            write_normactconv(name+".normactconvq", block.normactconvq)
        else:
            assert False, f"This kind of block is not supported for export right now: {type(block)}"

    def write_metadata_encoder(name,encoder):
        writeln(name)
        writeln(encoder.c_input)
        # Torch order is oc,ic. Flatten feature mask into the first mul
        write_matmul(name+".mul1", encoder.linear1.weight * encoder.feature_mask.reshape((1,-1)))
        write_matbias(name+".bias1", encoder.linear1.bias)
        write_activation(name+".act1", encoder.act1)
        write_matmul(name+".mul2", encoder.linear2.weight)
        write_matbias(name+".bias2", encoder.linear2.bias)
        write_activation(name+".act2", encoder.act2)
        write_matmul(name+".mul3", encoder.out_scale * encoder.linear_output_to_trunk.weight)
        assert encoder.linear_output_to_trunk.bias is None

    def write_trunk(name,model):
        writeln("trunk")
        writeln(len(model.blocks))
        writeln(model.c_trunk)
        writeln(model.c_mid)
        writeln(model.c_mid-model.c_gpool)
        writeln(model.c_gpool)
        writeln(model.c_gpool)
        if version >= 15:
            # Trunk final norm kind:
            # 0 = standard (batch norm or bias mask)
            # 1 = RMSNorm (non-spatial)
            # 2 = RMSNorm spatial (no groups)
            # 3 = RMSNorm spatial grouped
            if model.trunk_final_rmsnorm:
                assert isinstance(model.norm_trunkfinal, RMSNormMask)
                assert model.norm_trunkfinal.cgroup_size is None, \
                    "Grouped spatial RMSNorm (cgroup_size != None) is not supported for export"
                trunk_norm_kind = 1  # TRUNK_NORM_KIND_RMSNORM; spatial flag is in the RMSNormLayerDesc

                # Older katago c++ versions didn't actually do proper checking here and yet didn't support
                # this so it's not actually sound to output a non-zero trunk kind.
                # Version 17 loading code on c++ does do proper checking.
                assert version >= 17
            else:
                trunk_norm_kind = 0
            writeln(trunk_norm_kind)
            # Write some dummy placeholders for future features
            writeln(0)
            writeln(0)
            writeln(0)
            writeln(0)
            writeln(0)

        write_conv("model.conv_spatial", model.conv_spatial)
        write_matmul("model.linear_global", model.linear_global.weight)
        assert model.linear_global.bias is None
        if model.metadata_encoder is not None:
            assert version >= 15
            write_metadata_encoder("model.sgf_metadata_encoder",model.metadata_encoder)

        for i,block in enumerate(model.blocks):
            write_block("model.blocks."+str(i), block)
        if model.trunk_final_rmsnorm:
            write_rmsnorm("model.norm_trunkfinal", model.norm_trunkfinal)
        elif model.trunk_normless:
            write_biasmask("model.norm_trunkfinal", model.norm_trunkfinal)
        else:
            write_bn("model.norm_trunkfinal", model.norm_trunkfinal)
        write_activation("model.act_trunkfinal", model.act_trunkfinal)

    def write_policy_head(name,policyhead):
        writeln(name)
        if version >= 17:
            assert policyhead.conv2p.weight.shape[0] == 6 or policyhead.conv2p.weight.shape[0] == 8
            if policyhead.conv2p.weight.shape[0] == 6:
                writeln(2) # we're going to write 2 policy output channels - regular and optimistic (see below)
            else:
                writeln(4) # we're going to write 4 policy output channels - regular, optimistic, q winloss, q score (see below)
            # Write some dummy placeholders for future features
            writeln(0)
            writeln(0)
            writeln(0)
        write_conv(name+".conv1p", policyhead.conv1p)
        write_conv(name+".conv1g", policyhead.conv1g)
        write_biasmask(name+".biasg", policyhead.biasg)
        write_activation(name+".actg", policyhead.actg)
        write_matmul(name+".linear_g", policyhead.linear_g.weight)
        assert policyhead.linear_g.bias is None
        write_biasmask(name+".bias2", policyhead.bias2)
        write_activation(name+".act2", policyhead.act2)

        # Write the this-move prediction and the optimistic policy prediction
        if version <= 11:
            assert policyhead.conv2p.weight.shape[0] == 4
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0],), dim=0))
            assert policyhead.linear_pass.weight.shape[0] == 4
            write_matmul(name+".linear_pass", torch.stack((policyhead.linear_pass.weight[0],), dim=0))
            assert policyhead.linear_pass.bias is None
        elif version <= 14:
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5]), dim=0))
            assert policyhead.linear_pass.weight.shape[0] == 6
            write_matmul(name+".linear_pass", torch.stack((policyhead.linear_pass.weight[0], policyhead.linear_pass.weight[5]), dim=0))
            assert policyhead.linear_pass.bias is None
        elif version == 15 and true_version == 14:
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5]), dim=0))
            assert policyhead.linear_pass.weight.shape[0] == 6
            linear_pass_stack = [policyhead.linear_pass.weight[0], policyhead.linear_pass.weight[5]]
            c_p1 = int(policyhead.linear_g.weight.shape[0])
            for _ in range(c_p1-2):
                linear_pass_stack.append(torch.zeros_like(linear_pass_stack[0]))
            write_matmul(name+".linear_pass", torch.stack(linear_pass_stack, dim=0))
            assert policyhead.linear_pass.bias is None
            write_matbias(name+".linear_pass_bias", torch.tensor([0.0]*c_p1,dtype=torch.float32,device="cpu"))
            write_activation(name+".act_pass", torch.nn.Identity())
            write_matmul(name+".linear_pass2", torch.tensor([[1.0,0.0]+[0.0]*(c_p1-2),[0.0,1.0]+[0.0]*(c_p1-2)],dtype=torch.float32,device="cpu"))
        elif version <= 15 or policyhead.conv2p.weight.shape[0] == 6:
            assert policyhead.conv2p.weight.shape[0] == 6
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5]), dim=0))
            write_matmul(name+".linear_pass", policyhead.linear_pass.weight)
            write_matbias(name+".linear_pass_bias", policyhead.linear_pass.bias)
            write_activation(name+".act_pass", policyhead.act_pass)
            assert policyhead.linear_pass2.weight.shape[0] == 6
            write_matmul(name+".linear_pass2", torch.stack((policyhead.linear_pass2.weight[0], policyhead.linear_pass2.weight[5]), dim=0))
            assert policyhead.linear_pass2.bias is None
        else:
            assert policyhead.conv2p.weight.shape[0] == 8
            write_conv_weight(name+".conv2p", torch.stack((policyhead.conv2p.weight[0], policyhead.conv2p.weight[5], policyhead.conv2p.weight[6], policyhead.conv2p.weight[7]), dim=0))
            write_matmul(name+".linear_pass", policyhead.linear_pass.weight)
            write_matbias(name+".linear_pass_bias", policyhead.linear_pass.bias)
            write_activation(name+".act_pass", policyhead.act_pass)
            assert policyhead.linear_pass2.weight.shape[0] == 8
            write_matmul(name+".linear_pass2", torch.stack((policyhead.linear_pass2.weight[0], policyhead.linear_pass2.weight[5], policyhead.linear_pass2.weight[6], policyhead.linear_pass2.weight[7]), dim=0))
            assert policyhead.linear_pass2.bias is None

        assert policyhead.conv2p.bias is None


    def write_value_head(name, valuehead):
        writeln(name)
        if version >= 17:
            # Write some dummy placeholders for future features
            writeln(0)
            writeln(0)
            writeln(0)
        write_conv(name+".conv1", valuehead.conv1)
        write_biasmask(name+".bias1", valuehead.bias1)
        write_activation(name+".act1", valuehead.act1)
        write_matmul(name+".linear2", valuehead.linear2.weight)
        write_matbias(name+".bias2", valuehead.linear2.bias)
        write_activation(name+".act2", valuehead.act2)
        write_matmul(name+".linear_valuehead", valuehead.linear_valuehead.weight)
        write_matbias(name+".bias_valuehead", valuehead.linear_valuehead.bias)

        # For now, only output the scoremean and scorestdev and lead and vtime channels
        w = valuehead.linear_miscvaluehead.weight[0:4]
        b = valuehead.linear_miscvaluehead.bias[0:4]
        # Grab the shortterm channels
        w2 = valuehead.linear_moremiscvaluehead.weight[0:2]
        b2 = valuehead.linear_moremiscvaluehead.bias[0:2]
        w = torch.cat((w,w2),dim=0)
        b = torch.cat((b,b2),dim=0)
        write_matmul(name+".linear_miscvaluehead", w)
        write_matbias(name+".bias_miscvaluehead", b)

        write_conv(name+".conv_ownership",valuehead.conv_ownership)

    def write_model(model):
        write_trunk("model",model)
        write_policy_head("model.policy_head",model.policy_head)
        write_value_head("model.value_head",model.value_head)

    if swa_model is not None:
        logging.info("Writing SWA model")
        write_model(swa_model)
    else:
        logging.info("Writing model")
        write_model(model)
    f.close()

    with open(os.path.join(export_dir,"metadata.json"),"w") as f:
        train_state = other_state_dict.get("train_state", {})
        data = {}
        if "global_step_samples" in train_state:
            data["global_step_samples"] = train_state["global_step_samples"]
        if "total_num_data_rows" in train_state:
            data["total_num_data_rows"] = train_state["total_num_data_rows"]
        if "running_metrics" in other_state_dict:
            assert sorted(list(other_state_dict["running_metrics"].keys())) == ["sums", "weights"]
            data["extra_stats"] = {
                "sums": { key: value for (key,value) in other_state_dict["running_metrics"]["sums"].items() if "sopt" not in key and "lopt" not in key },
                "weights": { key: value for (key,value) in other_state_dict["running_metrics"]["weights"].items() if "sopt" not in key and "lopt" not in key },
            }
            if "last_val_metrics" in other_state_dict and "sums" in other_state_dict["last_val_metrics"] and "weights" in other_state_dict["last_val_metrics"]:
                data["extra_stats"]["last_val_metrics"] = {
                    "sums": { key: value for (key,value) in other_state_dict["last_val_metrics"]["sums"].items() if "sopt" not in key and "lopt" not in key },
                    "weights": { key: value for (key,value) in other_state_dict["last_val_metrics"]["weights"].items() if "sopt" not in key and "lopt" not in key },
                }
        json.dump(data,f)


    logging.info("Exported at: ")
    logging.info(str(datetime.datetime.utcnow()) + " UTC")

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main(args)
