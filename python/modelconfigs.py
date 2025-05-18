#!/usr/bin/python3
"""
This file contains a bunch of configs for models of different sizes.
See the bottom of this file "base_config_of_name" for a dictionary of all the different
base model architectures, and which ones are recommended of each different model size.

For each base model, additional configs are also pregenerated with different suffixes.

For example, for b10c384nbt, we also have models like:
b10c384nbt-mish  (use mish instead of relu)
b10c384nbt-bn-mish-rvgl (use batchnorm, mish, and repvgg-linear-style convolutions).

KataGo's main models for the distributed training run currently find the following to work
well or best: "-fson-mish-rvgl-bnh"
* Use fixed activation scale initialization + one batch norm for the whole net
* Mish activation
* Repvgg-linear-style convolutions
* Batch norm output head + non-batch-norm output head, where the former drives optimization
  but the latter is used for inference.
"""

from typing import Dict, Any, Union

ModelConfig = Dict[str,Any]

# version = 0 # V1 features, with old head architecture using crelus (no longer supported)
# version = 1 # V1 features, with new head architecture, no crelus
# version = 2 # V2 features, no internal architecture change.
# version = 3 # V3 features, selfplay-planned features with lots of aux targets
# version = 4 # V3 features, but supporting belief stdev and dynamic scorevalue
# version = 5 # V4 features, slightly different pass-alive stones feature
# version = 6 # V5 features, most higher-level go features removed
# version = 7 # V6 features, more rules support
# version = 8 # V7 features, asym, lead, variance time
# version = 9 # V7 features, shortterm value error prediction, inference actually uses variance time, unsupported now
# version = 10 # V7 features, shortterm value error prediction done properly
# version = 11 # V7 features, New architectures!
# version = 12 # V7 features, Optimistic policy head
# version = 13 # V7 features, Adjusted scaling on shortterm score variance, and made C++ side read in scalings.
# version = 14 # V7 features, Squared softplus for error variance predictions
# version = 15 # V7 features, Extra nonlinearity for pass output
# version = 16 # V7 features, Q value predictions in the policy head

def get_version(config: ModelConfig):
    return config["version"]

def get_num_bin_input_features(config: ModelConfig):
    version = get_version(config)
    if version == 10 or version == 11 or version == 12 or version == 13 or version == 14 or version == 15 or version == 16:
        return 22
    else:
        assert(False)

def get_num_global_input_features(config: ModelConfig):
    version = get_version(config)
    if version == 10 or version == 11 or version == 12 or version == 13 or version == 14 or version == 15 or version == 16:
        return 19
    else:
        assert(False)

def get_num_meta_encoder_input_features(config_or_meta_encoder_version: Union[ModelConfig,int]):
    if isinstance(config_or_meta_encoder_version,int):
        version = config_or_meta_encoder_version
    else:
        if "metadata_encoder" not in config:
            version = 0
        elif "meta_encoder_version" not in config["metadata_encoder"]:
            version = 1
        else:
            version = config["metadata_encoder"]["meta_encoder_version"]
    assert version == 1
    return 192

b1c6nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":6,
    "mid_num_channels":6,
    "gpool_num_channels":4,
    "use_attention_pool":False,
    "num_attention_pool_heads":2,
    "block_kind": [
        ["rconv1","bottlenest2"],
    ],
    "p1_num_channels":4,
    "g1_num_channels":4,
    "v1_num_channels":4,
    "sbv2_num_channels":4,
    "num_scorebeliefs":2,
    "v2_size":6,
}

b2c16 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":16,
    "mid_num_channels":16,
    "gpool_num_channels":8,
    "use_attention_pool":False,
    "num_attention_pool_heads":2,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regulargpool"],
    ],
    "p1_num_channels":8,
    "g1_num_channels":8,
    "v1_num_channels":8,
    "sbv2_num_channels":12,
    "num_scorebeliefs":2,
    "v2_size":12,
}

b4c32 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":32,
    "mid_num_channels":32,
    "gpool_num_channels":16,
    "use_attention_pool":False,
    "num_attention_pool_heads":2,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regulargpool"],
        ["rconv4","regular"],
    ],
    "p1_num_channels":12,
    "g1_num_channels":12,
    "v1_num_channels":12,
    "sbv2_num_channels":24,
    "num_scorebeliefs":4,
    "v2_size":24,
}

b6c96 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":96,
    "mid_num_channels":96,
    "gpool_num_channels":32,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regulargpool"],
        ["rconv4","regular"],
        ["rconv5","regulargpool"],
        ["rconv6","regular"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":48,
    "num_scorebeliefs":4,
    "v2_size":64,
}

b10c128 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":128,
    "mid_num_channels":128,
    "gpool_num_channels":32,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regulargpool"],
        ["rconv6","regular"],
        ["rconv7","regular"],
        ["rconv8","regulargpool"],
        ["rconv9","regular"],
        ["rconv10","regular"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":64,
    "num_scorebeliefs":6,
    "v2_size":80,
}

b5c192nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":192,
    "mid_num_channels":96,
    "gpool_num_channels":32,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2gpool"],
        ["rconv3","bottlenest2"],
        ["rconv4","bottlenest2gpool"],
        ["rconv5","bottlenest2"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":64,
    "num_scorebeliefs":6,
    "v2_size":80,
}

b15c192 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":192,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regular"],
        ["rconv7","regulargpool"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regular"],
        ["rconv12","regulargpool"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
    ],
    "p1_num_channels":32,
    "g1_num_channels":32,
    "v1_num_channels":32,
    "sbv2_num_channels":80,
    "num_scorebeliefs":8,
    "v2_size":96,
}

b20c256 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regular"],
        ["rconv7","regulargpool"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regular"],
        ["rconv12","regulargpool"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regular"],
        ["rconv17","regulargpool"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}

b30c256bt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":128,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv0","bottle"],
        ["rconv1","bottle"],
        ["rconv2","bottle"],
        ["rconv3","bottle"],
        ["rconv4","bottle"],
        ["rconv5","bottle"],
        ["rconv6","bottlegpool"],
        ["rconv7","bottle"],
        ["rconv8","bottle"],
        ["rconv9","bottle"],
        ["rconv10","bottle"],
        ["rconv11","bottle"],
        ["rconv12","bottlegpool"],
        ["rconv13","bottle"],
        ["rconv14","bottle"],
        ["rconv15","bottle"],
        ["rconv16","bottle"],
        ["rconv17","bottle"],
        ["rconv18","bottlegpool"],
        ["rconv19","bottle"],
        ["rconv20","bottle"],
        ["rconv21","bottle"],
        ["rconv22","bottle"],
        ["rconv23","bottle"],
        ["rconv24","bottlegpool"],
        ["rconv25","bottle"],
        ["rconv26","bottle"],
        ["rconv27","bottle"],
        ["rconv28","bottle"],
        ["rconv29","bottle"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}

b24c320bt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":320,
    "mid_num_channels":160,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottle"],
        ["rconv2","bottle"],
        ["rconv3","bottle"],
        ["rconv4","bottle"],
        ["rconv5","bottle"],
        ["rconv6","bottle"],
        ["rconv7","bottlegpool"],
        ["rconv8","bottle"],
        ["rconv9","bottle"],
        ["rconv10","bottle"],
        ["rconv11","bottle"],
        ["rconv12","bottle"],
        ["rconv13","bottlegpool"],
        ["rconv14","bottle"],
        ["rconv15","bottle"],
        ["rconv16","bottle"],
        ["rconv17","bottle"],
        ["rconv18","bottle"],
        ["rconv19","bottlegpool"],
        ["rconv20","bottle"],
        ["rconv21","bottle"],
        ["rconv22","bottle"],
        ["rconv23","bottle"],
        ["rconv24","bottle"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}

b20c384bt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottle"],
        ["rconv2","bottle"],
        ["rconv3","bottle"],
        ["rconv4","bottle"],
        ["rconv5","bottle"],
        ["rconv6","bottlegpool"],
        ["rconv7","bottle"],
        ["rconv8","bottle"],
        ["rconv9","bottle"],
        ["rconv10","bottle"],
        ["rconv11","bottlegpool"],
        ["rconv12","bottle"],
        ["rconv13","bottle"],
        ["rconv14","bottle"],
        ["rconv15","bottle"],
        ["rconv16","bottlegpool"],
        ["rconv17","bottle"],
        ["rconv18","bottle"],
        ["rconv19","bottle"],
        ["rconv20","bottle"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}


b10c512lbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":512,
    "mid_num_channels":256,
    "gpool_num_channels":128,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottle2"],
        ["rconv2","bottle2"],
        ["rconv3","bottle2"],
        ["rconv4","bottle2gpool"],
        ["rconv5","bottle2"],
        ["rconv6","bottle2"],
        ["rconv7","bottle2"],
        ["rconv8","bottle2gpool"],
        ["rconv9","bottle2"],
        ["rconv10","bottle2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}


b15c384lbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottle2"],
        ["rconv2","bottle2"],
        ["rconv3","bottle2"],
        ["rconv4","bottle2gpool"],
        ["rconv5","bottle2"],
        ["rconv6","bottle2"],
        ["rconv7","bottle2"],
        ["rconv8","bottle2gpool"],
        ["rconv9","bottle2"],
        ["rconv10","bottle2"],
        ["rconv11","bottle2"],
        ["rconv12","bottle2gpool"],
        ["rconv13","bottle2"],
        ["rconv14","bottle2"],
        ["rconv15","bottle2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}

b18c320lbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":320,
    "mid_num_channels":160,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottle2"],
        ["rconv2","bottle2"],
        ["rconv3","bottle2"],
        ["rconv4","bottle2"],
        ["rconv5","bottle2gpool"],
        ["rconv6","bottle2"],
        ["rconv7","bottle2"],
        ["rconv8","bottle2"],
        ["rconv9","bottle2"],
        ["rconv10","bottle2gpool"],
        ["rconv11","bottle2"],
        ["rconv12","bottle2"],
        ["rconv13","bottle2"],
        ["rconv14","bottle2"],
        ["rconv15","bottle2gpool"],
        ["rconv16","bottle2"],
        ["rconv17","bottle2"],
        ["rconv18","bottle2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}

b23c256lbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":128,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottle2"],
        ["rconv2","bottle2"],
        ["rconv3","bottle2"],
        ["rconv4","bottle2"],
        ["rconv5","bottle2"],
        ["rconv6","bottle2gpool"],
        ["rconv7","bottle2"],
        ["rconv8","bottle2"],
        ["rconv9","bottle2"],
        ["rconv10","bottle2"],
        ["rconv11","bottle2"],
        ["rconv12","bottle2gpool"],
        ["rconv13","bottle2"],
        ["rconv14","bottle2"],
        ["rconv15","bottle2"],
        ["rconv16","bottle2"],
        ["rconv17","bottle2"],
        ["rconv18","bottle2gpool"],
        ["rconv19","bottle2"],
        ["rconv20","bottle2"],
        ["rconv21","bottle2"],
        ["rconv22","bottle2"],
        ["rconv23","bottle2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}

b12c384llbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottle3"],
        ["rconv2","bottle3"],
        ["rconv3","bottle3"],
        ["rconv4","bottle3gpool"],
        ["rconv5","bottle3"],
        ["rconv6","bottle3"],
        ["rconv7","bottle3gpool"],
        ["rconv8","bottle3"],
        ["rconv9","bottle3"],
        ["rconv10","bottle3gpool"],
        ["rconv11","bottle3"],
        ["rconv12","bottle3"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}


b10c384nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}


b10c480nb3t = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":480,
    "mid_num_channels":160,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}


b7c384lnbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest3"],
        ["rconv3","bottlenest3gpool"],
        ["rconv5","bottlenest3"],
        ["rconv6","bottlenest3gpool"],
        ["rconv8","bottlenest3"],
        ["rconv9","bottlenest3gpool"],
        ["rconv10","bottlenest3"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}

b5c512nnbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": True,
    "trunk_num_channels":512,
    "outermid_num_channels":256,
    "mid_num_channels":128,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2bottlenest2"],
        ["rconv2","bottlenest2bottlenest2gpool"],
        ["rconv3","bottlenest2bottlenest2"],
        ["rconv4","bottlenest2bottlenest2gpool"],
        ["rconv5","bottlenest2bottlenest2gpool"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}


b20c384lbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottle2"],
        ["rconv2","bottle2"],
        ["rconv3","bottle2"],
        ["rconv4","bottle2"],
        ["rconv5","bottle2"],
        ["rconv6","bottle2gpool"],
        ["rconv7","bottle2"],
        ["rconv8","bottle2"],
        ["rconv9","bottle2"],
        ["rconv10","bottle2"],
        ["rconv11","bottle2gpool"],
        ["rconv12","bottle2"],
        ["rconv13","bottle2"],
        ["rconv14","bottle2"],
        ["rconv15","bottle2"],
        ["rconv16","bottle2gpool"],
        ["rconv17","bottle2"],
        ["rconv18","bottle2"],
        ["rconv19","bottle2"],
        ["rconv20","bottle2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":48,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}


b30c320 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":320,
    "mid_num_channels":320,
    "gpool_num_channels":96,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regulargpool"],
        ["rconv7","regular"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regulargpool"],
        ["rconv12","regular"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regulargpool"],
        ["rconv17","regular"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
        ["rconv21","regulargpool"],
        ["rconv22","regular"],
        ["rconv23","regular"],
        ["rconv24","regular"],
        ["rconv25","regular"],
        ["rconv26","regulargpool"],
        ["rconv27","regular"],
        ["rconv28","regular"],
        ["rconv29","regular"],
        ["rconv30","regular"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b40c256 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regulargpool"],
        ["rconv7","regular"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regulargpool"],
        ["rconv12","regular"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regulargpool"],
        ["rconv17","regular"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
        ["rconv21","regulargpool"],
        ["rconv22","regular"],
        ["rconv23","regular"],
        ["rconv24","regular"],
        ["rconv25","regular"],
        ["rconv26","regulargpool"],
        ["rconv27","regular"],
        ["rconv28","regular"],
        ["rconv29","regular"],
        ["rconv30","regular"],
        ["rconv31","regulargpool"],
        ["rconv32","regular"],
        ["rconv33","regular"],
        ["rconv34","regular"],
        ["rconv35","regular"],
        ["rconv36","regulargpool"],
        ["rconv37","regular"],
        ["rconv38","regular"],
        ["rconv39","regular"],
        ["rconv40","regular"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b18c384nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","bottlenest2"],
        ["rconv14","bottlenest2"],
        ["rconv15","bottlenest2gpool"],
        ["rconv16","bottlenest2"],
        ["rconv17","bottlenest2"],
        ["rconv18","bottlenest2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b18c384dnbt1 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","dilatedbottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","bottlenest2"],
        ["rconv14","bottlenest2"],
        ["rconv15","bottlenest2gpool"],
        ["rconv16","bottlenest2"],
        ["rconv17","bottlenest2"],
        ["rconv18","bottlenest2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b18c384dnbt2 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","dilatedbottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","dilatedbottlenest2"],
        ["rconv14","bottlenest2"],
        ["rconv15","bottlenest2gpool"],
        ["rconv16","bottlenest2"],
        ["rconv17","bottlenest2"],
        ["rconv18","bottlenest2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b14c448nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":448,
    "mid_num_channels":224,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","bottlenest2"],
        ["rconv14","bottlenest2"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":112,
    "num_scorebeliefs":8,
    "v2_size":128,
}

b40c384 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":384,
    "gpool_num_channels":128,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regulargpool"],
        ["rconv7","regular"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regulargpool"],
        ["rconv12","regular"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regulargpool"],
        ["rconv17","regular"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
        ["rconv21","regulargpool"],
        ["rconv22","regular"],
        ["rconv23","regular"],
        ["rconv24","regular"],
        ["rconv25","regular"],
        ["rconv26","regulargpool"],
        ["rconv27","regular"],
        ["rconv28","regular"],
        ["rconv29","regular"],
        ["rconv30","regular"],
        ["rconv31","regulargpool"],
        ["rconv32","regular"],
        ["rconv33","regular"],
        ["rconv34","regular"],
        ["rconv35","regular"],
        ["rconv36","regulargpool"],
        ["rconv37","regular"],
        ["rconv38","regular"],
        ["rconv39","regular"],
        ["rconv40","regular"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":96,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}


b60c320 = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":320,
    "mid_num_channels":320,
    "gpool_num_channels":96,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regulargpool"],
        ["rconv7","regular"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regulargpool"],
        ["rconv12","regular"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regulargpool"],
        ["rconv17","regular"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
        ["rconv21","regulargpool"],
        ["rconv22","regular"],
        ["rconv23","regular"],
        ["rconv24","regular"],
        ["rconv25","regular"],
        ["rconv26","regulargpool"],
        ["rconv27","regular"],
        ["rconv28","regular"],
        ["rconv29","regular"],
        ["rconv30","regular"],
        ["rconv31","regulargpool"],
        ["rconv32","regular"],
        ["rconv33","regular"],
        ["rconv34","regular"],
        ["rconv35","regular"],
        ["rconv36","regulargpool"],
        ["rconv37","regular"],
        ["rconv38","regular"],
        ["rconv39","regular"],
        ["rconv40","regular"],
        ["rconv41","regulargpool"],
        ["rconv42","regular"],
        ["rconv43","regular"],
        ["rconv44","regular"],
        ["rconv45","regular"],
        ["rconv46","regulargpool"],
        ["rconv47","regular"],
        ["rconv48","regular"],
        ["rconv49","regular"],
        ["rconv50","regular"],
        ["rconv51","regulargpool"],
        ["rconv52","regular"],
        ["rconv53","regular"],
        ["rconv54","regular"],
        ["rconv55","regular"],
        ["rconv56","regulargpool"],
        ["rconv57","regular"],
        ["rconv58","regular"],
        ["rconv59","regular"],
        ["rconv60","regular"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":96,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}


b41c384nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":384,
    "mid_num_channels":192,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","bottlenest2"],
        ["rconv14","bottlenest2"],
        ["rconv15","bottlenest2gpool"],
        ["rconv16","bottlenest2"],
        ["rconv17","bottlenest2"],
        ["rconv18","bottlenest2gpool"],
        ["rconv19","bottlenest2"],
        ["rconv20","bottlenest2"],
        ["rconv21","bottlenest2gpool"],
        ["rconv22","bottlenest2"],
        ["rconv23","bottlenest2"],
        ["rconv24","bottlenest2gpool"],
        ["rconv25","bottlenest2"],
        ["rconv26","bottlenest2"],
        ["rconv27","bottlenest2gpool"],
        ["rconv28","bottlenest2"],
        ["rconv29","bottlenest2"],
        ["rconv30","bottlenest2gpool"],
        ["rconv31","bottlenest2"],
        ["rconv32","bottlenest2"],
        ["rconv33","bottlenest2gpool"],
        ["rconv34","bottlenest2"],
        ["rconv35","bottlenest2"],
        ["rconv36","bottlenest2gpool"],
        ["rconv37","bottlenest2"],
        ["rconv38","bottlenest2"],
        ["rconv39","bottlenest2gpool"],
        ["rconv40","bottlenest2"],
        ["rconv41","bottlenest2"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":96,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}

b32c448nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":448,
    "mid_num_channels":224,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","bottlenest2"],
        ["rconv14","bottlenest2"],
        ["rconv15","bottlenest2gpool"],
        ["rconv16","bottlenest2"],
        ["rconv17","bottlenest2"],
        ["rconv18","bottlenest2gpool"],
        ["rconv19","bottlenest2"],
        ["rconv20","bottlenest2"],
        ["rconv21","bottlenest2gpool"],
        ["rconv22","bottlenest2"],
        ["rconv23","bottlenest2"],
        ["rconv24","bottlenest2gpool"],
        ["rconv25","bottlenest2"],
        ["rconv26","bottlenest2"],
        ["rconv27","bottlenest2gpool"],
        ["rconv28","bottlenest2"],
        ["rconv29","bottlenest2"],
        ["rconv30","bottlenest2gpool"],
        ["rconv31","bottlenest2"],
        ["rconv32","bottlenest2"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":96,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}


b28c512nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":512,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","bottlenest2"],
        ["rconv14","bottlenest2"],
        ["rconv15","bottlenest2gpool"],
        ["rconv16","bottlenest2"],
        ["rconv17","bottlenest2"],
        ["rconv18","bottlenest2gpool"],
        ["rconv19","bottlenest2"],
        ["rconv20","bottlenest2"],
        ["rconv21","bottlenest2gpool"],
        ["rconv22","bottlenest2"],
        ["rconv23","bottlenest2"],
        ["rconv24","bottlenest2gpool"],
        ["rconv25","bottlenest2"],
        ["rconv26","bottlenest2"],
        ["rconv27","bottlenest2gpool"],
        ["rconv28","bottlenest2"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":128,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}

b20c640nbt = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":640,
    "mid_num_channels":320,
    "gpool_num_channels":96,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","bottlenest2"],
        ["rconv2","bottlenest2"],
        ["rconv3","bottlenest2gpool"],
        ["rconv4","bottlenest2"],
        ["rconv5","bottlenest2"],
        ["rconv6","bottlenest2gpool"],
        ["rconv7","bottlenest2"],
        ["rconv8","bottlenest2"],
        ["rconv9","bottlenest2gpool"],
        ["rconv10","bottlenest2"],
        ["rconv11","bottlenest2"],
        ["rconv12","bottlenest2gpool"],
        ["rconv13","bottlenest2"],
        ["rconv14","bottlenest2"],
        ["rconv15","bottlenest2gpool"],
        ["rconv16","bottlenest2"],
        ["rconv17","bottlenest2"],
        ["rconv18","bottlenest2gpool"],
        ["rconv19","bottlenest2"],
        ["rconv20","bottlenest2"],
    ],
    "p1_num_channels":64,
    "g1_num_channels":64,
    "v1_num_channels":96,
    "sbv2_num_channels":128,
    "num_scorebeliefs":8,
    "v2_size":144,
}

sandbox = {
    "version":15,
    "norm_kind":"fixup",
    "bnorm_epsilon": 1e-4,
    "bnorm_running_avg_momentum": 0.001,
    "initial_conv_1x1": False,
    "trunk_num_channels":256,
    "mid_num_channels":256,
    "gpool_num_channels":64,
    "use_attention_pool":False,
    "num_attention_pool_heads":4,
    "block_kind": [
        ["rconv1","regular"],
        ["rconv2","regular"],
        ["rconv3","regular"],
        ["rconv4","regular"],
        ["rconv5","regular"],
        ["rconv6","regular"],
        ["rconv7","regulargpool"],
        ["rconv8","regular"],
        ["rconv9","regular"],
        ["rconv10","regular"],
        ["rconv11","regular"],
        ["rconv12","regulargpool"],
        ["rconv13","regular"],
        ["rconv14","regular"],
        ["rconv15","regular"],
        ["rconv16","regular"],
        ["rconv17","regulargpool"],
        ["rconv18","regular"],
        ["rconv19","regular"],
        ["rconv20","regular"],
    ],
    "p1_num_channels":48,
    "g1_num_channels":48,
    "v1_num_channels":96,
    "sbv2_num_channels":96,
    "num_scorebeliefs":8,
    "v2_size":112,
}


base_config_of_name = {
    # Micro-sized model configs
    "b1c6nbt": b1c6nbt,
    "b2c16": b2c16,
    "b4c32": b4c32,
    "b6c96": b6c96,

    # Small model configs, not too different in inference cost from b10c128
    "b10c128": b10c128,
    "b5c192nbt": b5c192nbt,

    # Medium model configs, not too different in inference cost from b15c192
    "b15c192": b15c192,

    # Roughly AlphaZero-sized, not too different in inference cost from b20c256
    "b20c256": b20c256,
    "b30c256bt": b30c256bt,
    "b24c320bt": b24c320bt,
    "b20c384bt": b20c384bt,
    "b23c256lbt": b23c256lbt,
    "b18c320lbt": b18c320lbt,
    "b15c384lbt": b15c384lbt,
    "b10c512lbt": b10c512lbt,
    "b12c384llbt": b12c384llbt,
    "b10c384nbt": b10c384nbt,  # Recommended best config for this cost
    "b10c480nb3t": b10c480nb3t,
    "b7c384lnbt": b7c384lnbt,
    "b5c512nnbt": b5c512nnbt,
    "b20c384lbt": b20c384lbt,

    # Roughly AlphaGoZero-sized, not too different in inference cost from b40c256
    "b30c320": b30c320,
    "b40c256": b40c256,
    "b18c384nbt": b18c384nbt,  # Recommended best config for this cost
    "b14c448nbt": b14c448nbt,
    "b18c384dnbt1": b18c384dnbt1,
    "b18c384dnbt2": b18c384dnbt2,

    # Large model configs, not too different in inference cost from b60c320
    "b40c384": b40c384,
    "b60c320": b60c320,
    "b41c384nbt": b41c384nbt,
    "b32c448nbt": b32c448nbt,
    "b28c512nbt": b28c512nbt,  # Recommended best config for this cost
    "b20c640nbt": b20c640nbt,

    "sandbox": sandbox,
}

config_of_name = {}
for name, base_config in base_config_of_name.items():
    config = base_config.copy()
    config_of_name[name] = config


for name, base_config in list(config_of_name.items()):
    # Fixup initialization
    config = base_config.copy()
    config["norm_kind"] = "fixup"
    config_of_name[name+""] = config

    # Fixed scaling normalization
    config = base_config.copy()
    config["norm_kind"] = "fixscale"
    config_of_name[name+"-fs"] = config

    # Batchnorm without gamma terms
    config = base_config.copy()
    config["norm_kind"] = "bnorm"
    config_of_name[name+"-bn"] = config

    # Batchrenorm without gamma terms
    config = base_config.copy()
    config["norm_kind"] = "brenorm"
    config_of_name[name+"-brn"] = config

    # Fixed scaling normalization + Batchrenorm without gamma terms
    config = base_config.copy()
    config["norm_kind"] = "fixbrenorm"
    config_of_name[name+"-fbrn"] = config

    # Batchnorm with gamma terms
    config = base_config.copy()
    config["norm_kind"] = "bnorm"
    config["bnorm_use_gamma"] = True
    config_of_name[name+"-bng"] = config

    # Batchrenorm with gamma terms
    config = base_config.copy()
    config["norm_kind"] = "brenorm"
    config["bnorm_use_gamma"] = True
    config_of_name[name+"-brng"] = config

    # Fixed scaling normalization + Batchrenorm with gamma terms
    config = base_config.copy()
    config["norm_kind"] = "fixbrenorm"
    config["bnorm_use_gamma"] = True
    config_of_name[name+"-fbrng"] = config

    # Fixed scaling normalization + ONE batch norm layer in the entire net.
    config = base_config.copy()
    config["norm_kind"] = "fixscaleonenorm"
    config["bnorm_use_gamma"] = True
    config_of_name[name+"-fson"] = config

for name, base_config in list(config_of_name.items()):
    config = base_config.copy()
    config["activation"] = "elu"
    config_of_name[name+"-elu"] = config

    config = base_config.copy()
    config["activation"] = "gelu"
    config_of_name[name+"-gelu"] = config

    config = base_config.copy()
    config["activation"] = "mish"
    config_of_name[name+"-mish"] = config

for name, base_config in list(config_of_name.items()):
    config = base_config.copy()
    config["use_attention_pool"] = True
    config_of_name[name+"-ap"] = config

for name, base_config in list(config_of_name.items()):
    config = base_config.copy()
    config["use_repvgg_init"] = True
    config_of_name[name+"-rvgi"] = config

    config = base_config.copy()
    config["use_repvgg_linear"] = True
    config_of_name[name+"-rvgl"] = config

    config = base_config.copy()
    config["use_repvgg_init"] = True
    config["use_repvgg_learning_rate"] = True
    config_of_name[name+"-rvglr"] = config

for name, base_config in list(config_of_name.items()):
    # Add intermediate heads, for use with self-distillation or embedding small net in big one.
    config = base_config.copy()
    config["has_intermediate_head"] = True
    config["intermediate_head_blocks"] = len(config["block_kind"]) // 2
    config_of_name[name+"-ih"] = config

    # Add parallel heads that uses the final trunk batchnorm.
    # The original normal heads disables the final trunk batchnorm
    # This only makes sense for configs that use some form of batchnorm.
    if "norm" in config["norm_kind"]:
        config = base_config.copy()
        config["has_intermediate_head"] = True
        config["intermediate_head_blocks"] = len(config["block_kind"])
        config["trunk_normless"] = True
        config_of_name[name+"-bnh"] = config

for name, base_config in list(config_of_name.items()):
    config = base_config.copy()
    config["metadata_encoder"] = {
        "meta_encoder_version": 1,
        "internal_num_channels": config["trunk_num_channels"],
    }
    config_of_name[name+"-meta"] = config
