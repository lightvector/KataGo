# Copyright (c) 2020, Apple Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:  
#
# 1.  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2.  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3.  Neither the name of the copyright holder(s) nor the names of any contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

if "mish" in _TORCH_OPS_REGISTRY:
    del _TORCH_OPS_REGISTRY["mish"]

__function__ = "mish_torch_ne_fast"

# Torch Mish operator that can run on Neural Engine
# This implementation sets the threshold to inf, so it is not used.
def mish_torch_ne_fast(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]

    # Softplus(x) = log(1 + exp(x))
    exp = mb.exp(x=x)
    add = mb.add(x=exp, y=1.0)
    softplus = mb.log(x=add)
    # Mish(x) = x * tanh(Softplus(x))
    tanh = mb.tanh(x=softplus)
    res = mb.mul(x=x, y=tanh, name=node.name)
    context.add(res)

# Torch Mish operator that can run on Neural Engine
def mish_torch_ne(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]

    # Softplus(x) = log(1 + exp(x)) if x < 20 else x
    less = mb.less(x=x, y=20.0)
    exp = mb.exp(x=x)
    add = mb.add(x=exp, y=1.0)
    log = mb.log(x=add)
    softplus = mb.select(cond=less, a=log, b=x)
    # Mish(x) = x * tanh(Softplus(x))
    tanh = mb.tanh(x=softplus)
    res = mb.mul(x=x, y=tanh, name=node.name)
    context.add(res)

# Torch Mish operator which is implemented by Softplus
def mish_torch_softplus(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]

    softplus = mb.softplus(x=x)
    tanh = mb.tanh(x=softplus)
    res = mb.mul(x=x, y=tanh, name=node.name)
    context.add(res)

@register_torch_op
def mish(context, node):
    if __function__ == "mish_torch_ne_fast":
        mish_torch_ne_fast(context, node)
    elif __function__ == "mish_torch_softplus":
        mish_torch_softplus(context, node)
    else:
        mish_torch_ne(context, node)
    