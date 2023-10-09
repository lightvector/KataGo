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

# Remove the original mish function
if "mish" in _TORCH_OPS_REGISTRY:
    del _TORCH_OPS_REGISTRY["mish"]

# Set the function to use
__function__ = "mish_torch_softplus"

# Torch Mish operator that can run on Neural Engine
#
# This function applies the Mish activation function on the input tensor `x`. The Mish function is defined as 
# x * tanh(Softplus(x)), where Softplus(x) is defined as log(1 + exp(min(x, 10.39))) if x < 10.39 and x otherwise.
#
# The function uses the `mb` module to perform operations such as `minimum`, `exp`, `add`, `log`, `less`, `select`, 
# and `tanh`.
#
# The threshold of softplus is modified to 10.39, which is different from the original 20. This is because 
# exp(10.39) = 32532.666936 < 32767.0 < 65504.0, so the result of exp(10.39) can be represented by float16. If the threshold 
# of softplus is 20, the result of exp(20) is 485165195.40979004, which is out of range of float16.
#
# Arguments:
# context: an object that contains information about the execution context of the function
# node: an object that represents a node in a computation graph
def mish_torch_ne(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]

    threshold = 10.39

    # Softplus(x) = log(1 + exp(min(x, 10.39))) if x < 10.39 else x
    min_x_threshold = mb.minimum(x=x, y=threshold)
    exp_min_x_threshold = mb.exp(x=min_x_threshold)
    add_exp_min_x_threshold_1 = mb.add(x=exp_min_x_threshold, y=1.0)
    log_add_exp_min_x_threshold_1 = mb.log(x=add_exp_min_x_threshold_1)
    # less(x, y) = x < y
    x_less_than_threshold = mb.less(x=x, y=threshold)
    # select(cond, a, b) = a if cond else b
    softplus = mb.select(cond=x_less_than_threshold, a=log_add_exp_min_x_threshold_1, b=x)

    # Mish(x) = x * tanh(Softplus(x))
    tanh_softplus = mb.tanh(x=softplus)
    res = mb.mul(x=x, y=tanh_softplus, name=node.name)
    context.add(res)

# Torch Mish operator which is implemented by Softplus
# Numerically stable, but cannot run on Neural Engine
def mish_torch_softplus(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]

    softplus = mb.softplus(x=x)
    tanh = mb.tanh(x=softplus)
    res = mb.mul(x=x, y=tanh, name=node.name)
    context.add(res)

# Register the function
@register_torch_op
def mish(context, node):
    if __function__ == "mish_torch_ne":
        mish_torch_ne(context, node)
    else:
        mish_torch_softplus(context, node)
    