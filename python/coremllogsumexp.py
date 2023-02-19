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
from coremltools.converters.mil.frontend.torch.ops import _get_inputs, _np
from coremltools.converters.mil.mil import types
from coremltools.converters.mil import Builder as mb

if "logsumexp" in _TORCH_OPS_REGISTRY:
    del _TORCH_OPS_REGISTRY["logsumexp"]

@register_torch_op
def logsumexp(context, node):
    inputs = _get_inputs(context, node)

    x = inputs[0]
    if types.is_bool(x.dtype):
        # TODO: In the future when MIL op supports bool, we need to use curr_opset_version to decide
        # if we want to cast or not.
        x = mb.cast(x=x, dtype="fp32")
    kwargs = {"x": x, "name": node.name}

    # @axes is optional, so omit if None.
    axes = inputs[1]
    if axes is not None:
        # @axes needs to be a list, but if only one axis was specified in the
        # model, it will be constructed as an int. Construct a new constant as a
        # list.
        if not isinstance(axes.val, _np.ndarray):
            axes = mb.const(val=[axes.val], name=axes.name + "_list")
            context.add(axes)
        kwargs["axes"] = axes

    # @keep_dims is optional.
    if len(inputs) >= 3:
        keep_dims = inputs[2]
        kwargs["keep_dims"] = keep_dims

    # Last input to mean is an optional output tensor. We always expect this to
    # be None or absent.
    assert len(inputs) <= 3 or inputs[3] is None
    if node.kind == "sum":
        res = mb.reduce_sum(**kwargs)
    elif node.kind == "logsumexp":
        res = mb.reduce_log_sum_exp(**kwargs)
    else:
        res = mb.reduce_mean(**kwargs)
    context.add(res)
