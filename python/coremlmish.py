from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY, register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil import Builder as mb

# Remove the original mish function
if "mish" in _TORCH_OPS_REGISTRY:
    del _TORCH_OPS_REGISTRY["mish"]

# Set the function to use
__function__ = "mish_torch_sigmoid"

# Torch Mish Operator with Sigmoid Approximation that can run on Neural Engine
#
# This function applies the Mish activation function to the input tensor `x`. The Mish function is defined as 
# x * tanh(Softplus(x)), where Softplus(x) is typically defined as log(1 + exp(x)). However, to avoid 
# computational issues with large values of x in float16 format, a sigmoid-based approximation is used.
#
# Instead of using a conditional operation to switch between log(1 + exp(x)) and x based on a threshold,
# a sigmoid function is utilized to smoothly transition between the standard Softplus function and a linear 
# approximation. This approach helps in managing large input values, maintaining numerical stability in 
# 16-bit floating point computations.
#
# The threshold for switching between Softplus and linear behavior is set at 10.39, rather than the original 20.
# This modification is made considering that exp(10.39) = 32532.666936, which is within the representable range 
# of float16, unlike exp(20) = 485165195.40979004, which exceeds the limits of float16.
#
# Arguments:
# context: An object containing information about the execution context of the function.
# node: An object representing a node in a computation graph.
def mish_torch_sigmoid(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]

    threshold = 10.39

    # Approximating conditional behavior using sigmoid function
    sigmoid_threshold = mb.sigmoid(x=mb.sub(x=x, y=threshold))
    
    # Approximate implementation of Softplus
    softplus_part = mb.softplus(x=mb.minimum(x=x, y=threshold))
    softplus = mb.add(x=mb.mul(x=x, y=sigmoid_threshold), 
                      y=mb.mul(x=softplus_part, y=mb.sub(x=1.0, y=sigmoid_threshold)))

    # Mish(x) = x * tanh(Softplus(x))
    tanh_softplus = mb.tanh(x=softplus)
    res = mb.mul(x=x, y=tanh_softplus, name=node.name)
    context.add(res)


# Torch Mish operator that *could* run on Neural Engine
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


# Register the function
@register_torch_op
def mish(context, node):
    if __function__ == "mish_torch_sigmoid":
        mish_torch_sigmoid(context, node)
    else:
        mish_torch_ne(context, node)
    