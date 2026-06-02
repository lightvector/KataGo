#ifndef NEURALNET_ACTIVATIONS_H
#define NEURALNET_ACTIVATIONS_H

// x
static constexpr int ACTIVATION_IDENTITY = 0;

// max(x,0)
static constexpr int ACTIVATION_RELU = 1;

// x * tanh(softplus(x))
static constexpr int ACTIVATION_MISH = 2;

// x / (1+exp(-x))
static constexpr int ACTIVATION_SILU = 3;

// x * tanh(softplus(8x))
static constexpr int ACTIVATION_MISH_SCALE8 = 12;

#endif
