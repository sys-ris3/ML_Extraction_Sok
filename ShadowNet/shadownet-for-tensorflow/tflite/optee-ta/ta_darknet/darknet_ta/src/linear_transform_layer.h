#ifndef LINEAR_TRANSFORM_LAYER_H
#define LINEAR_TRANSFORM_LAYER_H

#include "layer.h"
#include "network.h"

layer make_linear_transform_layer(int batch, int h, int w, int c, int units);

void forward_linear_transform_layer(layer l, network net);

#endif

