#ifndef ADD_MASK_LAYER_H
#define ADD_MASK_LAYER_H

#include "layer.h"
#include "network.h"

layer make_add_mask_layer(int batch, int h, int w, int c);

void forward_add_mask_layer(layer l, network net);

#endif

