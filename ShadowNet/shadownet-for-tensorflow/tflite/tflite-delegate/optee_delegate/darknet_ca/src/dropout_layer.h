#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer dropout_layer;

dropout_layer make_dropout_layer(int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer l, network net);

#endif
