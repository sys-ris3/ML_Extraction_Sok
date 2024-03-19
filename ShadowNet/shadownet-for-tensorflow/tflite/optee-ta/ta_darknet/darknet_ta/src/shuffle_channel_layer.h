#ifndef SHUFFLE_CHANNEL_LAYER_H
#define SHUFFLE_CHANNEL_LAYER_H

#include "layer.h"
#include "network.h"

layer make_shuffle_channel_layer(int batch, int h, int w, int c);

void forward_shuffle_channel_layer(layer l, network net);

#endif

