// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "layer.h"


const char *get_layer_string(LAYER_TYPE a);

network *make_network(int n);

#endif

