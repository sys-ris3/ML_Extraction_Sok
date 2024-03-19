#include "dropout_layer.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int __maybe_unused batch, int __maybe_unused inputs, float __maybe_unused probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.forward = forward_dropout_layer;
    //DMSG(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

void forward_dropout_layer(dropout_layer __maybe_unused l, network __maybe_unused net)
{
    return;
}
