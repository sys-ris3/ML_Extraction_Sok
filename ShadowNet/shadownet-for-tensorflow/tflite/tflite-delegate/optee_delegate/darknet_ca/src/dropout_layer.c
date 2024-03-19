#include "dropout_layer.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;
    l.forward = forward_dropout_layer;
    LOGD("dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

void forward_dropout_layer(dropout_layer l, network net)
{
    return;
}
