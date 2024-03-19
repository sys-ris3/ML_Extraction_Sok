#include "activation_layer.h"
#include "utils.h"
#include "blas.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*));

    l.forward = forward_activation_layer;
    l.activation = activation;
    LOGD("Activation Layer: %d inputs\n", inputs);
    return l;
}

void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
    DUMPW8("activation output bytes:",l.outputs, l.output);
    DUMPW4F("activation output float:",l.outputs, l.output);
}
