#include "activation_layer.h"
#include "utils.h"
#include "blas.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tee_internal_api.h>

layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    //l.output = calloc(batch*inputs, sizeof(float*));

    l.forward = forward_activation_layer;
    l.activation = activation;
    //fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
    DMSG("Activation Layer: %d inputs  \n", inputs);
    DMSG("Activation Layer: l.output:%p \n", l.output);
    return l;
}

void forward_activation_layer(layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);
    DUMPW8("activation output bytes:",l.outputs, l.output);
    DUMPW4F("activation output float:",l.outputs, l.output);
}
