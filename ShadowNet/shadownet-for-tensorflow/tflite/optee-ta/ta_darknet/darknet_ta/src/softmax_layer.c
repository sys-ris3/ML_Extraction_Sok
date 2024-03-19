#include "softmax_layer.h"
#include "blas.h"

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <tee_internal_api.h>

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    DMSG("softmax                                        %4d\n",  inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
//    l.output = calloc(inputs*batch, sizeof(float));
    DMSG("softmax                                        %4d\n",  inputs);
    DMSG("softmax, l.output%p\n",  l.output);

    l.forward = forward_softmax_layer;
    return l;
}

void forward_softmax_layer(const softmax_layer l, network net)
{
    softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);

}
