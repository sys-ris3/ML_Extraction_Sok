#include "add_mask_layer.h"
#include "utils.h"
#include "blas.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_add_mask_layer(int batch, int h, int w, int c)
{
    layer l = {0};
    l.type = ADD_MASK;

    l.h = h;
    l.w = w;
    l.c = c;
    l.inputs = h * w * c;
    l.outputs = l.inputs;
    l.batch=batch;

    l.out_h = h;
    l.out_w = w;
    l.out_c = c;
    l.output = calloc(batch*l.inputs, sizeof(float));
    l.weights = calloc(l.inputs, sizeof(float));
    l.rscalar = calloc(1, sizeof(float));

    l.forward = forward_add_mask_layer;
    LOGD("Add_Mask Layer: %d inputs\n", l.inputs);
    return l;
}

void forward_add_mask_layer(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.batch;
    int k = l.inputs;
    int i, j;
    for (i = 0; i < m; ++i){
        for (j = 0; j < k; ++j) {
            l.output[i * k + j] = net.input[i * k + j] + l.weights[j] * l.rscalar[0]; 
        }
    }
    DUMPW8("add_mask output bytes:",l.outputs, l.output);
    DUMPW4F("add_mask output float:",l.outputs, l.output);
}
