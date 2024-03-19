#include "linear_transform_layer.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_linear_transform_layer(int batch, int h, int w, int c, int units)
{
    layer l = {0};
    l.type = LINEAR_TRANSFORM;

    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.units = units;
    l.inputs = l.h * l.w * l.c;

    l.out_h = h;
    l.out_w = w;
    l.out_c = units;
    l.outputs = l.h * l.w * l.units;

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.obfweights = calloc(2*l.units, sizeof(int));
    l.rbias = calloc(l.units, sizeof(float));

    l.forward = forward_linear_transform_layer;
    LOGD("Linear_Transform Layer: %d inputs\n", l.inputs);
    return l;
}

void forward_linear_transform_layer(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int B = l.batch;
    int H = l.h;
    int W = l.w;
    int M = l.c;
    int N = l.units;

    int b, h, w, n;
    int idx_from, idx_rand;
    float scalar;
    for (b = 0; b < B; ++b){
        for (h = 0; h < H; ++h) {
            for (w = 0; w < W; ++w) {
                for (n = 0; n < N; ++n) {
                    idx_from = l.obfweights[n];
                    idx_rand = l.obfweights[N + n];
                    scalar = l.rbias[n];

                    l.output[(b * H * W * N) + (h * W * N) + (w * N) + n] = 
                      net.input[(b * H * W * M) + (h * W * M) + (w * M) + idx_from] * scalar +  
                        net.input[(b * H * W * M) + (h * W * M) + (w * M) + idx_rand]; 
                }
            }
        }
    }

    DUMPW8("linear_transform output bytes:",l.outputs, l.output);
    DUMPW4F("linear_transform output float:",l.outputs, l.output);
}
