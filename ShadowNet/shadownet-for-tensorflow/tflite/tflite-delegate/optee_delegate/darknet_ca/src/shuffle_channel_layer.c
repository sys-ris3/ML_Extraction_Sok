#include "shuffle_channel_layer.h"
#include "utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_shuffle_channel_layer(int batch, int h, int w, int c)
{
    layer l = {0};
    l.type = SHUFFLE_CHANNEL;

    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.inputs = l.h * l.w * l.c;

    l.out_h = h;
    l.out_w = w;
    l.out_c = c;
    l.outputs = l.h * l.w * l.c;

    l.output = calloc(batch*l.outputs, sizeof(float));
    l.obfweights = calloc(l.c, sizeof(int));
    l.rbias = calloc(l.c, sizeof(float));

    l.forward = forward_shuffle_channel_layer;
    LOGD("Shuffle_Channel Layer: %d inputs\n", l.inputs);
    return l;
}

void forward_shuffle_channel_layer(layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int B = l.batch;
    int H = l.h;
    int W = l.w;
    int N = l.c;

    int b, h, w, n;
    int idx_from;
    float scalar;
    for (b = 0; b < B; ++b){
        for (h = 0; h < H; ++h) {
            for (w = 0; w < W; ++w) {
                for (n = 0; n < N; ++n) {
                    idx_from = l.obfweights[n];
                    scalar = l.rbias[n];

                    l.output[(b * H * W * N) + (h * W * N) + (w * N) + n] = 
                      net.input[(b * H * W * N) + (h * W * N) + (w * N) + idx_from] * scalar;
                }
            }
        }
    }
    DUMPW8("shuffle channel output bytes:",l.outputs, l.output);
    DUMPW4F("shuffle channel output float:",l.outputs, l.output);
}
