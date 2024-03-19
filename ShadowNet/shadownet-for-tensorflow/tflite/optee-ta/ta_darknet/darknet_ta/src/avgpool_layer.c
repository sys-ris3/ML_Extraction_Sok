#include "avgpool_layer.h"
#include "utils.h"
#include <stdio.h>

avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
{
    //fprintf(stderr, "avg                     %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    avgpool_layer l = {0};
    l.type = AVGPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = c;
    l.outputs = l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
   // l.output =  calloc(output_size, sizeof(float));
    l.forward = forward_avgpool_layer;
    return l;
}

void resize_avgpool_layer(avgpool_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
}

void forward_avgpool_layer(const avgpool_layer l, network net)
{
    int b,i,k;
    int out_index,in_index;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            out_index = k + b*l.c;
            l.output[out_index] = 0;
            for(i = 0; i < l.h*l.w; ++i){
                if (l.nhwc == 1)
                    in_index = k + l.c*(i + b*l.h*l.w); 
                else
                    in_index = i + l.h*l.w*(k + b*l.c);

                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.h*l.w;
        }
    }
    DUMPW8("avgpool output bytes:",l.outputs, l.output);
    DUMPW4F("avgpool output float:",l.outputs, l.output);
}
