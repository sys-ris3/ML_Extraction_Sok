#include "batchnorm_layer.h"
#include "blas.h"
#include "utils.h"
#include <stdio.h>

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void scale_bias_nhwc(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*size + j)*n + i] *= scales[i];
            }
        }
    }
}
void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void add_bias_nhwc(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*size + j)*n + i] += biases[i];
            }
        }
    }
}

layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    LOGD("Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer l = {0};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.output = calloc(h * w * c * batch, sizeof(float));
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.scales = calloc(c, sizeof(float));
    l.biases = calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        l.scales[i] = 1;
    }

    l.rolling_mean = calloc(c, sizeof(float));
    l.rolling_variance = calloc(c, sizeof(float));

    l.forward = forward_batchnorm_layer;
    return l;
}

void forward_batchnorm_layer(layer l, network net)
{
    printf("forward_bn\n");
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    if (l.nhwc == 1) {
        normalize_cpu_nhwc(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    } else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    if (l.nhwc == 1) {
        scale_bias_nhwc(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_nhwc(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
    } else {
        scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
    }

    DUMPW8("batchnorm output bytes:",l.outputs, l.output);
    DUMPW4F("batchnorm output float:",l.outputs, l.output);
}
