#include <float.h>
#include <shadownet.h>
#include <math_ta.h>
#include <tee_internal_api_extensions.h>
#include <utee_defines.h>

//#define DEBUG_WEIGHTS
//#define DEBUG_TIME
#define sqrt sqrtf
#define NEON_MULADD
#define NEON_SQRT 
#define NEON_RELU6

#define MAX_FILTERS     1024

#ifdef DEBUG_WEIGHTS
/* dump weigths first 8 bytes */
#define DUMPW4F(TAG, size, pw) DMSG(TAG " bufsize:%d [0-7]: %6f %6f %6f %6f\n",size, \
        pw[0], \
        pw[1], \
        pw[2], \
        pw[3])

#define DUMPW4I(TAG, size, pw) DMSG(TAG " bufsize:%d [0-7]: %6d %6d %6d %6d\n",size, \
        pw[0], \
        pw[1], \
        pw[2], \
        pw[3]) 

/* dump weigths first 8 bytes */
#define DUMPW8(TAG, size, pw) DMSG(TAG " bufsize:%d [0-7]: %3u %3u %3u %3u %3u %3u %3u %3u\n",size, \
        ((unsigned char *)pw)[0], \
        ((unsigned char *)pw)[1], \
        ((unsigned char *)pw)[2], \
        ((unsigned char *)pw)[3], \
        ((unsigned char *)pw)[4], \
        ((unsigned char *)pw)[5], \
        ((unsigned char *)pw)[6], \
        ((unsigned char *)pw)[7])

#define DUMPW4(TAG, size, pw) DMSG(TAG " bufsize:%d [0-7]: %3u %3u %3u %3u\n",size, \
        ((unsigned char *)pw)[0], \
        ((unsigned char *)pw)[1], \
        ((unsigned char *)pw)[2], \
        ((unsigned char *)pw)[3]) 
#else

#define DUMPW4F(TAG, size, pw) 
#define DUMPW4I(TAG, size, pw) 
#define DUMPW8(TAG, size, pw)
#define DUMPW4(TAG, size, pw)

#endif

#ifdef DEBUG_TIME

#define MEASURE_TIME_HEAD       TEE_Time start, end, time; \
                                uint32_t delta; \
                                TEE_GetSystemTime(&start)

#define MEASURE_TIME_TAIL       TEE_GetSystemTime(&end);   \
                                TEE_TIME_SUB(end, start, time); \
                                delta = time.seconds * 1000 + time.millis;      \
                                EMSG("%s takes %d ms\n", __func__, delta)

#else

#define MEASURE_TIME_HEAD       do{}while(0)
#define MEASURE_TIME_TAIL       do{}while(0)

#endif // DEBUG_TIME


void forward_linear_transform_layer(shadownet *net, float *input, float *output)
{
    int H = net->h;
    int W = net->w;
    int M = net->c;
    int N = net->units;

    int h, w, n;
    int idx_from, idx_rand;
    float scalar;

    MEASURE_TIME_HEAD;

    DMSG("%s h:%d,w:%d,m:%d,n:%d",__func__, H, W, M, N);
    for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
            for (n = 0; n < N; ++n) {
                idx_from = net->lt_obfweights[n];
                idx_rand = net->lt_obfweights[N + n];
                scalar = net->lt_rbias[n];
                if (idx_from >M || idx_rand >= M)
                    DMSG("idx_from:%d,idx_rand:%d,scalar:%f",idx_from,idx_rand,scalar);

                output[(h * W * N) + (w * N) + n] = 
                  input[(h * W * M) + (w * M) + idx_from] * scalar +  
                    input[(h * W * M) + (w * M) + idx_rand]; 
            }
        }
    }
    
    MEASURE_TIME_TAIL;

    DUMPW8("linear_transform output bytes:",net->outputs, output);
    DUMPW4F("linear_transform output float:",net->outputs, output);
}

void forward_shuffle_channel_layer(shadownet *net, float *input, float *output)
{
    // TODO N == units ? C
    int H = net->h;
    int W = net->w;
    int N = net->units;

    int h, w, n;
    int idx_from;
    float scalar;

    MEASURE_TIME_HEAD;

    DMSG("%s",__func__);
    for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
            for (n = 0; n < N; ++n) {
                idx_from = net->sf_obfweights[n];
                scalar = net->sf_rbias[n];

                output[(h * W * N) + (w * N) + n] = 
                  input[(h * W * N) + (w * N) + idx_from] * scalar;
            }
        }
    }

    MEASURE_TIME_TAIL;

    DUMPW8("shuffle channel output bytes:",net->outputs, output);
    DUMPW4F("shuffle channel output float:",net->outputs, output);
}

void forward_add_mask_layer_a(shadownet *net, float *input, float *output)
{
    // TODO set inputs
    int k = net->inputs;

    MEASURE_TIME_HEAD;

#ifdef NEON_MULADD
    neon_muladd_fixed_scalar(net->am_weights_a, net->rscalar_a, input, output, k); 
#else
    int j;
    for (j = 0; j < k; ++j) {
        output[j] = input[j] + net->am_weights_a[j] * net->rscalar_a; 
    }
#endif

    MEASURE_TIME_TAIL;

    DUMPW8("add_mask output bytes:",k, output);
    DUMPW4F("add_mask output float:",k, output);
}

void forward_add_mask_layer_b(shadownet *net, float *input, float *output)
{
    // TODO set inputs
    int k = net->inputs;

    MEASURE_TIME_HEAD;

#ifdef NEON_MULADD
    neon_muladd_fixed_scalar(net->am_weights_b, net->rscalar_b, input, output, k); 
#else
    int j;
    for (j = 0; j < k; ++j) {
        output[j] = input[j] + net->am_weights_b[j] * net->rscalar_b; 
    }
#endif

    MEASURE_TIME_TAIL;

    DUMPW8("add_mask output bytes:",k, output);
    DUMPW4F("add_mask output float:",k, output);
}

static inline float relu6_activate(float x){return (x < 0.) ? 0 : (6.0 < x) ? 6.0: x;}

void forward_relu6_layer(shadownet *net, float *input, float *output) {
    int i, n;

    MEASURE_TIME_HEAD;

    // TODO set inputs
    n = net->inputs;
#ifdef NEON_RELU6
   neon_relu6(input, output, n); 
#else
    for(i = 0; i < n; ++i){
        output[i] = relu6_activate(input[i]);
    }
#endif

    MEASURE_TIME_TAIL;
}

#ifdef NEON_MULADD
void scale_add_nhwc(float *output, float *scales, float *biases, int n, int size)
{
    int i,j;
    for(j = 0; j < size; ++j){
        neon_muladd(output + j*n, scales, biases, output + j*n, n);
    }
}
#endif

void add_bias_nhwc(float *output, float *biases, int n, int size)
{
    int i,j;
    for(j = 0; j < size; ++j){
        for(i = 0; i < n; ++i){
            output[j*n + i] += biases[i];
        }
    }
}

void scale_bias_nhwc(float *output, float *scales, int n, int size)
{
    int i,j;
    for(j = 0; j < size; ++j){
        for(i = 0; i < n; ++i){
            output[j*n + i] *= scales[i];
        }
    }
}

void normalize_cpu_nhwc(float *x, float *mean, float *variance, int filters, int spatial)
{
    int f, i;
    float sqrt_var[MAX_FILTERS];

#ifdef NEON_SQRT
    sqrt_buf(variance, sqrt_var, filters); 
#else
    for (i = 0; i < filters; ++i) {
       sqrt_var[i] = sqrt(variance[i] + .001f); 
    }
#endif

    for(i = 0; i < spatial; ++i){
        for(f = 0; f < filters; ++f){
            int index = i*filters+ f;
            //x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .001f));
            x[index] = (x[index] - mean[f])/sqrt_var[f];
        }
    }
}

void forward_batchnorm_layer(shadownet *net, float *input, float *output) {
    // TODO set outputs, out_c, out_h, out_w
    int size = net->h * net->w;
    int chn = net->units;

    MEASURE_TIME_HEAD;

    DMSG("%s size:%d, chn:%d, %bn_output:%p",__func__,size, chn,net->bn_output);
    DMSG("output:%p,mean:%p,var:%p,scales:%p,bias:%p",output, net->bn_rolling_mean,net->bn_rolling_variance,net->bn_scales,net->bn_biases);
    normalize_cpu_nhwc(output, net->bn_rolling_mean, net->bn_rolling_variance, chn, size);
#ifdef NEON_MULADD
    scale_add_nhwc(output, net->bn_scales, net->bn_biases, chn, size);
#else
    scale_bias_nhwc(output, net->bn_scales, chn, size);
    add_bias_nhwc(output, net->bn_biases, chn, size);
#endif

    MEASURE_TIME_TAIL;

    DUMPW8("batchnorm output bytes:",net->outputs, output);
    DUMPW4F("batchnorm output float:",net->outputs, output);
}

#ifdef FUSE_LAYERS

#ifdef NO_MASK
void forward_bn_ac_layers(shadownet *net, float *input, float *output) {
    int  spatial = net->h * net->w;
    int filters = net->units;
    int f, i;
    float temp;
    MEASURE_TIME_HEAD;
    for(f = 0; f < filters; ++f){
        for(i = 0; i < spatial; ++i){
            int index = i*filters+ f;
            temp = (input[index] - net->bn_rolling_mean[f])/(sqrt(net->bn_rolling_variance[f] + .001f));
            temp = temp * net->bn_scales[f] + net->bn_biases[f];
            output[index] = relu6_activate(temp); 
        }
    }
    MEASURE_TIME_TAIL;
}

#else // NO_MASK
void forward_bn_ac_ama_layers(shadownet *net, float *input, float *output) {
    int  spatial = net->h * net->w;
    int filters = net->units;
    int f, i;
    float temp;
    MEASURE_TIME_HEAD;
    for(f = 0; f < filters; ++f){
        for(i = 0; i < spatial; ++i){
            int index = i*filters+ f;
            temp = (input[index] - net->bn_rolling_mean[f])/(sqrt(net->bn_rolling_variance[f] + .001f));
            temp = temp * net->bn_scales[f] + net->bn_biases[f];
            temp = relu6_activate(temp); 
            output[index] = temp + (net->am_weights_a[index] * net->rscalar_a); 
        }
    }
    MEASURE_TIME_TAIL;
}

void forward_ama_bn_ac_amb_layers(shadownet *net, float *input, float *output) {
    int  spatial = net->h * net->w;
    int filters = net->units;
    int f, i;
    float temp;
    MEASURE_TIME_HEAD;
    for(f = 0; f < filters; ++f){
        for(i = 0; i < spatial; ++i){
            int index = i*filters+ f;
            temp = input[index]+ (net->am_weights_a[index] * net->rscalar_a); 
            temp = (temp - net->bn_rolling_mean[f])/(sqrt(net->bn_rolling_variance[f] + .001f));
            temp = temp * net->bn_scales[f] + net->bn_biases[f];
            temp = relu6_activate(temp); 
            output[index] = temp + (net->am_weights_b[index] * net->rscalar_b); 
        }
    }
    MEASURE_TIME_TAIL;
}

void forward_ama_bn_ac_layers(shadownet *net, float *input, float *output) {
    int  spatial = net->h * net->w;
    int filters = net->units;
    int f, i;
    float temp;
    MEASURE_TIME_HEAD;
    for(f = 0; f < filters; ++f){
        for(i = 0; i < spatial; ++i){
            int index = i*filters+ f;
            temp = input[index]+ (net->am_weights_a[index] * net->rscalar_a); 
            temp = (temp - net->bn_rolling_mean[f])/(sqrt(net->bn_rolling_variance[f] + .001f));
            temp = temp * net->bn_scales[f] + net->bn_biases[f];
            output[index] = relu6_activate(temp); 
        }
    }
    MEASURE_TIME_TAIL;
}

void forward_ama_amb_layers(shadownet *net, float *input, float *output) {
    int  spatial = net->h * net->w;
    int filters = net->units;
    int f, i;
    float temp;
    MEASURE_TIME_HEAD;
    for(f = 0; f < filters; ++f){
        for(i = 0; i < spatial; ++i){
            int index = i*filters+ f;
            temp = input[index]+ (net->am_weights_a[index] * net->rscalar_a); 
            output[index] = temp + (net->am_weights_b[index] * net->rscalar_b); 
        }
    }
    MEASURE_TIME_TAIL;
}

#endif // NO_MASK
#endif // FUSE_LAYERS

void forward_avgpool_layer(shadownet *net, float *input, float *output)
{
    int H, W, C;
    int i,k;
    int out_index,in_index;
    MEASURE_TIME_HEAD;

    H = net->h;
    W = net->w;
    C = net->units;

    // TODO set out_w = 1 & out_h = 1 for the following layer
    net->inputs = C;

    for(k = 0; k < C; ++k){
        out_index = k;
        output[out_index] = 0;
        for(i = 0; i < H*W; ++i){
            in_index = k + C*i; 
            output[out_index] += input[in_index];
        }
        output[out_index] /= H*W;
    }
    MEASURE_TIME_TAIL;
    DUMPW8("avgpool output bytes:",net->outputs, output);
    DUMPW4F("avgpool output float:",net->outputs, output);
}

void forward_softmax_layer(shadownet *net, float *input, float *output)
{
    //TODO set temperature, inputs
    MEASURE_TIME_HEAD;
    softmax(input, net->inputs, net->temperature, output);
    MEASURE_TIME_TAIL;
    return;
}

void softmax(float *input, int n, float temp, float *output)
{
    int i;
    float e;
    float sum = 0;
    float largest = -FLT_MAX;
    MEASURE_TIME_HEAD;
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
    }
    for(i = 0; i < n; ++i){
        e = ta_exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
    }
    MEASURE_TIME_TAIL;
    return;
}
