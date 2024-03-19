#ifndef SHADOWNET_API
#define SHADOWNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <tee_internal_api.h>

#ifdef __cplusplus
extern "C" {
#endif

//#define FUSE_LAYERS

#define MB      (1*1024*1024)
#define MAX_INPUT_LEN (64*64*38*4)

#define MEMPOOL_SIZE   (10*MB)

typedef enum {
LT, AM_A, AM_B, BN
}WEIGHTS_TYPE;

#define SHADOWNET_CFG_NUM 7 
typedef struct shadownet_config {
    int h, w, c, units; // input height, width, channel
    int oh, ow; // output height, width
    int has_maxpool;
} shadownet_config;

typedef enum {
CONV1 = 0,  
CONV2, 
CONV3,
CONV4, 
CONV5,
CONV6, 
RESULTS
}SHADOWNET_CFG_IDX;

/* use static memory allocation to allocate
 * buffer for all possible layer's weights and output
 * to avoid memory fragmentation
 * 
 * here, mempool records pointer to layer input/output buffer
 */
typedef struct mempool{
    /* we rotate output buf between layers */
    float *OUTPUT_A;
    float *OUTPUT_B;
    float *INOUT; /*network input buffer can be used as output buf too*/
} mempool;

/* shadow_net is a fused layer of all continuous non-linear layers 
 * after a linear layer in Normal World.
 */
typedef struct shadownet{
    SHADOWNET_CFG_IDX sn_idx;
    int initialized;

    int net_inputs; /* for the first layer */
    int layer_inputs; /* for all following layers */
    int outputs;
    int inputs;

    /* add_mask */
    float *am_weights_a;
    float rscalar_a;

    float *am_weights_b;
    float rscalar_b;

    /* linear_transform */
    float *lt_rscale;
    float *lt_rbias; // bias of previous conv layer
    int *lt_obfweights;

    /* batchnorm */
    float *bn_biases;
    float *bn_scales;
    float *bn_rolling_mean;
    float *bn_rolling_variance;

    /* net input & output */
    float *input;
    float *output;

}shadownet;

int init_shadownets(void);

// subnets
int network_predict(shadownet *net, float *inout);
int forward_minivgg_conv_wo_maxpool(shadownet *net, float *input);
int forward_minivgg_conv_w_maxpool(shadownet *net, float *input);
int forward_minivgg_conv1(shadownet *net, float *input);
int forward_minivgg_results(shadownet *net, float *input);

// layers
void forward_linear_transform_layer(shadownet *net, float *input, float *output);
void forward_batchnorm_layer(shadownet *net, float *input, float *output);
void forward_add_mask_layer_a(shadownet *net, float *input, float *output);
void forward_add_mask_layer_b(shadownet *net, float *input, float *output);
void forward_relu_layer(shadownet *net, float *input, float *output); 
void forward_maxpool_layer(shadownet *net, float *input, float *output);
void softmax(float *input, int n, float temp, float *output);
void forward_softmax_layer(shadownet *net, float *input, float *output); 
void forward_batchnorm_layer(shadownet *net, float *input, float *output); 
void scale_bias_nhwc(float *output, float *scales, int n, int size);
void add_bias_nhwc(float *output, float *biases, int n, int size);
void normalize_cpu_nhwc(float *x, float *mean, float *variance, int filters, int spatial);
int sqrt_buf(float *input, float *output, int size); 
int neon_muladd(float *x, float *s, float *y, float *z, int size); 
int neon_muladd_fixed_scalar(float *x, float s, float *y, float *z, int size);
int neon_relu(float *input, float *output, int size); 

#ifdef __cplusplus
}
#endif
#endif
