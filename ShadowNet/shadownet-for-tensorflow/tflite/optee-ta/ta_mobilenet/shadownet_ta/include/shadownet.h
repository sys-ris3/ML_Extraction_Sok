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
//#define NO_MASK // remove mask layers, so we can preload all weights
#define TEST_DYN_SHM // test dynamic memory sharing, no weights loading, no memory copy

#define MAX_AM_WEIGHTS_LEN (112*112*64*4)
#define MAX_INPUT_LEN (112*112*76*4)
#define MAX_CHN_NUM     (1024)

#ifdef TEST_DYN_SHM
#define MEMPOOL_SIZE    ((MAX_INPUT_LEN) + (MAX_AM_WEIGHTS_LEN * 2) + (9 * MAX_CHN_NUM * 4)) 
#else
#define MEMPOOL_SIZE    ((MAX_INPUT_LEN) + (MAX_AM_WEIGHTS_LEN * 3) + (9 * MAX_CHN_NUM * 4)) 
#endif

#define ALL_WEIGHTS_SIZE (1*1024*1024)
#define MB      (1*1024*1024)

/**
 * Shadownet has 5 types, according to the layer sequence.
 * SN_CONV : LT, BN, ReLU, AM, SF;
 * SN_DW: SF, AM, BN, ReLU, AM;
 * SN_PW: LT, AM, BN, ReLU, AM, SF;
 * SN_PRED: LT, AM, BN, ReLU, PL, RS, DO, AM;
 * SN_RET: LT, AM, AM, RS, SM;
 */

typedef enum{
SN_CONV, SN_DW, SN_PW, SN_PRED, SN_RET
}SHADOWNET_TYPE;

typedef enum {
LT, AM_A, AM_B, SF, BN, BIG_INPUT 
}WEIGHTS_TYPE;

#define SHADOWNET_CFG_NUM 28
typedef struct shadownet_config {
    SHADOWNET_TYPE type;
    int h, w, c, units;
} shadownet_config;

typedef enum {
CONV1 = 0,  
DWCONV1, 
PWCONV1,
DWCONV2, 
PWCONV2,
DWCONV3, 
PWCONV3,
DWCONV4, 
PWCONV4,
DWCONV5, 
PWCONV5,
DWCONV6, 
PWCONV6,
DWCONV7, 
PWCONV7,
DWCONV8, 
PWCONV8,
DWCONV9, 
PWCONV9,
DWCONV10, 
PWCONV10,
DWCONV11, 
PWCONV11,
DWCONV12,
PWCONV12,
DWCONV13,
PWCONV13,
RESULTS
}SHADOWNET_CFG_IDX;

/* use static memory allocation to allocate
 * buffer for all possible layer's weights and output
 * to avoid memory fragmentation
 */
typedef struct mempool{
    /* at most two Add Mask layer: A and B */
    float *AM_WEIGHTS_A;
    float *AM_WEIGHTS_B;

    /* one Linear Transform layer */
    int *LT_OBF_WEIGHTS;
    float *LT_BIAS_WEIGHTS;

    /* one Shuffle Channel layer */
    int *SF_OBF_WEIGHTS;
    float *SF_BIAS_WEIGHTS;

    /* one Batchnorm layer */
    float *BN_SCALE_WEIGHTS;
    float *BN_BIAS_WEIGHTS;
    float *BN_MEAN_WEIGHTS;
    float *BN_VAR_WEIGHTS;

    /* we rotate output buf between layers */
    float *OUTPUT_A;
    float *OUTPUT_B;
    float *INOUT; /*network input buffer can be used as output buf too*/
    
} mempool;

/* shadow_net is a fused layer of all continuous non-linear layers 
 * after a linear layer in Normal World.
 */
typedef struct shadownet{
    SHADOWNET_TYPE type; 
    SHADOWNET_CFG_IDX sn_idx;
    int initialized;
    int h, w, c; 
    int units;
    int out_h, out_w, out_c; 

    int net_inputs; /* for the first layer */
    int layer_inputs; /* for all following layers */
    int outputs;
    int inputs;

    /* batchnorm */
    float temperature;
    float probability;
    float scale;

    /* add_mask */
    float *am_weights_a;
    float rscalar_a;

    float *am_weights_b;
    float rscalar_b;

    /* linear_transform & shuffle_channel */
    float *lt_rbias;
    int *lt_obfweights;
    float *sf_rbias;
    int *sf_obfweights;

    /* batchnorm weights */
    float * bn_biases;
    float * bn_scales;
    float * bn_rolling_mean;
    float * bn_rolling_variance;

    /* net input & output */
    float *input;
    float *output;

    /* layer output */
    float * am_output_a;
    float * am_output_b;
    float * lt_output;
    float * bn_output;
    float * sf_output;
    float * avg_output;
    float * ac_output;
    float * sm_output;
}shadownet;

SHADOWNET_TYPE get_sn_type_from_cfg_idx(SHADOWNET_CFG_IDX sn_idx); 
int get_output_len(SHADOWNET_CFG_IDX sn_idx); 
int get_input_len(SHADOWNET_CFG_IDX sn_idx);

#ifdef NO_MASK
int init_shadownets(void);
int init_all_weights(void);
#else
int init_mempool(mempool* pool);
void init_shadownet_config(void); 
void init_sn_weights(shadownet *net);
int init_shadownet(shadownet *net);
#endif // NO_MASK

// subnets
int network_predict(shadownet *net, float *inout);
int forward_sn_conv(shadownet *net, float *input);
int forward_sn_dw(shadownet *net, float *input);
int forward_sn_pw(shadownet *net, float *input);
int forward_sn_pred(shadownet *net, float *input);
int forward_sn_ret(shadownet *net, float *input);

// layers
void forward_linear_transform_layer(shadownet *net, float *input, float *output);
void forward_shuffle_channel_layer(shadownet *net, float *input, float *output);
void forward_add_mask_layer_a(shadownet *net, float *input, float *output);
void forward_add_mask_layer_b(shadownet *net, float *input, float *output);
void forward_relu6_layer(shadownet *net, float *input, float *output); 
void forward_batchnorm_layer(shadownet *net, float *input, float *output);
void add_bias_nhwc(float *output, float *biases, int n, int size);
void scale_bias_nhwc(float *output, float *scales, int n, int size);
void normalize_cpu_nhwc(float *x, float *mean, float *variance, int filters, int spatial);
void forward_avgpool_layer(shadownet *net, float *input, float *output);
void softmax(float *input, int n, float temp, float *output);
void forward_softmax_layer(shadownet *net, float *input, float *output); 
int sqrt_buf(float *input, float *output, int size); 
int neon_muladd(float *x, float *s, float *y, float *z, int size); 
int neon_muladd_fixed_scalar(float *x, float s, float *y, float *z, int size);
int neon_relu6(float *input, float *output, int size); 

#ifdef FUSE_LAYERS
#ifdef NO_MASK
void forward_bn_ac_layers(shadownet *net, float *input, float *output); 
#else // NO_MASK
void forward_bn_ac_ama_layers(shadownet *net, float *input, float *output); 
void forward_ama_bn_ac_amb_layers(shadownet *net, float *input, float *output); 
void forward_ama_bn_ac_layers(shadownet *net, float *input, float *output); 
void forward_ama_amb_layers(shadownet *net, float *input, float *output); 
#endif // NO_MASK
#endif // FUSE_LAYERS

#ifdef __cplusplus
}
#endif
#endif
