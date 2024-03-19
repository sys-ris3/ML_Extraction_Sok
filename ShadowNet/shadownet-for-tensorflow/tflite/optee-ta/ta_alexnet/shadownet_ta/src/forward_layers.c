#include <float.h>
#include <shadownet.h>
#include <math_ta.h>
#include <tee_internal_api_extensions.h>
#include <utee_defines.h>

//#define DEBUG_WEIGHTS
//#define DEBUG_TIME
#define sqrt sqrtf
//#define NEON_MULADD
//#define NEON_SQRT 
//#define NEON_RELU

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


extern shadownet_config sdw_cfg[SHADOWNET_CFG_NUM];

void forward_linear_transform_layer(shadownet *net, float *input, float *output)
{
    
    int H = sdw_cfg[net->sn_idx].h;
    int W = sdw_cfg[net->sn_idx].w;
    int M = sdw_cfg[net->sn_idx].c;
    int N = sdw_cfg[net->sn_idx].units;

    int h, w, n;
    int idx_from, idx_rand;
    float scalar;

    MEASURE_TIME_HEAD;

    //DMSG("%s h:%d,w:%d,m:%d,n:%d, input:%p, output:%p",__func__, H, W, M, N, input, output);
    for (h = 0; h < H; ++h) {
        for (w = 0; w < W; ++w) {
            for (n = 0; n < N; ++n) {
                idx_from = net->lt_obfweights[n];
                idx_rand = net->lt_obfweights[N + n];
                scalar = net->lt_rscale[n];
                if (idx_from >M || idx_rand >= M)
                    DMSG("idx_from:%d,idx_rand:%d,scalar:%f",idx_from,idx_rand,scalar);

                output[(h * W * N) + (w * N) + n] = 
                  input[(h * W * M) + (w * M) + idx_from] * scalar +  
                    input[(h * W * M) + (w * M) + idx_rand] + net->lt_rbias[n]; 
            }
        }
                //DMSG("%s h:%d,w:%d,n:%d",__func__, h, w, n);
    }
    //DMSG("%s h:%d,w:%d,n:%d",__func__, h, w, n);
    
    MEASURE_TIME_TAIL;

    DUMPW8("linear_transform output bytes:",net->outputs, output);
    DUMPW4F("linear_transform output float:",net->outputs, output);
}

void forward_add_mask_layer_a(shadownet *net, float *input, float *output)
{
    // TODO set inputs
    
    int k;
    if (net->sn_idx == CONV1)
         k = sdw_cfg[net->sn_idx].oh * sdw_cfg[net->sn_idx].ow * sdw_cfg[net->sn_idx].units;
    else
         k = sdw_cfg[net->sn_idx].h * sdw_cfg[net->sn_idx].w * sdw_cfg[net->sn_idx].units;

    MEASURE_TIME_HEAD;

    DMSG("%s enter",__func__);

#ifdef NEON_MULADD
    neon_muladd_fixed_scalar(net->am_weights_a, net->rscalar_a, input, output, k); 
#else
    int j;
    for (j = 0; j < k; ++j) {
        //if (j % 1000 == 0)
        //    DMSG("%s j: %d, k:%d",__func__,j,  k);
        output[j] = input[j] + net->am_weights_a[j] * net->rscalar_a; 
    }
#endif

    DMSG("%s eixt",__func__);

    MEASURE_TIME_TAIL;

    DUMPW8("add_mask output bytes:",k, output);
    DUMPW4F("add_mask output float:",k, output);
}

void forward_add_mask_layer_b(shadownet *net, float *input, float *output)
{
    // TODO set inputs
    int k = sdw_cfg[net->sn_idx].oh * sdw_cfg[net->sn_idx].ow * sdw_cfg[net->sn_idx].units;

    MEASURE_TIME_HEAD;

    DMSG("%s enter",__func__);

#ifdef NEON_MULADD
    neon_muladd_fixed_scalar(net->am_weights_b, net->rscalar_b, input, output, k); 
#else
    int j;
    for (j = 0; j < k; ++j) {
        //if (j % 1000 == 0)
        //    DMSG("%s j: %d, k:%d",__func__,j,  k);
        output[j] = input[j] + net->am_weights_b[j] * net->rscalar_b; 
    }
#endif

    MEASURE_TIME_TAIL;

    DMSG("%s exit",__func__);

    DUMPW8("add_mask output bytes:",k, output);
    DUMPW4F("add_mask output float:",k, output);
}

static inline float relu_activate(float x){return (x < 0.) ? 0 : x;}

void forward_relu_layer(shadownet *net, float *input, float *output) {
    int i, n;

    MEASURE_TIME_HEAD;

    // TODO set inputs
    n = sdw_cfg[net->sn_idx].h * sdw_cfg[net->sn_idx].w * sdw_cfg[net->sn_idx].units;
    DMSG("%s n:%d",__func__, n);
#ifdef NEON_RELU
   neon_relu(input, output, n); 
#else
    for(i = 0; i < n; ++i){
        //if (i % 10000 == 0)
        //    DMSG("%s i:%d",__func__, i);
        output[i] = relu_activate(input[i]);
    }
#endif

    MEASURE_TIME_TAIL;
}

static inline get_max_of_four(float a, float b, float c, float d) {
    float x = a > b ? a: b;
    float y = c > d ? c: d;
    return x > y ? x : y;
}

// TODO: only consider stride == 2!
// like 55X55 after pooling: 27X27
void forward_maxpool_layer(shadownet *net, float *input, float *output)
{
    int H, W, C, OH, OW;
    int i,j,k;
    int out_index,in_index;
    int p1,p2,p3,p4;
    float a, b, c ,d;
    MEASURE_TIME_HEAD;


    H = sdw_cfg[net->sn_idx].h;
    W = sdw_cfg[net->sn_idx].w;
    OH = sdw_cfg[net->sn_idx].oh;
    OW = sdw_cfg[net->sn_idx].ow;
    C = sdw_cfg[net->sn_idx].units;

    DMSG("%s H: %d, OH: %d C:%d enter",__func__, H, OH, C);

    // TODO: assume NHWC
    for(i = 0; i < OH; ++i){
        for (j = 0; j < OW; ++j) {
            for (k = 0; k < C; ++k) {
                out_index = i * OW * C + j * C + k;
                p1 = (i*2) * W * C + (j * 2) * C + k;
                p2 = (i*2 + 1) * W * C + (j * 2) * C + k;
                p3 = (i*2) * W * C + (j * 2 + 1) * C + k;
                p4 = (i*2 + 1) * (W) * C + (j * 2 + 1) * C + k;
        //        if (i==0 && j < 1 && k < 10)
         //           DMSG("%s i : %d, p4:%d, input:%p",__func__, i, p4, input);
                a = input[p1]; 
                b = input[p2]; 
                c = input[p3]; 
                d = input[p4]; 
                output[out_index] = get_max_of_four(a,b,c,d);
            }
        }
       // DMSG("%s i : %d, p4:%d, input:%p",__func__, i, p4, input);
    }
    MEASURE_TIME_TAIL;
    DMSG("%s exit",__func__);
    DUMPW8("maxpool output bytes:",net->outputs, output);
    DUMPW4F("maxpool output float:",net->outputs, output);
}

void forward_softmax_layer(shadownet *net, float *input, float *output)
{
    //TODO set temperature, inputs
    MEASURE_TIME_HEAD;
    softmax(input, sdw_cfg[net->sn_idx].units, 1.0, output);
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
