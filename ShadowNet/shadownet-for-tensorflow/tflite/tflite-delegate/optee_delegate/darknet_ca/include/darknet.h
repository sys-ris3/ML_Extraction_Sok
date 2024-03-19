#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234

typedef enum{
    RELU, NOACT, RELU6
} ACTIVATION;

typedef enum {
    SOFTMAX,
    DROPOUT,
    AVGPOOL,
    ACTIVE,
    BATCHNORM,
    NETWORK,
    ADD_MASK,
    LINEAR_TRANSFORM,
    SHUFFLE_CHANNEL,
    BLANK
} LAYER_TYPE;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    void (*forward)   (struct layer, struct network);
    int batch_normalize;
    int batch;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int groups;
    int size;
    int stride;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int index;
    int binary;
    int noloss;
    int softmax;
    int total;


    float temperature;
    float probability;
    float scale;

    /* custom op layer weights*/
    /* add_mask */
    float *rscalar;
    /* linear_transform & shuffle_channel */
    float *rbias;
    int *obfweights;
    int units;
    /* NHWC(tflite) to NCHW(darknet) */
    int nhwc;
    /* batchnorm layer has bias weights */
    int bn_bias;

    float * biases;
    float * scales;
    float * weights;
    float * output;

    float * rolling_mean;
    float * rolling_variance;
};

void free_layer(layer);

typedef struct network{
    int ta_netid;

    int n;
    int batch;
    size_t *seen;
    int *t;
    layer *layers;
    float *output;

    int inputs;
    int outputs;
    int h, w, c;

    float *input;
    int index;

    int num_inputs;
    float ** real_inputs;
} network;


network *load_network(char *cfg, char *weights, int clear);

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);

void forward_network(network *net);
void forward_network_multinputs(network *net);
void set_batch_network(network *net, int b);


void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void normalize_cpu_nhwc(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

layer get_network_output_layer(network *net);
float *network_predict(network *net, float *input);
float *network_predict_multinputs(network *net,int num_inputs, float ** input);

int network_width(network *net);
int network_height(network *net);

void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(float *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
void free_list(list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);

network *load_network_ca(char *cfg, char *weights, int clear); 
float *network_predict_ca(network *net, float *input); 
#ifdef __cplusplus
}
#endif
#endif
