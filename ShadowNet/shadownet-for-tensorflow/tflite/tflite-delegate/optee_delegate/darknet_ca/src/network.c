#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "utils.h"
#include "blas.h"

#include "activation_layer.h"
#include "batchnorm_layer.h"
#include "avgpool_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "add_mask_layer.h"
#include "linear_transform_layer.h"
#include "shuffle_channel_layer.h"
#include "parser.h"
#include "ca_cmds.h"

network *parse_network_cfg_ca(char *cfg); // declare before use!!!
network *load_network_ca(char *cfg, char *weights, int clear); 
network *load_network_ca(char *cfg, char *weights, int clear) {
    network *net;
    net  = parse_network_cfg_ca(cfg);
    if (net == NULL) {
        return NULL;
    }
    if(weights && weights[0] != 0){
        //LOGD("load weights: net:%p",net);
        load_weights_ca(net, weights);
        //LOGD("after load weights: net:%p",net);
    }
    if(clear) (*net->seen) = 0;
    //LOGD("return load network,%p", net);
    return net;
}

network *load_network_ca_shadownet(char *cfg, char *weights, int clear); 
network *load_network_ca_shadownet(char *cfg, char *weights, int clear) {
    network *net;
    net  = parse_network_cfg_ca_shadownet(cfg);
    if (net == NULL) {
        return NULL;
    }
    if(weights && weights[0] != 0){
        //LOGD("load weights: net:%p",net);
        load_weights_ca(net, weights);
        //LOGD("after load weights: net:%p",net);
    }
    if(clear) (*net->seen) = 0;
    //LOGD("return load network,%p", net);
    return net;
}

network *load_network(char *cfg, char *weights, int clear)
{
    //network *net = parse_network_cfg(cfg);
    network *net = parse_network_cfg_wrapper(cfg);
    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }
    if(clear) (*net->seen) = 0;
    return net;
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case ACTIVE:
            return "activation";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DROPOUT:
            return "dropout";
        case BATCHNORM:
            return "batchnorm";
        case ADD_MASK:
            return "add_mask";
        case LINEAR_TRANSFORM:
            return "linear_transform";
        case SHUFFLE_CHANNEL:
            return "shuffle_channel";
        default:
            break;
    }
    return "none";
}

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    if (net == NULL) {
        LOGD("net calloc fail!\n");
        return NULL;
    }
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    //LOGD("net:%p,net->seen:%p,net->t:%p\n", net, net->seen, net->t);
    return net;
}

void forward_network(network *netp)
{
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        l.forward(l, net);
        net.input = l.output;
    }
}

void forward_network_multinputs(network *netp)
{
    network net = *netp;
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        l.forward(l, net);
        net.input = l.output;
    }
}

int get_predicted_class_network(network *net)
{
    return max_index(net->output, net->outputs);
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}

float *network_predict_ca(network *net, float *input) {
    //LOGD("%s net:%p",__func__, net);
    network_predict_ca_cmd(net->ta_netid, input,net->inputs * sizeof(float), net->output, net->outputs*sizeof(float));
    return net->output;
}

float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}


float *network_predict_multinputs(network *net, int num_inputs, float ** real_inputs)
{
    network orig = *net;

    net->num_inputs = num_inputs;
    net->real_inputs = real_inputs;
    forward_network_multinputs(net);
    float *out = net->output;
    *net = orig;
    return out;
}


int network_width(network *net){return net->w;}
int network_height(network *net){return net->h;}

layer get_network_output_layer(network *net)
{
    return net->layers[net->n-1];
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    free(net);
}


layer network_output_layer(network *net)
{
    return net->layers[net->n - 1];
}

int network_inputs(network *net)
{
    return net->layers[0].inputs;
}

int network_outputs(network *net)
{
    return network_output_layer(net).outputs;
}

float *network_output(network *net)
{
    return network_output_layer(net).output;
}
