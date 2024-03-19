#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <tee_internal_api.h>

#include "activation_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "dropout_layer.h"
#include "list.h"
#include "option_list.h"
#include "parser.h"
#include "softmax_layer.h"
#include "add_mask_layer.h"
#include "linear_transform_layer.h"
#include "shuffle_channel_layer.h"
#include "utils.h"

typedef struct{
    char *type;
    list *options;
}section;


//list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[add_mask]")==0) return ADD_MASK;
    if (strcmp(type, "[linear_transform]")==0) return LINEAR_TRANSFORM;
    if (strcmp(type, "[shuffle_channel]")==0) return SHUFFLE_CHANNEL;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    network *net;
} size_params;

layer parse_add_mask(list *options, size_params params)
{
    layer l = make_add_mask_layer(params.batch, params.h, params.w, params.c);
    return l;
}

layer parse_linear_transform(list *options, size_params params)
{
    int units = option_find_int(options, "units",1);
    layer l = make_linear_transform_layer(params.batch, params.h, params.w, params.c, units);
    return l;
}

layer parse_shuffle_channel(list *options, size_params params)
{
    layer l = make_shuffle_channel_layer(params.batch, params.h, params.w, params.c);
    return l;
}

layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    layer l = make_softmax_layer(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, "temperature", 1);
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, "spatial", 0);
    l.noloss =  option_find_int_quiet(options, "noloss", 0);
    return l;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) DMSG("Layer before avgpool layer must output image.");

    int use_nhwc = option_find_int(options, "nhwc", 0);
    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    layer.nhwc = use_nhwc;
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_batchnorm(list *options, size_params params)
{
    int load_bias = option_find_int(options, "bn_bias", 0);
    int use_nhwc = option_find_int(options, "nhwc", 0);
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
    l.bn_bias = load_bias;
    l.nhwc = use_nhwc;
    return l;
}

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;

    return l;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    if(!net->inputs && !(net->h && net->w && net->c)) DMSG("No input parameters supplied");

}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}

network *parse_network_cfg_from_buf(char *cfg_buf)
{
    DMSG("%s\n",__func__);

    list *sections = read_cfg_from_buf(cfg_buf);
    node *n = sections->front;
    if(!n) DMSG("Config file has no sections");
    network *net = make_network(sections->size - 1);
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) DMSG("First section must be [net] or [network]");
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.net = net;

    n = n->next;
    int count = 0;
    free_section(s);
    //fprintf(stderr, "layer     filters    size              input                output\n");
    DMSG("layer     filters    size              input                output\n");
    while(n){
        params.index = count;
        //fprintf(stderr, "%5d ", count);
        DMSG("%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == ADD_MASK){
            l = parse_add_mask(options, params);
        }else if(lt == LINEAR_TRANSFORM){
            l = parse_linear_transform(options, params);
        }else if(lt == SHUFFLE_CHANNEL){
            l = parse_shuffle_channel(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
        }else{
            //fprintf(stderr, "Type not recognized: %s\n", s->type);
            DMSG("Type not recognized: %s\n", s->type);
        }
        option_unused(options);
        net->layers[count] = l;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->output = out.output;
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    return net;
}

#if 0
network *parse_network_cfg_wrapper(char *filename)
{
    char *cfg_buf = read_file(filename);
    printf("%s %s\n",__func__, cfg_buf);
    list *sections = read_cfg_from_buf(cfg_buf);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.net = net;

    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n){
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == ADD_MASK){
            l = parse_add_mask(options, params);
        }else if(lt == LINEAR_TRANSFORM){
            l = parse_linear_transform(options, params);
        }else if(lt == SHUFFLE_CHANNEL){
            l = parse_shuffle_channel(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        option_unused(options);
        net->layers[count] = l;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->output = out.output;
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    return net;
}

network *parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, net);

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.net = net;

    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output\n");
    while(n){
        params.index = count;
        fprintf(stderr, "%5d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == ADD_MASK){
            l = parse_add_mask(options, params);
        }else if(lt == LINEAR_TRANSFORM){
            l = parse_linear_transform(options, params);
        }else if(lt == SHUFFLE_CHANNEL){
            l = parse_shuffle_channel(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net->layers[count-1].output;
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        option_unused(options);
        net->layers[count] = l;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->output = out.output;
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    return net;
}
#endif

char* sp = NULL; /* the start position of the string */
 
char* strtok(char* str, const char* delimiters); 
char* strtok(char* str, const char* delimiters) {
 
    int i = 0;
    int len = strlen(delimiters);
    DMSG("strtok, len:%d",len);
 
    /* check in the delimiters */
    if(len == 0)
        DMSG("delimiters are empty\n");
 
    /* if the original string has nothing left */
    if(!str && !sp)
        return NULL;
 
    /* initialize the sp during the first call */
    if(str && !sp)
        sp = str;
 
    /* find the start of the substring, skip delimiters */
    char* p_start = sp;
    while(true) {
        for(i = 0; i < len; i ++) {
            if(*p_start == delimiters[i]) {
                p_start ++;
                break;
            }
        }
        DMSG("while, len:%d",len);
 
        if(i == len) {
               sp = p_start;
               break;
        }
    }
 
    /* return NULL if nothing left */
    if(*sp == '\0') {
        sp = NULL;
        return sp;
    }
 
    /* find the end of the substring, and
        replace the delimiter with null */
    while(*sp != '\0') {
        for(i = 0; i < len; i ++) {
            if(*sp == delimiters[i]) {
                *sp = '\0';
                break;
            }
        }
        DMSG("while, len:%d",len);
 
        sp ++;
        if (i < len)
            break;
    }
 
    return p_start;
}

list *read_cfg_from_buf(char * cfg_buf)
{
    char *line, *cfg_item;
    const char delim[] = "\n";
    int nu = 0;
    DMSG("read_cfg_from_buf");
    list *options = make_list();
    section *current = 0;
    cfg_item = strtok(cfg_buf, delim);
    while(cfg_item != NULL) {
        line = malloc(strlen(cfg_item) + 1);
        strcpy(line, cfg_item);
        DMSG("line:%s\n",line);
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    //fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    DMSG("Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
        cfg_item = strtok(NULL, delim);
    }
    return options;
}

#if 0
list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

void save_add_mask_weights(layer l, FILE *fp)
{
    fwrite(l.rscalar, sizeof(float), 1, fp);
    fwrite(l.weights, sizeof(float), l.outputs, fp);
}

void save_linear_transform_weights(layer l, FILE *fp)
{
    fwrite(l.obfweights, sizeof(int), l.units*2, fp);
    fwrite(l.rbias, sizeof(float), l.units, fp);
}

void save_shuffle_channel_weights(layer l, FILE *fp)
{
    fwrite(l.obfweights, sizeof(int), l.units, fp);
    fwrite(l.rbias, sizeof(float), l.units, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
    fwrite(l.scales, sizeof(float), l.c, fp);
    if (l.bn_bias== 1) {
        fwrite(l.biases, sizeof(float), l.c, fp);
    }
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_weights_upto(network *net, char *filename, int cutoff)
{
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        } if(l.type == ADD_MASK){
            save_add_mask_weights(l, fp);
        } if(l.type == LINEAR_TRANSFORM){
            save_linear_transform_weights(l, fp);
        } if(l.type == SHUFFLE_CHANNEL){
            save_shuffle_channel_weights(l, fp);
        }
    }
    fclose(fp);
}
void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}

void load_add_mask_weights(layer l, FILE *fp)
{
    fread(l.weights, sizeof(float), l.outputs, fp);
    DUMPW8("add_mask w",l.outputs*4, l.weights);
    fread(l.rscalar, sizeof(float), 1, fp);
    DUMPW8("add_mask r",4, l.rscalar);
}

void load_linear_transform_weights(layer l, FILE *fp)
{
    fread(l.obfweights, sizeof(int), l.units*2, fp);
    DUMPW8("linear_transform obfw",l.units*8, l.obfweights);
    fread(l.rbias, sizeof(float), l.units, fp);
    DUMPW8("linear_transform rbias",l.units*4,l.rbias);
}

void load_shuffle_channel_weights(layer l, FILE *fp)
{
    fread(l.obfweights, sizeof(int), l.c, fp);
    DUMPW8("shuffle_channel obfw",l.c*4, l.obfweights);
    fread(l.rbias, sizeof(float), l.c, fp);
    DUMPW8("shuffle_channel rbias",l.c*4, l.rbias);
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    DUMPW4F("bn scales",l.c, l.scales);
    if (l.bn_bias == 1) {
        printf("bn_bias = 1,load biases\n");
        fread(l.biases, sizeof(float), l.c, fp);
        DUMPW4F("bn biases",l.c, l.biases);
    }
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    DUMPW4F("bn mean",l.c, l.rolling_mean);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
    DUMPW4F("bn variance",l.c, l.rolling_variance);
}

void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if(l.type == ADD_MASK){
            load_add_mask_weights(l, fp);
        }
        if(l.type == LINEAR_TRANSFORM){
            load_linear_transform_weights(l, fp);
        }
        if(l.type == SHUFFLE_CHANNEL){
            load_shuffle_channel_weights(l, fp);
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
    }
    fprintf(stderr, "Done!\n");
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}

#endif
