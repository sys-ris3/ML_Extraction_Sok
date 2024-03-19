#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <darknet.h>
#include <hello_world_ta.h>
#include "parser.h"

void free_network(network *);
TEE_Result TA_CreateEntryPoint(void)
{
    DMSG("has been called");

    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void)
{
    DMSG("has been called");
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types,
                                    TEE_Param __maybe_unused params[4],
                                    void __maybe_unused **sess_ctx)
{
    uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE,
                                               TEE_PARAM_TYPE_NONE);

    DMSG("has been called");

    if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

    /* Unused parameters */
    (void)&params;
    (void)&sess_ctx;

    IMSG("Hello, Secure World!\n");
    return TEE_SUCCESS;
}


void TA_CloseSessionEntryPoint(void __maybe_unused *sess_ctx)
{
    (void)&sess_ctx; /* Unused parameter */
    IMSG("Goodbye!\n");
}

#define MAX_NET_NUM 20
typedef struct{
    void *netparr[MAX_NET_NUM];
    int status[MAX_NET_NUM];
}net_bucket;

net_bucket NETS = {0};

int alloc_netid(void); 
int alloc_netid(void) {
#if 0
    int i = 0;
    while(i < MAX_NET_NUM && NETS.status[i] != 0)
        ++i;
    if (i < MAX_NET_NUM) {
        NETS.status[i] = 1; // busy!
        return i;
    } else
        return -1;
#endif
    if (NETS.status[0] != 1) {
        DMSG("free_network before set NETS!");
        //free_network(NETS.netparr[0]);
        NETS.netparr[0] = NULL;
    }

    // DEBUG , use only 1 to avoid excess memory limit.
    NETS.status[0] = 1;

    return 0;
}

static TEE_Result parse_network_cfg_ta_cmd(uint32_t param_types, 
                                    TEE_Param params[4])
{
  int ta_netid = 0;
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE );
  DMSG("parse has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  char *cfg_buf = params[0].memref.buffer;
  DMSG("cfg_buf %s", cfg_buf);
  ta_netid = alloc_netid();
  network *net;
  DMSG("after alloc_netids");

  if (ta_netid >= 0) {
    net = parse_network_cfg_from_buf(cfg_buf);
    // default batch number
    set_batch_network(net, 1);
    NETS.netparr[ta_netid] = net; 
  }
  DMSG("before return parse");

  params[1].value.a = ta_netid;

  return TEE_SUCCESS;
}

#define MAX_COPY 0x400000 // max buffer copy 4 MB

void copy_float(float *src, float *dest, int size);
void copy_float(float *src, float *dest, int size){
    int i;
    DMSG("copy_int:src:%p,dest:%p,size:%d",src,dest,size);
    for (i = 0; i < size && i < MAX_COPY; i++) {
        dest[i] = src[i];
    } 
    return;
}

void copy_int(int *src, int *dest, int size);
void copy_int(int *src, int *dest, int size){
    int i;
    DMSG("copy_int:src:%p,dest:%p,size:%d",src,dest,size);
    for (i = 0; i < size && i < MAX_COPY; i++) {
        dest[i] = src[i];
    } 
    return;
}

static TEE_Result load_weights_ta_cmd(uint32_t param_types, 
                                    TEE_Param params[4])
{
  int ta_netid = 0;
  int layer_id = 0;
  int weight_idx = 0;
  int weights_len = 0;
  int offset = 0;
  void *weights, *cur;
  network *net;

  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT);
  DMSG("load weights has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  weights  = params[0].memref.buffer;
  ta_netid = params[1].value.a;
  layer_id = params[1].value.b;
  weight_idx = params[2].value.a;
  weights_len = params[2].value.b;
  offset = params[3].value.a;
  DMSG("weights:%p, netid:%d, layer_id:%d, widx:%d, wlen:%d",weights,ta_netid,layer_id,weight_idx, weights_len);

  net = NETS.netparr[ta_netid];
  DMSG("load weights net:%p",net);

  layer l = net->layers[layer_id];
  if (l.type == ADD_MASK) {
      if (weight_idx == 1) {
        cur = (uint8_t *)l.weights + offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else if(weight_idx == 2) {
        cur = (uint8_t *)l.rscalar+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else {
          DMSG("Error! wrong weight_idx:%d for layer:%d\n",weight_idx, layer_id);
      }
  } else if (l.type == LINEAR_TRANSFORM) {
      if (weight_idx == 1) {
          // TODO Dangerous! length controlled by normal world, used for arbitrary overwrite.
       DMSG("copy linear transform weights, l.obfweights:%p", l.obfweights);
        cur = (uint8_t *)l.obfweights+ offset;
        copy_int(weights, cur, weights_len/sizeof(int));
      } else if(weight_idx == 2) {
        cur = (uint8_t *)l.rbias+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else {
          DMSG("Error! wrong weight_idx:%d for layer:%d\n",weight_idx, layer_id);
      }
  } else if (l.type == SHUFFLE_CHANNEL) {
      DMSG("copy shuff chan weights");
      if (weight_idx == 1) {
        cur = (uint8_t *)l.obfweights+ offset;
        copy_int(weights, cur, weights_len/sizeof(int));
      } else if(weight_idx == 2) {
        cur = (uint8_t *)l.rbias+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else {
          DMSG("Error! wrong weight_idx:%d for layer:%d\n",weight_idx, layer_id);
      }
  } else if (l.type == BATCHNORM) {
      if (weight_idx == 1) {
        cur = (uint8_t *)l.scales+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else if(weight_idx == 2) {
        cur = (uint8_t *)l.biases+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else if (weight_idx == 3) {
        cur = (uint8_t *)l.rolling_mean+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else if(weight_idx == 4) {
        cur = (uint8_t *)l.rolling_variance+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else {
          DMSG("Error! wrong weight_idx:%d for layer:%d\n",weight_idx, layer_id);
      }
  } else {
      DMSG("Error! wrong layer for layer:%d\n",layer_id);
  }
  return TEE_SUCCESS;

}

static TEE_Result network_predict_ta_cmd(uint32_t param_types, 
                                    TEE_Param params[4])
{
  int ta_netid = 0;
  int out_len= 0;
  float *input;
  float *output=NULL;
  float *results;
  network *net;
  DMSG("network predict has been called");
//  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
//                                             TEE_PARAM_TYPE_MEMREF_INOUT,
//                                             TEE_PARAM_TYPE_VALUE_INOUT,
//                                             TEE_PARAM_TYPE_NONE );
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_NONE );
  DMSG("network predict has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  input = params[0].memref.buffer;
  //output = params[1].memref.buffer;
  ta_netid = params[2].value.a;
  out_len = params[2].value.b;

  DMSG("input:%p, output:%p, ta_netid:%d\n",input, output, ta_netid);

  if (ta_netid >= 0 && ta_netid < MAX_NET_NUM && NETS.status[ta_netid] != 0) {
      net = NETS.netparr[ta_netid];
      DMSG("net:%p, ta_netid:%d\n",net, ta_netid);
      //  //DEBUG return 
      //return TEE_SUCCESS;
      results = network_predict(net, (float*)input);
      DMSG("outlen: %d, l.outputs:%d",out_len,net->outputs);
      //copy_float(results, output, net->outputs);
      copy_float(results, input, net->outputs);
      free(results);
      DMSG("before free network, netid:%d",ta_netid);
      free_network(NETS.netparr[ta_netid]);
  } else {
      DMSG("Wrong ta_netid:%d\n",ta_netid);
  }

  return TEE_SUCCESS;
}

TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4])
{
    (void)&sess_ctx; /* Unused parameter */
   DMSG(" entryp point has been called");

    switch (cmd_id) {
        case CMD_PARSE_NETWORK_CFG:
        return parse_network_cfg_ta_cmd(param_types, params);

        case CMD_LOAD_WEIGHTS:
        return load_weights_ta_cmd(param_types, params);

        case CMD_NETWORK_PREDICT:
        return network_predict_ta_cmd(param_types, params);

        default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
