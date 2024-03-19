#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <utee_defines.h>
#include <shadownet.h>
#include <hello_world_ta.h>

int eval(void);

// use mpool.OUTPUT_B for input size > 2MB
extern mempool mpool;
extern void *big_output;
extern int flag_bigoutput;
int output_left, output_offset;

#ifdef NO_MASK
extern int nets_initialized;
extern shadownet sdw_nets[SHADOWNET_CFG_NUM];
#else
extern shadownet sdw_net;
#endif

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
  int type = 0;
  int weight_idx = 0;
  int weights_len = 0;
  int offset = 0;
  void *weights, *cur;
  shadownet *net;
  SHADOWNET_CFG_IDX sn_idx;

  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT);
  DMSG("load weights has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  weights  = params[0].memref.buffer;
  type = params[1].value.b;
  weight_idx = params[2].value.a;
  weights_len = params[2].value.b;
  offset = params[3].value.a;
  sn_idx = params[3].value.b;

  DMSG("weights:%p, type :%d, widx:%d, wlen:%d, sn_idx:%d",weights,type, weight_idx, weights_len, sn_idx);

#ifdef NO_MASK
  net = &sdw_nets[sn_idx];
#else
  net = &sdw_net;
#endif

  if (type == AM_A || type == AM_B) {
      if (weight_idx == 1) {
        if (type == AM_A)
            cur = (uint8_t *)net->am_weights_a + offset;
        else
            cur = (uint8_t *)net->am_weights_b + offset;

        copy_float(weights, cur, weights_len/sizeof(float));
      } else if(weight_idx == 2) {
        if (type == AM_A)
            net->rscalar_a = ((float *)weights)[0];
        else
            net->rscalar_b = ((float *)weights)[0];
      } else {
          DMSG("Error! wrong weight_idx:%d for layer:%d\n",weight_idx, type);
      }
  } else if (type == LT) {
      if (weight_idx == 1) {
        cur = (uint8_t *)net->lt_obfweights + offset;
        copy_int(weights, cur, weights_len/sizeof(int));
      } else if(weight_idx == 2) {
        cur = (uint8_t *)net->lt_rbias+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else {
          DMSG("Error! wrong weight_idx:%d for layer:%d\n",weight_idx, type);
      }
  } else if (type == SF) {
      DMSG("copy shuff chan weights");
      if (weight_idx == 1) {
        cur = (uint8_t *)net->sf_obfweights+ offset;
        copy_int(weights, cur, weights_len/sizeof(int));
      } else if(weight_idx == 2) {
        cur = (uint8_t *)net->sf_rbias+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else {
          DMSG("Error! wrong weight_idx:%d for layer:%d\n",weight_idx, type);
      }
  } else if (type == BN) {
      if (weight_idx == 1) {
        cur = (uint8_t *)net->bn_scales+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else if(weight_idx == 2) {
        cur = (uint8_t *)net->bn_biases+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else if (weight_idx == 3) {
        cur = (uint8_t *)net->bn_rolling_mean+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else if(weight_idx == 4) {
        cur = (uint8_t *)net->bn_rolling_variance+ offset;
        copy_float(weights, cur, weights_len/sizeof(float));
      } else {
          DMSG("Error! wrong weight_idx:%d for layer:%d\n",weight_idx, type);
      }
  } else if (type == BIG_INPUT) { // when input size is bigger than 2MB
        cur = (uint8_t *)mpool.OUTPUT_B + offset;
        copy_float(weights, cur, weights_len/sizeof(float));
  } else {
      DMSG("Error! wrong layer for layer:%d\n",type);
  }
  return TEE_SUCCESS;

}

static TEE_Result network_predict_ta_cmd(uint32_t param_types, 
                                    TEE_Param params[4])
{
  float *inout;
  SHADOWNET_CFG_IDX sn_idx;
  WEIGHTS_TYPE type;
  shadownet *net;
  DMSG("network predict has been called");
  TEE_Time start, end, time;

  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_NONE );
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  inout = params[0].memref.buffer;
  sn_idx = params[2].value.b;
  type = params[2].value.a;
  DMSG("input:%p sn_idx:%d, type:%d\n",inout, sn_idx, type);

#ifdef NO_MASK
  net = &sdw_nets[sn_idx];
#else
  net = &sdw_net;
#endif

TEE_GetSystemTime(&start);
  if ( type == BIG_INPUT) { // BIG INPUT use OUTPUT_B as input buffer
    flag_bigoutput = 1;
    output_left = get_output_len(sn_idx);
    output_offset = 0;
    network_predict(net, mpool.OUTPUT_B);
  } else
    network_predict(net, (float*)inout);
TEE_GetSystemTime(&end);

TEE_TIME_SUB(end, start, time);

// measure pure computation time
params[2].value.a = time.seconds*1000 + time.millis;
  //EMSG("sn_idx:%d, time:%d ms\n", sn_idx, params[2].value.a);

  // reset flag
  flag_bigoutput = 0;

  return TEE_SUCCESS;
}

static fetch_results_ta_cmd(uint32_t param_types, 
                                    TEE_Param params[4])
{
  DMSG("fetch results has been called");
  float *output;

  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE );
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  output = params[0].memref.buffer;

  if (big_output == NULL || output_left == 0) {
      EMSG("ERROR in fetch results, big_output(%p) == NULL or left(%d) == 0!",big_output, output_left);
  }

  if (output_left > MB) {
    output_left = output_left - MB;
    params[1].value.a = MB;
    params[1].value.b = output_left;
    copy_float((float *)((char*)big_output + output_offset), output, MB/sizeof(float));
    output_offset += MB;
  } else {
    params[1].value.a = output_left;
    params[1].value.b = 0;
    copy_float((float *)((char*)big_output + output_offset), output, output_left/sizeof(float));
    output_left = 0;
    output_offset = 0;
    big_output = NULL;
  }
  
  return TEE_SUCCESS;
}

static TEE_Result init_shadownet_ta_cmd(uint32_t param_types, 
                                    TEE_Param  __maybe_unused params[4])
{
  int ret = 0;
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE,
                                             TEE_PARAM_TYPE_NONE );
  DMSG("parse has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

#ifdef NO_MASK
    if (nets_initialized == 0) {
        ret = init_shadownets();
        if (ret != 0) 
          EMSG("init_shadownet failed!");
        else
            nets_initialized = 1;
    } else
        EMSG("shadownets already initialized!");
#else
  shadownet *net;
  net = &sdw_net;
  net->sn_idx = params[0].value.a;

  ret = init_shadownet(net); 
  if (ret != 0) 
    EMSG("init_shadownet failed!");
#endif // NO_MASK

//  eval();

  return TEE_SUCCESS;
}

TEE_Result TA_InvokeCommandEntryPoint(void __maybe_unused *sess_ctx,
                                      uint32_t cmd_id,
                                      uint32_t param_types, TEE_Param params[4])
{
    (void)&sess_ctx; /* Unused parameter */
   DMSG(" entryp point has been called");

    switch (cmd_id) {
        case CMD_INIT_SHADOWNET:
        return init_shadownet_ta_cmd(param_types, params);

        case CMD_LOAD_WEIGHTS:
        return load_weights_ta_cmd(param_types, params);

        case CMD_NETWORK_PREDICT:
        return network_predict_ta_cmd(param_types, params);

        case CMD_FETCH_RESULTS:
        return fetch_results_ta_cmd(param_types, params);

        default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
