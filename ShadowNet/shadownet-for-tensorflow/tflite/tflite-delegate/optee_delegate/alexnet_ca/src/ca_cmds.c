#include "ca_cmds.h"
#include <shadownet_ca.h>

void setup_tee_session(void);
void teardown_tee_session(void);
TEEC_Context ctx;
TEEC_Session sess;
TEEC_SharedMemory shm;
int tee_initialized = 0;

int fetch_results_ca_cmd(void *buf, int out_len) {
  TEEC_Operation op;
  TEEC_Result res;
  uint32_t origin;
  void *cur = NULL;
  int offset, left, payload;

  offset = 0;
  left = out_len;
  cur = (char *)buf + offset;

  while(left > 0) {
    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT, TEEC_VALUE_INOUT, 
                                     TEEC_NONE, TEEC_NONE);

    op.params[0].tmpref.buffer = cur;
    if (left > MB)
        op.params[0].tmpref.size = MB; // always allow 1 MB to be copied 
    else
        op.params[0].tmpref.size = left; // always allow 1 MB to be copied 

    res = TEEC_InvokeCommand(&sess, CMD_FETCH_RESULTS,
                             &op, &origin);

    if (res != TEEC_SUCCESS) {
        LOGD( "TEEC_InvokeCommand(CMD_FETCH_RESULTS) failed 0x%x origin 0x%x", res, origin);
        break;
    }

    payload = op.params[1].value.a;
    left = op.params[1].value.b;

    offset += payload;
    cur = (char *)buf + offset; 

    //LOGD("out_len:%d offset:%d, left:%d",out_len, offset, left);
  }
  
  if (offset != out_len) {
      LOGD(" ERROR! fetch results offset(%d) != out_len(%d) ",offset, out_len);
  }
  return 0;
}

int init_shadownet_tee_ca_cmd(SHADOWNET_CFG_IDX sn_idx) {
  TEEC_Operation op;
  TEEC_Result res;
  uint32_t origin;

  //LOGD("init_shadownet_tee called sn_idx:%d ", sn_idx);

  if (tee_initialized == 0) {
      setup_tee_session();
      tee_initialized = 1;
  }

  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INOUT, TEEC_NONE,
                                   TEEC_NONE, TEEC_NONE);

  op.params[0].value.a = sn_idx;

  res = TEEC_InvokeCommand(&sess, CMD_INIT_SHADOWNET,
                           &op, &origin);

  if (res != TEEC_SUCCESS)
  LOGD( "TEEC_InvokeCommand(CMD_INIT_SHADOWNET) failed 0x%x origin 0x%x",
          res, origin);
  
  return 0;
}

void load_weights_ca_cmd(WEIGHTS_TYPE type, void *weights, int weight_idx, size_t length, SHADOWNET_CFG_IDX sn_idx)
{
  void *cur = NULL;
  int offset, left, payload;

  //LOGD("load_weights_ca_cmd");
  //LOGD("load_weights: weigths:%p, length:%d,ta_netid:%d,lid:%d,idx:%d,length:%d",weights,length,ta_netid,layer_id,weight_idx,length);
  if (length > MB) {
      left = length;
      offset = 0;
      while (left > 0) {
          if (left > MB)
              payload = MB; 
           else
              payload = left;

          cur = (uint8_t *)weights + offset;
          load_weights_ca_cmd_unit(type, cur, weight_idx, payload, offset, sn_idx);

          offset += payload;
          left -= payload;
      }
  } else
    load_weights_ca_cmd_unit(type, weights, weight_idx, length, 0, sn_idx);

  return;
}

void load_weights_ca_cmd_unit(WEIGHTS_TYPE type, void *weights, int weight_idx, size_t length, size_t offset, SHADOWNET_CFG_IDX sn_idx) {
  TEEC_Operation op;
  TEEC_Result res;
  uint32_t origin;

  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INOUT,
                                   TEEC_VALUE_INOUT, TEEC_VALUE_INOUT);

  //LOGD("load_weights: weigths:%p, length:%d,ta_netid:%d,lid:%d,idx:%d,length:%d, offset:%d ",weights,length,ta_netid,layer_id,weight_idx,length,offset);
  op.params[0].tmpref.buffer = weights;
  op.params[0].tmpref.size = length; 

  op.params[1].value.b = type;

  op.params[2].value.a = weight_idx;
  op.params[2].value.b = length;

  op.params[3].value.a = offset;
  op.params[3].value.b = sn_idx;

  res = TEEC_InvokeCommand(&sess, CMD_LOAD_WEIGHTS,
                           &op, &origin);

  if (res != TEEC_SUCCESS)
  LOGD("TEEC_InvokeCommand(LOAD_WEIGHTS) failed 0x%x origin 0x%x", res, origin);
}

void network_predict_ca_cmd(SHADOWNET_CFG_IDX sn_idx, void *input, int in_len, void *output, int out_len)
{
  TEEC_Operation op;
  TEEC_Result res;
  uint32_t origin;
  int fetch_results_needed = 0;

  memset(&op, 0, sizeof(op));

#ifdef USE_MEMREF_TEMP
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT, TEEC_NONE,
                                   TEEC_VALUE_INOUT, TEEC_NONE);

  op.params[0].tmpref.buffer = input;
  op.params[0].tmpref.size = in_len; 

#else // RegisterShredMemory
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INOUT, TEEC_NONE,
                                   TEEC_VALUE_INOUT, TEEC_NONE);
  

  shm.buffer = input;
  shm.size = in_len;
  shm.flags = TEEC_MEM_INPUT|TEEC_MEM_OUTPUT;
  res = TEEC_RegisterSharedMemory(&ctx, &shm);
  if (res != TEEC_SUCCESS) {
      LOGD("TEEC_InvokeCommand(RegisterSharedMemory) failed 0x%x, input size:%d", res, in_len);

      //fetch_results_needed = 1;
      //load_weights_ca_cmd(BIG_INPUT, input, 0, in_len, sn_idx);
      //op.params[2].value.a = BIG_INPUT;
  }

  if (fetch_results_needed == 0) {
    op.params[0].memref.parent = &shm;
    op.params[0].memref.offset = 0; 
    op.params[0].memref.size = in_len; 
  } else {
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT, TEEC_NONE,
                                     TEEC_VALUE_INOUT, TEEC_NONE);

    op.params[0].tmpref.buffer = input;
    op.params[0].tmpref.size = 1; 
  }
#endif

  op.params[2].value.b = sn_idx;


  res = TEEC_InvokeCommand(&sess, CMD_NETWORK_PREDICT,
                           &op, &origin);

  if (res != TEEC_SUCCESS)
  LOGD("TEEC_InvokeCommand(NETWORK_PREDICT) failed 0x%x origin 0x%x",
       res, origin);

  if (fetch_results_needed == 1) {
        fetch_results_ca_cmd(output, out_len);
        fetch_results_needed = 0;
  } else {
        // reuse input buffer for output in tee and then copy out
        memcpy(output, input, out_len);
  }

#if USE_MEMREF_TEMP
#else
  TEEC_ReleaseSharedMemory(&shm);
#endif

  return;
}

void setup_tee_session(void)
{
    TEEC_UUID uuid = TA_TEE_SHADOW_UUID;
    TEEC_Result res;
    uint32_t origin;

    /* Initialize a context connecting us to the TEE */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
    LOGD("TEEC_InitializeContext failed with code 0x%x", res);

    /* Open a session with the TA */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS)
    LOGD("TEEC_Opensession failed with code 0x%x origin 0x%x",res, origin);


}

void teardown_tee_session(void)
{
    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
}

