#include "ca_cmds.h"
#include "utils.h"

void setup_tee_session(void);
void teardown_tee_session(void);
TEEC_Context ctx;
TEEC_Session sess;
TEEC_SharedMemory shm;
int tee_initialized = 0;

int parse_network_cfg_ca_cmd(char *cfg_buf) {
  TEEC_Operation op;
  TEEC_Result res;
  uint32_t origin;
  int ta_netid = -1;

  if (tee_initialized == 0) {
      setup_tee_session();
      tee_initialized = 1;
  }

  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INOUT,
                                   TEEC_NONE, TEEC_NONE);

  op.params[0].tmpref.buffer = cfg_buf;
  op.params[0].tmpref.size = strlen(cfg_buf); 

  op.params[1].value.a = -1;

  res = TEEC_InvokeCommand(&sess, CMD_PARSE_NETWORK_CFG,
                           &op, &origin);

  if (res != TEEC_SUCCESS)
  errx(1, "TEEC_InvokeCommand(PARSE_NETWORK_CFG) failed 0x%x origin 0x%x",
       res, origin);
  else
      ta_netid = op.params[1].value.a;
  
  //LOGD("ca netid:%d ", ta_netid);

  return ta_netid;
}

void load_weights_ca_cmd_unit(int ta_netid, int layer_id, void *weights, int weight_idx, size_t length, size_t offset); 

#define MB  (1024*1024)

void load_weights_ca_cmd(int ta_netid, int layer_id, void *weights, int weight_idx, size_t length)
{
  void *cur = NULL;
  int offset, left, payload;

  //LOGD("load_weights_ca_cmd");
  //LOGD("load_weights: weigths:%p, length:%d,ta_netid:%d,lid:%d,idx:%d,length:%d",weights,length,ta_netid,layer_id,weight_idx,length);
  if (length > MB) {
      left = length;
      offset = 0;
      while (left > 0) {
          if (left > MB) {
            payload = MB; 
          } else
              payload = left;

          cur = (uint8_t *)weights + offset;
          load_weights_ca_cmd_unit(ta_netid, layer_id, cur, weight_idx, payload, offset);
          offset += payload;
          left -= payload;
      }
  } else {
    load_weights_ca_cmd_unit(ta_netid, layer_id, weights, weight_idx, length, 0);
  }
  return;
      
}

void load_weights_ca_cmd_unit(int ta_netid, int layer_id, void *weights, int weight_idx, size_t length, size_t offset) {
  TEEC_Operation op;
  TEEC_Result res;
  uint32_t origin;

  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_INOUT,
                                   TEEC_VALUE_INOUT, TEEC_VALUE_INOUT);

  //LOGD("load_weights: weigths:%p, length:%d,ta_netid:%d,lid:%d,idx:%d,length:%d, offset:%d ",weights,length,ta_netid,layer_id,weight_idx,length,offset);
  op.params[0].tmpref.buffer = weights;
  op.params[0].tmpref.size = length; 

  op.params[1].value.a = ta_netid;
  op.params[1].value.b = layer_id;

  op.params[2].value.a = weight_idx;
  op.params[2].value.b = length;

  op.params[3].value.a = offset;

  res = TEEC_InvokeCommand(&sess, CMD_LOAD_WEIGHTS,
                           &op, &origin);

  if (res != TEEC_SUCCESS)
  errx(1, "TEEC_InvokeCommand(LOAD_WEIGHTS) failed 0x%x origin 0x%x",
       res, origin);
}

void network_predict_ca_cmd(int ta_netid, void *input, int in_len, void *output, int out_len)
{
  TEEC_Operation op;
  TEEC_Result res;
  uint32_t origin;
  //LOGD("predict ca cmd : input:%p, length:%d,ta_netid:%d,output:%p,len:%d",input,in_len,ta_netid,output,out_len);

  memset(&op, 0, sizeof(op));
  //LOGD("%s, after memset, after input",__func__);
  //op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_PARTIAL_INOUT,
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INOUT, TEEC_NONE,
                                   TEEC_VALUE_INOUT, TEEC_NONE);
  //LOGD("%s, after params, after input",__func__);

  op.params[0].tmpref.buffer = input;
  op.params[0].tmpref.size = in_len; 

  //shm.buffer = output;
  //shm.size = out_len;
  //shm.flags = TEEC_MEM_INPUT|TEEC_MEM_OUTPUT;
  //res = TEEC_RegisterSharedMemory(&ctx, &shm);
  //if (res != TEEC_SUCCESS) {
  //  LOGD("TEEC_InvokeCommand(ReigsterSharedMemory) failed 0x%x origin 0x%x",
  //     res, origin);
  //  return;
  //} else
  //    LOGD("register shared memory success!");

  //LOGD("%s, before invoke, after input",__func__);
 // op.params[1].memref.parent = &shm;
 // op.params[1].memref.offset = 0;
 // op.params[1].memref.size = out_len;
 // op.params[1].memref.size = out_len;

  op.params[2].value.a = ta_netid;
  op.params[2].value.b = out_len;

  //LOGD("%s, before invoke",__func__);
  res = TEEC_InvokeCommand(&sess, CMD_NETWORK_PREDICT,
                           &op, &origin);

  if (res != TEEC_SUCCESS)
  LOGD("TEEC_InvokeCommand(NETWORK_PREDICT) failed 0x%x origin 0x%x",
       res, origin);

  // reuse input buffer for output in tee and then copy out
  memcpy(output, input, out_len);
  //LOGD("return predict");

  //LOGD("close tee to release memory");
  teardown_tee_session();
  tee_initialized = 0;
}

void setup_tee_session(void)
{
    TEEC_UUID uuid = TA_TEE_SHADOW_UUID;
    TEEC_Result res;
    uint32_t origin;

    /* Initialize a context connecting us to the TEE */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

    /* Open a session with the TA */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS)
    errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
         res, origin);


}

void teardown_tee_session(void)
{
    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
}

