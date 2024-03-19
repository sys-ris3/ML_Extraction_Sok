#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <shadownet.h>
#include <hello_world_ta.h>
//#include <stdlib.h>

void *model_buf = NULL;
void *masks_buf = NULL;
size_t model_size = 0;
size_t masks_size = 0;
bool model_initialized = false;
bool masks_initialized = false;

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

static TEE_Result load_model_ta_cmd(uint32_t param_types, 
                                    TEE_Param params[4])
{

  void *weights = NULL;
  unsigned weights_len = 0;
  unsigned weights_offset = 0;
  unsigned model_len= 0;
  unsigned is_last_chunk = 0;
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_NONE);
  DMSG("load weights has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  weights = params[0].memref.buffer;
  weights_len = params[1].value.a;
  weights_offset = params[1].value.b;
  model_len = params[2].value.a;
  is_last_chunk = params[2].value.b;

  DMSG("model len:%u",model_len);
  DMSG("weights:%p, wlen:%u",weights,weights_len);
  DMSG("woffset:%u",weights_offset);
  DMSG("is_last_chunk :%u",is_last_chunk);


  if(model_len >= MAX_MODEL_SIZE) {
    DMSG("ERROR! weights len is bigger than maximum buffer size!");
    return TEE_ERROR_BAD_PARAMETERS;
  }

  // we transfer model in 1MB chunks, needs several cmds, first chunk 
  if (model_initialized == false && model_buf == NULL) {
    // allocate buf for model and copy it from nomal world
    model_size = model_len;
    model_buf = (void *)TEE_Malloc(model_size, TEE_MALLOC_FILL_ZERO);
    if (model_buf == NULL) {
      model_initialized = false;
      model_size = 0;
      DMSG("model_buf alloc fail!");
      return TEE_ERROR_OUT_OF_MEMORY;
    } 
    model_initialized = true;
  }

  // TODO check offset
  TEE_MemMove(model_buf+weights_offset, weights, weights_len);
  DMSG("model buffer offset:%u ,len:%u copied!",weights_offset, weights_len);

  if (is_last_chunk == 1) { // all model buffer copied, ready to start
    // initialize tflite model
    initialize_tflite_model_from_buffer(model_buf);
    init_output_buf(); 
    init_resnet_plan();// initialize it anyway 
    DMSG("model_buf fully copied! Ready to go!");
  }

  return TEE_SUCCESS;
}

// for performance evaluation only
static TEE_Result fake_load_masks_ta_cmd(uint32_t param_types, 
                                    TEE_Param params[4])
{

  void *masks= NULL;
  unsigned masks_len = 0;
  unsigned masks_offset = 0;
  unsigned total_masks_len= 0;
  unsigned is_last_chunk = 0;
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_NONE);
  DMSG("load masks has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  masks = params[0].memref.buffer;
  masks_len = params[1].value.a;
  masks_offset = params[1].value.b;
  total_masks_len = params[2].value.a;
  is_last_chunk = params[2].value.b;

  DMSG("masks len:%u",masks_len);
  DMSG("masks:%p, mlen:%u",masks, total_masks_len);
  DMSG("moffset:%u",masks_offset);
  DMSG("is_last_chunk :%u",is_last_chunk);


  if(total_masks_len > MAX_MASKS_SIZE) {
    DMSG("ERROR! masks len is bigger than maximum buffer size!");
    return TEE_ERROR_BAD_PARAMETERS;
  }

  // we transfer model in chunks, needs several cmds, first chunk 
  if (masks_initialized == false && masks_buf == NULL) {
    // allocate buf for model and copy it from nomal world
    masks_size = total_masks_len;
    masks_initialized = true;
  }


  fake_update_masks(masks, masks_len);

  return TEE_SUCCESS;
}

static TEE_Result load_masks_ta_cmd(uint32_t param_types, 
                                    TEE_Param params[4])
{

  void *masks= NULL;
  unsigned masks_len = 0;
  unsigned masks_offset = 0;
  unsigned total_masks_len= 0;
  unsigned is_last_chunk = 0;
  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_VALUE_INOUT,
                                             TEE_PARAM_TYPE_NONE);
  DMSG("load masks has been called");
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  masks = params[0].memref.buffer;
  masks_len = params[1].value.a;
  masks_offset = params[1].value.b;
  total_masks_len = params[2].value.a;
  is_last_chunk = params[2].value.b;

  DMSG("masks len:%u",masks_len);
  DMSG("masks:%p, mlen:%u",masks, total_masks_len);
  DMSG("moffset:%u",masks_offset);
  DMSG("is_last_chunk :%u",is_last_chunk);


  if(total_masks_len > MAX_MASKS_SIZE) {
    DMSG("ERROR! masks len is bigger than maximum buffer size!");
    return TEE_ERROR_BAD_PARAMETERS;
  }

  // we transfer model in chunks, needs several cmds, first chunk 
  if (masks_initialized == false && masks_buf == NULL) {
    // allocate buf for model and copy it from nomal world
    masks_size = total_masks_len;
    masks_buf = (void *)TEE_Malloc(total_masks_len, TEE_MALLOC_FILL_ZERO);
    if (masks_buf == NULL) {
      masks_initialized = false;
      masks_size = 0;
      DMSG("masks_buf alloc fail!");
      return TEE_ERROR_OUT_OF_MEMORY;
    } 
    masks_initialized = true;
  }

  // TODO check offset
  TEE_MemMove(masks_buf+masks_offset, masks, masks_len);
  DMSG("masks buffer offset:%u ,len:%u copied!",masks_offset, masks_len);

  if (is_last_chunk == 1) { // all model buffer copied, ready to start
    // initialize tflite model
    update_masks(masks_buf, masks_size);
    DMSG("masks_buf fully copied and updated!");
  }

  return TEE_SUCCESS;
}

typedef enum model_type_t {
    MOBILENET,
    MINIVGG,
    RESNET,
    ALEXNET
}model_type_t;

static TEE_Result shadownet_inference_ta_cmd(uint32_t param_types, 
                                    TEE_Param params[4])
{
  unsigned op_id = 0;
  unsigned model_type = 0;
  float *input = NULL;
  float *output = NULL;

  DMSG("network predict has been called");


  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,TEE_PARAM_TYPE_MEMREF_OUTPUT,
            TEE_PARAM_TYPE_VALUE_INOUT, TEE_PARAM_TYPE_NONE);
  if (param_types != exp_param_types)
  return TEE_ERROR_BAD_PARAMETERS;

  input = params[0].memref.buffer;
  output = params[1].memref.buffer;
  output = input;
  op_id = params[2].value.a; // for resnet, it's plan id
  model_type = params[2].value.b;
  DMSG("input:%p output:%p op_id:%d, model_type(resnet=2,mb=0):%d\n",input, output, op_id, model_type);

  
  if (model_type == RESNET) {
    resnet_inference(op_id, input, output);
  } else {
    shadownet_inference(op_id, input, output);
  }

  return TEE_SUCCESS;
}

static TEE_Result shadownet_inference_ta_multinputs_cmd(uint32_t param_types, TEE_Param params[4]){
  unsigned op_id = 0;
  unsigned model_type = 0;
  float **inputs = NULL;
  float *output = NULL;

  DMSG("network predict has been called");


  uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INPUT,TEE_PARAM_TYPE_MEMREF_OUTPUT,
            TEE_PARAM_TYPE_VALUE_INOUT, TEE_PARAM_TYPE_NONE);
  if (param_types != exp_param_types)
    return TEE_ERROR_BAD_PARAMETERS;

  //input = params[0].memref.buffer;
  uint64_t * buf_ptr = params[0].memref.buffer;
  uint64_t num_inputs = *buf_ptr;
  inputs = (float **)TEE_Malloc(num_inputs * sizeof(float *), TEE_MALLOC_FILL_ZERO);
  if (!inputs) {
          IMSG("Cannot malloc - out of memory");
  }

  buf_ptr += 1;
  uint8_t * buf_pass_header_ptr = (uint8_t* )(buf_ptr + num_inputs);
  
  int i = 0;
  for(i = 0; i < num_inputs, i ++){
    inputs[i] = (float*) buf_pass_header_ptr;
    buf_pass_header_ptr += *buf_ptr;
    buf_ptr ++;
  }

  

  output = params[1].memref.buffer;
  //output = input;
  op_id = params[2].value.a; // for resnet, it's plan id
  model_type = params[2].value.b;

  
  if (model_type == RESNET) {
    resnet_inference(op_id, input, output);
  } else {
    shadownet_inference_multinputs(op_id,(int) num_inputs, inputs, output);
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
        case TA_HELLO_WORLD_CMD_INC_VALUE:
        return TEE_SUCCESS;

        case TA_SHADOWNET_CMD_LOAD_MODEL:
        return load_model_ta_cmd(param_types, params);

        case TA_SHADOWNET_CMD_LOAD_MASKS:
        return fake_load_masks_ta_cmd(param_types, params);

        case TA_SHADOWNET_CMD_INFERENCE:
        return shadownet_inference_ta_cmd(param_types, params);

        case TA_SHADOWNET_CMD_MULTINPUTS_INFERENCE:
        return shadownet_inference_ta_multinputs_cmd(param_types,params);

        default:
        return TEE_ERROR_BAD_PARAMETERS;
    }
}
