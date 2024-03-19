#include <err.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <tee_client_api.h>
#include "ca_cmds.h"
#include "shadownet_ca.h"

#define WT_BUF_SIZE      (6*1024*1024)
#define INPUT_BUF_SIZE  (112*112*76*4)
#define OUTPUT_BUF_SIZE  (112*112*76*4)

void setup_tee_session(void);
void teardown_tee_session(void);
TEEC_Context ctx;
TEEC_Session sess;
TEEC_SharedMemory shm;
TEEC_SharedMemory shmout;
int tee_initialized = 0;
uint8_t *masks_buf = NULL;
size_t masks_size = 0;
extern clock_t t;
extern double time_taken;
double tee_time = 0.0;


const int INPUT_SZ_LIMIT = 10 * 1024 * 1024;

size_t ReadFile(char *name, char **buffer);
size_t ReadFile(char *name, char **buffer)
{
    FILE *file;
    size_t fileLen;

    //Open file
    file = fopen(name, "rb");
    if (!file)
    {
        LOGD("Unable to open file %s", name);
        return 0;
    }
    
    //Get file length
    fseek(file, 0, SEEK_END);
    fileLen=ftell(file);
    fseek(file, 0, SEEK_SET);

    LOGD("file length: %ld\n", fileLen);

    //Allocate memory
    *buffer =(char *)malloc(fileLen+1);
    if (!(*buffer))
    {
        LOGD("Memory error!");
        fclose(file);
        return 0;
    }


    //Read file contents into buffer
    fread(*buffer, fileLen, 1, file);
    fclose(file);
    return fileLen;
}

// load model or masks
// CMD:
//TA_SHADOWNET_CMD_LOAD_MODEL
//TA_SHADOWNET_CMD_LOAD_MASKS
int load_weights_ca_cmd(char *weights_path, unsigned CMD) {
    TEEC_Operation op;
    TEEC_Result res;
    uint32_t err_origin;
    char *model_buffer;
    size_t total_len;
    unsigned weights_len;
    unsigned offset;

    if (tee_initialized == 0) {
        setup_tee_session();
        tee_initialized = 1;
    }

    if (CMD == TA_SHADOWNET_CMD_LOAD_MODEL) {
        total_len = ReadFile(weights_path, &model_buffer);
        if (total_len == 0)
            return -1;
    } else { // handle masks
        if (masks_buf == NULL) {
            masks_size = ReadFile(weights_path, &masks_buf);
            if (masks_size == 0)
                return -1;
        }
        //assert(masks_size > 0);
        model_buffer = masks_buf;
        total_len = masks_size;
    }


	memset(&op, 0, sizeof(op));
    offset = 0;
    while (offset < total_len) {
        shm.buffer = model_buffer + offset;

        weights_len = ((total_len - offset) > WT_BUF_SIZE) ? WT_BUF_SIZE : (total_len - offset); 
        shm.size = weights_len;

        LOGD("total_len:%d weights_len :%d", total_len, weights_len);
        shm.flags = TEEC_MEM_INPUT;
        res = TEEC_RegisterSharedMemory(&ctx, &shm);
        if (res != TEEC_SUCCESS) {
            LOGD("TEEC_InvokeCommand(RegisterSharedMemory) failed 0x%x, \n", res);
            return -1;
        }

        op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INPUT, TEEC_VALUE_INOUT,
                                         TEEC_VALUE_INOUT, TEEC_NONE);
        op.params[0].memref.parent = &shm;
        op.params[0].memref.offset = 0;
        op.params[0].memref.size = weights_len;
        op.params[1].value.a = weights_len; // weights len 
        op.params[1].value.b = offset; // weights len 
        op.params[2].value.a = total_len;
        if (total_len - offset <= WT_BUF_SIZE) // last chunk
            op.params[2].value.b = 1;
        else
            op.params[2].value.b = 0;
            
    
        res = TEEC_InvokeCommand(&sess, CMD, &op, &err_origin);
        if (res != TEEC_SUCCESS)
        	errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
        		res, err_origin);
    
        TEEC_ReleaseSharedMemory(&shm);
        offset += WT_BUF_SIZE;
    }
    LOGD("TA_SHADOWNET_CMD_LOAD_MODEL/MASKS succeed!\n");
    return 0;
}

int network_predict_ca_cmd(unsigned op_id, model_type_t mtype, unsigned input_size, void *input, unsigned output_size, void *output)
{
  TEEC_Operation op;
  TEEC_Result res;
  uint32_t err_origin;

  //LOGD("%s input_size:%u output_size:%u\n", __func__, input_size, output_size);
  memset(&op, 0, sizeof(op));

  shm.buffer = input;
  shm.size = input_size;
  shm.flags = TEEC_MEM_INPUT;
  res = TEEC_RegisterSharedMemory(&ctx, &shm);
  if (res != TEEC_SUCCESS) {
      LOGD("TEEC_InvokeCommand(RegisterSharedMemory) shm failed 0x%x\n", res);
      return -1;
  }

  shmout.buffer = output;
  shmout.size = output_size;
  shmout.flags = TEEC_MEM_OUTPUT;
  res = TEEC_RegisterSharedMemory(&ctx, &shmout);
  if (res != TEEC_SUCCESS) {
      LOGD("TEEC_InvokeCommand(RegisterSharedMemory) shmout failed 0x%x\n", res);
      goto out1;
  }

  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INPUT, TEEC_MEMREF_PARTIAL_OUTPUT,
                                     TEEC_VALUE_INOUT, TEEC_NONE);
  op.params[0].memref.parent = &shm;
  op.params[0].memref.offset = 0;
  op.params[0].memref.size = input_size;

  op.params[1].memref.parent = &shmout;
  op.params[1].memref.offset = 0;
  op.params[1].memref.size = output_size;

  op.params[2].value.a = op_id;
  op.params[2].value.b = (unsigned)mtype;

  if (op_id == 0) {
      TOTAL_TIME(tee_time);
  }

  CLOCK_START;
  res = TEEC_InvokeCommand(&sess, TA_SHADOWNET_CMD_INFERENCE, &op,
  			 &err_origin);
  CLOCK_END(tee_time, "invoke command"); 
  if (res != TEEC_SUCCESS)
  	errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
  		res, err_origin);

  // copy results
  //output_size = op.params[2].value.a;
  //memcpy(output, temp_output_buf, output_size);
  //LOGD("copy model output, size:%d", output_size);
  
  TEEC_ReleaseSharedMemory(&shmout);
  TEEC_ReleaseSharedMemory(&shm);

  return 0;

out1:
  TEEC_ReleaseSharedMemory(&shm);
  return -1;
}




void network_predict_ca_cmd_multinputs(unsigned op_id, model_type_t model_type, int num_inputs, unsigned input_lens[] , void * inputs[],  unsigned output_size, void *output)
{
  TEEC_Operation op;
  TEEC_Result res;
  uint32_t err_origin;

  //LOGD("%s input_size:%u output_size:%u\n", __func__, input_size, output_size);
  memset(&op, 0, sizeof(op));

  // total length of the buffer in bytes
  int total_len = 0 ;
  // 4 bytes in the front indicating the number of inputs
  total_len += 4;
  for (int i = 0 ; i < num_inputs; i++){
      total_len += input_lens[i];
      // for each input, needs 4 bytes in the header to store its length
      total_len += 4;
  }
  // append the \0 to the end
  total_len += 1;
  if (total_len >= INPUT_SZ_LIMIT){
      LOGD("requiring size of buffer %d exceeds input limit %d\n", total_len, INPUT_SZ_LIMIT);
      return -1;
  }
  //allocate the construct the buffer
  uint64_t* buf_ptr = (uint64_t*)malloc(total_len * sizeof(char));

  *buf_ptr = (uint64_t)num_inputs;
  buf_ptr ++;
  for (int i = 0 ; i < num_inputs; i++){
      *buf_ptr = (uint64_t) input_lens[i];
      buf_ptr++;
  }
  buf_ptr = (unsigned char*)buf_ptr;
  for (int i = 0 ; i < num_inputs; i++){
      memcpy((void*)buf_ptr, inputs[i],input_lens[i]);
      buf_ptr += input_lens[i];
  }
  *buf_ptr = 0x00;
  
  

  shm.buffer = (void *)buf_ptr;
  shm.size = total_len;
  shm.flags = TEEC_MEM_INPUT;
  res = TEEC_RegisterSharedMemory(&ctx, &shm);
  if (res != TEEC_SUCCESS) {
      LOGD("TEEC_InvokeCommand(RegisterSharedMemory) shm failed 0x%x\n", res);
      return -1;
  }

  shmout.buffer = output;
  shmout.size = output_size;
  shmout.flags = TEEC_MEM_OUTPUT;
  res = TEEC_RegisterSharedMemory(&ctx, &shmout);
  if (res != TEEC_SUCCESS) {
      LOGD("TEEC_InvokeCommand(RegisterSharedMemory) shmout failed 0x%x\n", res);
      goto out1;
  }

  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INPUT, TEEC_MEMREF_PARTIAL_OUTPUT,
                                     TEEC_VALUE_INOUT, TEEC_NONE);
  op.params[0].memref.parent = &shm;
  op.params[0].memref.offset = 0;
  op.params[0].memref.size = input_size;

  op.params[1].memref.parent = &shmout;
  op.params[1].memref.offset = 0;
  op.params[1].memref.size = output_size;

  op.params[2].value.a = op_id;
  op.params[2].value.b = (unsigned)mtype;

  if (op_id == 0) {
      TOTAL_TIME(tee_time);
  }

  CLOCK_START;
  res = TEEC_InvokeCommand(&sess, TA_SHADOWNET_CMD_MULTINPUTS_INFERENCE, &op,
  			 &err_origin);
  CLOCK_END(tee_time, "invoke command"); 
  if (res != TEEC_SUCCESS)
  	errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
  		res, err_origin);

  // copy results
  //output_size = op.params[2].value.a;
  //memcpy(output, temp_output_buf, output_size);
  //LOGD("copy model output, size:%d", output_size);
  
  TEEC_ReleaseSharedMemory(&shmout);
  TEEC_ReleaseSharedMemory(&shm);
  free(buf_ptr)
  
  return 0;

out1:
  TEEC_ReleaseSharedMemory(&shm);
  free(buf_ptr);
  return -1;
}




void setup_tee_session(void)
{
    TEEC_UUID uuid = TA_TEE_SHADOW_UUID;
    TEEC_Result res;
    uint32_t origin;

    /* Initialize a context connecting us to the TEE */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS) {
      LOGD("TEEC_InitializeContext failed with code 0x%x", res);
      return;
    }

    /* Open a session with the TA */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS) {
      LOGD("TEEC_Opensession failed with code 0x%x origin 0x%x",res, origin);
      return;
    }

    LOGD("sunzc:TEEC_Opensession succeed!");
}

void teardown_tee_session(void)
{
    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
}

