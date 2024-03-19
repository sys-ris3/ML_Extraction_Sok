/*
 * Copyright (c) 2016, Linaro Limited
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <err.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* OP-TEE TEE client API (built by optee_client) */
#include <tee_client_api.h>

/* To the the UUID (found the the TA's h-file(s)) */
#include <hello_world_ta.h>

#define WT_BUF_SIZE      (6*1024*1024)
#define INPUT_BUF_SIZE  (112*112*76*4)
#define OUTPUT_BUF_SIZE  (112*112*76*4)

size_t ReadFile(char *name, char **buffer);
size_t ReadFile(char *name, char **buffer)
{
    FILE *file;
    size_t fileLen;

    //Open file
    file = fopen(name, "rb");
    if (!file)
    {
        fprintf(stderr, "Unable to open file %s", name);
        return -1;
    }
    
    //Get file length
    fseek(file, 0, SEEK_END);
    fileLen=ftell(file);
    fseek(file, 0, SEEK_SET);

    printf("file length: %ld\n", fileLen);

    //Allocate memory
    *buffer =(char *)malloc(fileLen+1);
    if (!(*buffer))
    {
        fprintf(stderr, "Memory error!");
        fclose(file);
        return -1;
    }


    //Read file contents into buffer
    fread(*buffer, fileLen, 1, file);
    fclose(file);
    return fileLen;
}


int main(void)
{
	TEEC_Result res;
	TEEC_Context ctx;
	TEEC_Session sess;
	TEEC_Operation op;
        TEEC_SharedMemory shm;
        TEEC_SharedMemory shmout;
	TEEC_UUID uuid = TA_HELLO_WORLD_UUID;
	uint32_t err_origin;
        char *model_buffer;
        size_t buflen;
        unsigned offset;
        unsigned weights_len;
        void *input_buf;
        void *output_buf;
        unsigned op_id;
        //int i;
        //size_t lenarr[5] = {30*1024*1024, 8*1024*1024,4*1024*1024,1*1024*1024, 5*1024};

	/* Initialize a context connecting us to the TEE */
	res = TEEC_InitializeContext(NULL, &ctx);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_InitializeContext failed with code 0x%x", res);



	/*
	 * Open a session to the "hello world" TA, the TA will print "hello
	 * world!" in the log when the session is created.
	 */
	res = TEEC_OpenSession(&ctx, &sess, &uuid,
			       TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
	if (res != TEEC_SUCCESS)
		errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
			res, err_origin);

	/*
	 * Execute a function in the TA by invoking it, in this case
	 * we're incrementing a number.
	 *
	 * The value of command ID part and how the parameters are
	 * interpreted is part of the interface provided by the TA.
	 */

        /* allocate buffer and read model into buffer*/
        
        buflen = ReadFile("/vendor/etc/mobilenet_model.tflite", &model_buffer);
        if (buflen == -1)
            printf("ReadFile mobilenet.tflite failed!\n");
        else
            printf("ReadFile mobilenet.tflite succeed! filelen:%lu!\n", buflen);

	/* Clear the TEEC_Operation struct */
	memset(&op, 0, sizeof(op));

        /* register shared memory buffer */
        
        
        offset = 0;
        while (offset < buflen) {
            shm.buffer = model_buffer + offset;
            weights_len = ((buflen-offset) > WT_BUF_SIZE)?WT_BUF_SIZE:(buflen-offset); 
            shm.size = weights_len;
            shm.flags = TEEC_MEM_INPUT;
            res = TEEC_RegisterSharedMemory(&ctx, &shm);
            printf("register shared memory with size:%d\n",WT_BUF_SIZE);
            if (res != TEEC_SUCCESS) {
                printf("TEEC_InvokeCommand(RegisterSharedMemory) failed 0x%x, try load input\n", res);
            } else {
                printf("TEEC_InvokeCommand(RegisterSharedMemory) succeed!\n");
            }

            op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INPUT, TEEC_VALUE_INOUT,
                                             TEEC_VALUE_INOUT, TEEC_NONE);
            op.params[0].memref.parent = &shm;
            op.params[0].memref.offset = 0;
            op.params[0].memref.size = weights_len;
            op.params[1].value.a = weights_len; // weights len 
            op.params[1].value.b = offset; // weights len 
            op.params[2].value.a = buflen;
            if (buflen - offset <= WT_BUF_SIZE) // last chunk
                op.params[2].value.b = 1;
            else
                op.params[2].value.b = 0;
                
    
    	    res = TEEC_InvokeCommand(&sess, TA_SHADOWNET_CMD_LOAD_MODEL, &op,
    
    	    			 &err_origin);
    	    if (res != TEEC_SUCCESS)
    	    	errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
    	    		res, err_origin);
    
            TEEC_ReleaseSharedMemory(&shm);
            offset += WT_BUF_SIZE;
        }

        op_id = 13; // operator_id for model inference
        input_buf = malloc(INPUT_BUF_SIZE);
        output_buf = malloc(OUTPUT_BUF_SIZE);
        if (input_buf == NULL || output_buf == NULL) {
            printf("ERROR! malloc failed!\n");
            return -1;
        }

        shm.buffer = input_buf;
        shm.size = INPUT_BUF_SIZE;
        shm.flags = TEEC_MEM_INPUT;
        res = TEEC_RegisterSharedMemory(&ctx, &shm);
        printf("register shared memory with size:%d\n",INPUT_BUF_SIZE);
        if (res != TEEC_SUCCESS) {
            printf("TEEC_InvokeCommand(RegisterSharedMemory) failed 0x%x, try load input\n", res);
        } else {
            printf("TEEC_InvokeCommand(RegisterSharedMemory) succeed!\n");
        }

        shmout.buffer = output_buf;
        shmout.size = OUTPUT_BUF_SIZE;
        shmout.flags = TEEC_MEM_OUTPUT;
        res = TEEC_RegisterSharedMemory(&ctx, &shmout);
        printf("register shared memory with size:%d\n",OUTPUT_BUF_SIZE);
        if (res != TEEC_SUCCESS) {
            printf("TEEC_InvokeCommand(RegisterSharedMemory) shmout failed 0x%x, try load input\n", res);
        } else {
            printf("TEEC_InvokeCommand(RegisterSharedMemory) shmout succeed!\n");
        }

        op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INPUT, TEEC_MEMREF_PARTIAL_OUTPUT, TEEC_VALUE_INOUT,
                                         TEEC_NONE);
        op.params[0].memref.parent = &shm;
        op.params[0].memref.offset = 0;
        op.params[0].memref.size = INPUT_BUF_SIZE;

        op.params[1].memref.parent = &shmout;
        op.params[1].memref.offset = 0;
        op.params[1].memref.size = OUTPUT_BUF_SIZE;

        op.params[2].value.a = op_id;

    	res = TEEC_InvokeCommand(&sess, TA_SHADOWNET_CMD_INFERENCE, &op,
    
    				 &err_origin);
    	if (res != TEEC_SUCCESS)
    		errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
    			res, err_origin);
    
	/*
	 * We're done with the TA, close the session and
	 * destroy the context.
	 *
	 * The TA will print "Goodbye!" in the log when the
	 * session is closed.
	 */
        TEEC_ReleaseSharedMemory(&shm);
        TEEC_ReleaseSharedMemory(&shmout);
	TEEC_CloseSession(&sess);
	TEEC_FinalizeContext(&ctx);

	return 0;
}
