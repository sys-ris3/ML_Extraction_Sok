/*
 * ModelSafe Project. 
 * May, 2020
 * Zhichuang Sun(sun.zhi@husky.neu.edu)
 */
#include <err.h>
#include "tee_delegate.h"

// tee delegate status: 1(initialized), -1(not init)
int tee_initialized = -1;

TEEC_Context ctx;
TEEC_Session sess;

teeDelegateStatus tee_linear_transform(uint32_t *pres, size_t B, size_t H, size_t W, size_t M, size_t N,
       const float* x_flat, const int* w_flat, const float* r_flat, float* output_flat) {
    TEEC_Result res;
    TEEC_Operation op;
    uint32_t err_origin;

    // tflite op parameters blob
    struct tee_linear_transform_blob tee_blob;

    // check if tee delegate is initialized
	if (tee_initialized != 1) {
		res = tee_delegate_init();
    	if (res != TEEC_SUCCESS) {
			printf("tee_delegate_init() failed!");
            *pres = res; 
			return teeDelegateNoinit;
		}
	}

    /*
     * Execute a function in the TA by invoking it, in this case
     * we're incrementing a number.
     *
     * The value of command ID part and how the parameters are
     * interpreted is part of the interface provided by the TA.
     */

    /* Clear the TEEC_Operation struct */
    memset(&op, 0, sizeof(op));

    /*
     * Prepare the argument. Pass a value in the first parameter,
     * the remaining three parameters are unused.
     */
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INOUT, TEEC_NONE,
                                     TEEC_NONE, TEEC_NONE);
    op.params[0].value.a = 42;


    /*  
     * TA_HELLO_WORLD_CMD_INC_VALUE is the actual function in the TA to be
     * called.
     */
    printf("Invoking TA to increment %d\n", op.params[0].value.a);
    res = TEEC_InvokeCommand(&sess, TA_HELLO_WORLD_CMD_INC_VALUE, &op,
                             &err_origin);
    if (res != TEEC_SUCCESS) {
            errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
                    res, err_origin);
            *pres = res; 
			return teeDelegateFail;
    }
    printf("TA incremented value to %d\n", op.params[0].value.a);
    *pres = op.params[0].value.a;

    /* Fill tflite op parameters tee_blob */
    tee_blob.B = B;

    /* TODO create new command in TA to handle different ops */

    return teeDelegateOk;
}

TEEC_Result tee_delegate_init(){
    TEEC_Result res;
    TEEC_UUID uuid = TA_HELLO_WORLD_UUID;
    uint32_t err_origin;

    /* Initialize a context connecting us to the TEE */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
    {
            //errx(1, "TEEC_InitializeContext failed with code 0x%x", res);
            return res;
    }

    /*
     * Open a session to the "hello world" TA, the TA will print "hello
     * world!" in the log when the session is created.
     */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                           TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
    if (res != TEEC_SUCCESS)
    {
            //errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x", res, err_origin);
            return 2;
    }

    tee_initialized = 1;
    return TEEC_SUCCESS;
}

TEEC_Result tee_delegate_exit(){
    /*  
     * We're done with the TA, close the session and
     * destroy the context.
     *
     * The TA will print "Goodbye!" in the log when the
     * session is closed.
     */
    tee_initialized = -1;

    TEEC_CloseSession(&sess);

    TEEC_FinalizeContext(&ctx);

    return TEEC_SUCCESS;
}
