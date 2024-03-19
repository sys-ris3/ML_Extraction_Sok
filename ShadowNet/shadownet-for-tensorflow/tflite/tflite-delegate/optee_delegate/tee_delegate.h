/*
 * ModelSafe Project. 
 * May, 2020
 * Zhichuang Sun(sun.zhi@husky.neu.edu)
 */
#include <stdio.h>
#include <string.h>
#include "include/tee_client_api.h"

#define TA_HELLO_WORLD_UUID \
        { 0x8aaaf200, 0x2450, 0x11e4, \
                { 0xab, 0xe2, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x1b} }
/* The function IDs implemented in this TA */
#define TA_HELLO_WORLD_CMD_INC_VALUE            0
#define TA_HELLO_WORLD_CMD_DEC_VALUE            1


typedef enum {
    teeDelegateOk = 0,
    teeDelegateNoinit,
    teeDelegateFail
} teeDelegateStatus;

struct tee_linear_transform_blob{
    size_t B;
    size_t H;
    size_t W;
    size_t M;
    size_t N;
    const float* x_flat;
    const float* r_flat;
    const int* w_flat;
    const float* o_flat;
};

TEEC_Result tee_delegate_init(void);
TEEC_Result tee_delegate_exit(void);
teeDelegateStatus tee_linear_transform(uint32_t *pres, size_t B, size_t H, size_t W, size_t M, size_t N,
       const float* x_flat, const int* w_flat, const float* r_flat, float* output_flat);
