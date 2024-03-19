#include <err.h>
#include <stdio.h>
#include <string.h>
#include <tee_client_api.h>

#define TA_HELLO_WORLD_UUID \
        { 0x8aaaf200, 0x2450, 0x11e4, \
                { 0xab, 0xe2, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x1b} }

// TODO
#define CMD_TEST                    0

#define MODEL_SAFE_BASE             518 
#define CMD_INIT_SHADOWNET         (MODEL_SAFE_BASE + 0)
#define CMD_LOAD_WEIGHTS            (MODEL_SAFE_BASE + 1) 
#define CMD_NETWORK_PREDICT         (MODEL_SAFE_BASE + 2)
#define CMD_FETCH_RESULTS           (MODEL_SAFE_BASE + 3)

#define TA_TEE_SHADOW_UUID           TA_HELLO_WORLD_UUID 
