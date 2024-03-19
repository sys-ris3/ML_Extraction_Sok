#ifndef CA_CMDS_H
#define CA_CMDS_H
#define TA_HELLO_WORLD_UUID \
        { 0x8aaaf200, 0x2450, 0x11e4, \
                { 0xab, 0xe2, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x1b} }

#define TA_SHADOWNET_CMD_LOAD_MODEL             2
#define TA_SHADOWNET_CMD_INFERENCE              3
#define TA_SHADOWNET_CMD_LOAD_MASKS             4
#define TA_SHADOWNET_CMD_MULTINPUTS_INFERENCE  5

#define TA_TEE_SHADOW_UUID           TA_HELLO_WORLD_UUID 

#endif
