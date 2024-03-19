#include<string.h>
#include<stdio.h>
//#include<android/log.h>

//#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "shadownet_ca ", __VA_ARGS__)
#define LOGD printf

#define WEIGHTS_PATH_PREFIX "/data/data/android.example.com.tflitecamerademo/cache/mobilenet-submodels/"
#define WEIGHTS_PATH_SUFFIX ".weights"
#define MB  (1024*1024)

//#define NO_MASK // do not use AddMask layer to allow preload all weights

typedef enum {
SN_CONV, SN_DW, SN_PW, SN_PRED, SN_RET
}SHADOWNET_TYPE;

typedef enum {
LT, AM_A, AM_B, SF, BN, BIG_INPUT 
}WEIGHTS_TYPE;

#define SHADOWNET_CFG_NUM 28
typedef struct shadownet_config {
    SHADOWNET_TYPE type;
    int h, w, c, units;
} shadownet_config;

typedef enum {
CONV1 = 0, 
DWCONV1, 
PWCONV1,
DWCONV2, 
PWCONV2,
DWCONV3, 
PWCONV3,
DWCONV4, 
PWCONV4,
DWCONV5, 
PWCONV5,
DWCONV6, 
PWCONV6,
DWCONV7, 
PWCONV7,
DWCONV8, 
PWCONV8,
DWCONV9, 
PWCONV9,
DWCONV10, 
PWCONV10,
DWCONV11, 
PWCONV11,
DWCONV12, 
PWCONV12,
DWCONV13, 
PWCONV13,
RESULTS
}SHADOWNET_CFG_IDX;

SHADOWNET_CFG_IDX get_cfg_idx(char *position); 

void setup_shadownet(void);
void *map_weights(int idx);
void init_shadownet_config(void); 
void shadownet_predict(char *pos, void *input, void *output);
void load_weights(SHADOWNET_CFG_IDX sn_idx); 

int init_shadownet_tee_ca_cmd(SHADOWNET_CFG_IDX sn_idx); 
void network_predict_ca_cmd(SHADOWNET_CFG_IDX sn_idx, void *input, int in_len, void *output, int out_len);
void load_weights_ca_cmd_unit(WEIGHTS_TYPE type, void *weights, int weight_idx, size_t length, size_t offset, SHADOWNET_CFG_IDX sn_idx); 
void load_weights_ca_cmd(WEIGHTS_TYPE type, void *weights, int weight_idx, size_t length, SHADOWNET_CFG_IDX sn_idx);
int fetch_results_ca_cmd(void *buf, int out_len); 

void load_sn_res_weights(SHADOWNET_CFG_IDX sn_idx);
void load_sn_pred_weights(SHADOWNET_CFG_IDX sn_idx);
void load_sn_pw_weights(SHADOWNET_CFG_IDX sn_idx);
void load_sn_dw_weights(SHADOWNET_CFG_IDX sn_idx);
void load_sn_conv_weights(SHADOWNET_CFG_IDX sn_idx);

int load_add_mask_a_weights(void *addr, SHADOWNET_CFG_IDX sn_idx); 
int load_add_mask_b_weights(void *addr, SHADOWNET_CFG_IDX sn_idx); 
int load_batchnorm_weights(void *addr, SHADOWNET_CFG_IDX sn_idx); 
int load_shuffle_channel_weights(void *addr, SHADOWNET_CFG_IDX sn_idx); 
int load_linear_transform_weights(void *addr, SHADOWNET_CFG_IDX sn_idx);

// skip 20 bytes of weights header: major(4), minor(4), xx(4), seen(8) 
#define WEIGHTS_HEADER 	20
