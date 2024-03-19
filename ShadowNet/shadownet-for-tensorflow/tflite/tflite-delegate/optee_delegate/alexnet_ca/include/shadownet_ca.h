#include<string.h>
#include<android/log.h>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "alexnet_ca ", __VA_ARGS__)
//#define LOGD

#define WEIGHTS_PATH_PREFIX "/data/data/android.example.com.tflitecamerademo/cache/alexnet-submodels/"
#define WEIGHTS_PATH_SUFFIX ".weights"
#define MB  (1024*1024)

// skip 20 bytes of weights header: major(4), minor(4), xx(4), seen(8) 
#define WEIGHTS_HEADER 	20

// do not use AddMask layer to allow preload all weights
//#define NO_MASK 


typedef enum {
LT, AM_A, AM_B
}WEIGHTS_TYPE;

#define SHADOWNET_CFG_NUM 9 
typedef struct shadownet_config {
    int prepl_hw, pospl_hw, in_chn, units;
} shadownet_config;

typedef enum {
CONV1 = 0, 
CONV2, 
CONV3, 
CONV4, 
CONV5, 
CONV6, 
CONV7, 
CONV8, 
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

void load_alex_conv1_weights(SHADOWNET_CFG_IDX sn_idx);
void load_alex_conv_norm_weights(SHADOWNET_CFG_IDX sn_idx);
void load_alex_res_weights(SHADOWNET_CFG_IDX sn_idx);

int load_add_mask_a_weights(void *addr, SHADOWNET_CFG_IDX sn_idx); 
int load_add_mask_b_weights(void *addr, SHADOWNET_CFG_IDX sn_idx); 
int load_linear_transform_weights(void *addr, SHADOWNET_CFG_IDX sn_idx);

