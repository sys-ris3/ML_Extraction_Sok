#include <shadownet_ca.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void *weights_addrs[SHADOWNET_CFG_NUM];
shadownet_config sdw_cfg[SHADOWNET_CFG_NUM];
int shadownet_initialized = 0;

char *shadownet_names[SHADOWNET_CFG_NUM] = {
    "conv1", 
    "conv2", 
    "conv3",
    "conv4",
    "conv5",
    "conv6",
    "conv7",
    "conv8",
    "results"
};

void *map_weights(int idx) {
    int fd;
    struct stat sb;
    size_t length;
    char weights_path[256]={0};
    void *weights_addr;

    // get weights path
    strcat(weights_path, WEIGHTS_PATH_PREFIX);
    strcat(weights_path, shadownet_names[idx]); 
    strcat(weights_path, WEIGHTS_PATH_SUFFIX);

    fd = open(weights_path, O_RDONLY);
    //LOGD("fd:%d, weights_path:%s",fd, weights_path);
    if (fd == -1)
        LOGD("open error! weights_path:%s",weights_path);

    if (fstat(fd, &sb) == -1)           /* To obtain file size */
        LOGD("fstat error!");

	length = sb.st_size;

    weights_addr = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);

    if (weights_addr == MAP_FAILED)
        LOGD("mmap error!");

	// OK to close even if not unmap
	//fclose(fd);
	return weights_addr;
}

void map_all_weights(void){
    int i = 0;
    for (i = 0; i < SHADOWNET_CFG_NUM; i++) {
        weights_addrs[i] = map_weights(i); 
		if(weights_addrs[i] == MAP_FAILED)
            LOGD("MAP_FAILED! %d", i);
        //LOGD("map weights i:%d ,weights_addr:%p",i, weights_addrs[i]);
    }
}

// prepl_hw: pre-pooling-layer input Height * Width 
// pospl_hw: post-pooling-layer input Height * Width
// for many sub-layers, there is no pooling layer
#define FILL_SDW_CFG(idx, PRE, POS, CHN, UNITS)    \
    sdw_cfg[idx].prepl_hw = PRE;                     \
    sdw_cfg[idx].pospl_hw = POS;                     \
    sdw_cfg[idx].in_chn = CHN;                     \
    sdw_cfg[idx].units = UNITS

void init_shadownet_config(void) {
    FILL_SDW_CFG(CONV1, 55*55, 27*27, 115, 96); 
    FILL_SDW_CFG(CONV2, 27*27, 13*13, 307, 256); 
    FILL_SDW_CFG(CONV3, 13*13, 13*13, 460, 384); 
    FILL_SDW_CFG(CONV4, 13*13, 13*13, 460, 384); 
    FILL_SDW_CFG(CONV5, 13*13, 6*6, 307, 256); 
    FILL_SDW_CFG(CONV6, 1, 1, 4915, 4096); 
    FILL_SDW_CFG(CONV7, 1, 1, 4915, 4096); 
    FILL_SDW_CFG(CONV8, 1, 1, 1200, 1000); 
    FILL_SDW_CFG(RESULTS, 1, 1, 1200, 1000); 
}


SHADOWNET_CFG_IDX get_cfg_idx(char *position) {
    if (strcmp(position, "conv_1") == 0) 
        return CONV1;
    else if (strcmp(position, "conv_2") == 0)
        return CONV2;
    else if (strcmp(position, "conv_3") == 0)
        return CONV3;
    else if (strcmp(position, "conv_4") == 0)
        return CONV4;
    else if (strcmp(position, "conv_5") == 0)
        return CONV5;
    else if (strcmp(position, "conv_6") == 0)
        return CONV6;
    else if (strcmp(position, "conv_7") == 0)
        return CONV7;
    else if (strcmp(position, "conv_8") == 0)
        return CONV8;
    else if (strcmp(position, "results") == 0)
        return RESULTS;
    else
        return -1;
}

void setup_shadownet(void) {
	init_shadownet_config(); 
	map_all_weights();
}

static inline int get_sn_input_len (int idx){
    return sdw_cfg[idx].prepl_hw * sdw_cfg[idx].in_chn *sizeof(float);
}
static inline int get_sn_output_len (int idx){
	return sdw_cfg[idx].pospl_hw * sdw_cfg[idx].units *sizeof(float);
}

static inline void copy_result(float *output, float *res, int outputs) {
	if (!(outputs >= 0 && output != NULL && res != NULL))
        LOGD("copy_result error!");
	for (int i = 0; i < outputs; i++)
		output[i] = res[i];
}

void shadownet_predict(char *pos, void *input, void *output) {
	int i, in_len, out_len;
	SHADOWNET_CFG_IDX sn_idx;

	if (shadownet_initialized == 0) {
		setup_shadownet();
		shadownet_initialized = 1; 
        // ignore sn_idx param, init all weights in TA
        init_shadownet_tee_ca_cmd(0);
        // load all layer weights
        for (i = CONV1; i < SHADOWNET_CFG_NUM; ++i)
	        load_weights(i);
	}

    //LOGD("load weights begin!");
	sn_idx = get_cfg_idx(pos);
    if (sn_idx == -1) 
        LOGD("get_cfg_idx failed! pos:%s", pos);

    //LOGD("load weights done!");

    in_len = get_sn_input_len(sn_idx);
    out_len = get_sn_output_len(sn_idx);
	network_predict_ca_cmd(sn_idx, input, in_len, output, out_len);

	return;
}

void load_weights(SHADOWNET_CFG_IDX sn_idx) {
	switch(sn_idx) {
		case CONV1:
			load_alex_conv1_weights(sn_idx);
			break;
		case CONV2:
		case CONV3:
		case CONV4:
		case CONV5:
		case CONV6:
		case CONV7:
		case CONV8:
            load_alex_conv_norm_weights(sn_idx);
			break;
		case RESULTS:
			load_alex_res_weights(sn_idx);
			break;
		default:
			LOGD("Load alexnet weights Error!");
	}	
	return;
} 

#define LOAD_WEIGTHS_HEADER                                 \
        void *addr = weights

#define LOAD_WEIGHTS_CA(type, idx, size)                        \
         load_weights_ca_cmd(type, addr, idx, size, sn_idx);    \
         offset += size;                                        \
         addr = (char *) weights + offset


int load_linear_transform_weights(void *weights, SHADOWNET_CFG_IDX sn_idx) {
    int units = sdw_cfg[sn_idx].units;
    int offset = 0;

    LOAD_WEIGTHS_HEADER;
    // obfw
    LOAD_WEIGHTS_CA(LT, 1, units*2*sizeof(int));
    // rscalar
    LOAD_WEIGHTS_CA(LT, 2, units*sizeof(float));
    // bias
    LOAD_WEIGHTS_CA(LT, 3, units*sizeof(float));

    return offset;
}


int load_add_mask_a_weights(void *weights, SHADOWNET_CFG_IDX sn_idx) {
    int hw;
    int units = sdw_cfg[sn_idx].units;
    int outputs;
    int offset = 0;
    
    if (sn_idx == CONV1) 
        hw = sdw_cfg[sn_idx].pospl_hw;
    else
        hw = sdw_cfg[sn_idx].prepl_hw;
    outputs = hw*units;

    LOAD_WEIGTHS_HEADER;

    // weights 
    LOAD_WEIGHTS_CA(AM_A, 1, outputs*sizeof(float));
    // rscalar 
    LOAD_WEIGHTS_CA(AM_A, 2, sizeof(float));

    return offset;
}

int load_add_mask_b_weights(void *weights, SHADOWNET_CFG_IDX sn_idx) {
    int pospl_hw= sdw_cfg[sn_idx].pospl_hw;
    int units = sdw_cfg[sn_idx].units;
    int outputs = pospl_hw*units;
    int offset = 0;

    LOAD_WEIGTHS_HEADER;

    // weights 
    LOAD_WEIGHTS_CA(AM_B, 1, outputs*sizeof(float));
    // rscalar 
    LOAD_WEIGHTS_CA(AM_B, 2, sizeof(float));

    return offset;
}

#define LOAD_WEIGHTS_DECLARE(sn_idx)        \
	void *base = weights_addrs[sn_idx]; \
	void *addr;                         \
	int offset = WEIGHTS_HEADER;        \
	int len = 0                         \

#define LOAD_WEIGHTS(func)          \
         addr = (char*) base + offset;      \
         len = func(addr, sn_idx); \
         offset += len

// AN_CONV1 : LT, AB, ReLU, MP, AM_A;
void load_alex_conv1_weights(SHADOWNET_CFG_IDX sn_idx){
    LOAD_WEIGHTS_DECLARE(sn_idx); 

	// linear transform
    LOAD_WEIGHTS(load_linear_transform_weights);

    // add mask a 
    LOAD_WEIGHTS(load_add_mask_a_weights);

	return;
} 

// AN_CONV2 - CONV8 : LT, AB, AM_A, ReLU, (MP,) AM_B;
void load_alex_conv_norm_weights(SHADOWNET_CFG_IDX sn_idx){
    LOAD_WEIGHTS_DECLARE(sn_idx); 

	// linear transform
    LOAD_WEIGHTS(load_linear_transform_weights);

    // pop mask a 
    LOAD_WEIGHTS(load_add_mask_a_weights);

    // push mask b
    LOAD_WEIGHTS(load_add_mask_b_weights);

	return;
} 

// AN_RET: LT, AB, AM_A, ReLU, softmax;
void load_alex_res_weights(SHADOWNET_CFG_IDX sn_idx){
    LOAD_WEIGHTS_DECLARE(sn_idx); 

	// linear transform
    LOAD_WEIGHTS(load_linear_transform_weights);

    // pop mask a 
    LOAD_WEIGHTS(load_add_mask_a_weights);

	return;
} 
