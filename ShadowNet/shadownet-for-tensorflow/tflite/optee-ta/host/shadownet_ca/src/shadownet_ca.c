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
    "dwconv1", 
    "pwconv1",
    "dwconv2", 
    "pwconv2",
    "dwconv3", 
    "pwconv3",
    "dwconv4", 
    "pwconv4",
    "dwconv5", 
    "pwconv5",
    "dwconv6", 
    "pwconv6",
    "dwconv7", 
    "pwconv7",
    "dwconv8", 
    "pwconv8",
    "dwconv9", 
    "pwconv9",
    "dwconv10", 
    "pwconv10",
    "dwconv11", 
    "pwconv11",
    "dwconv12", 
    "pwconv12",
    "dwconv13", 
    "pwconv13",
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
#define FILL_SDW_CFG(idx, T, H, W, C, U)    \
    sdw_cfg[idx].type = T;                  \
    sdw_cfg[idx].h = H;                     \
    sdw_cfg[idx].w = W;                     \
    sdw_cfg[idx].c = C;                     \
    sdw_cfg[idx].units = U

void init_shadownet_config(void) {
    FILL_SDW_CFG(CONV1, SN_CONV, 112, 112, 38, 32); 
    FILL_SDW_CFG(DWCONV1, SN_DW, 112, 112, 32, 32); 
    FILL_SDW_CFG(PWCONV1, SN_PW, 112, 112, 76, 64); 
    FILL_SDW_CFG(DWCONV2, SN_DW, 56, 56, 64, 64); 
    FILL_SDW_CFG(PWCONV2, SN_PW, 56, 56, 153, 128); 
    FILL_SDW_CFG(DWCONV3, SN_DW, 56, 56, 128, 128); 
    FILL_SDW_CFG(PWCONV3, SN_PW, 56, 56, 153, 128); 
    FILL_SDW_CFG(DWCONV4, SN_DW, 28, 28, 128, 128); 
    FILL_SDW_CFG(PWCONV4, SN_PW, 28, 28, 307, 256); 
    FILL_SDW_CFG(DWCONV5, SN_DW, 28, 28, 256, 256); 
    FILL_SDW_CFG(PWCONV5, SN_PW, 28, 28, 307, 256); 
    FILL_SDW_CFG(DWCONV6, SN_DW, 14, 14, 256, 256); 
    FILL_SDW_CFG(PWCONV6, SN_PW, 14, 14, 614, 512); 
    FILL_SDW_CFG(DWCONV7, SN_DW, 14, 14, 512, 512); 
    FILL_SDW_CFG(PWCONV7, SN_PW, 14, 14, 614, 512); 
    FILL_SDW_CFG(DWCONV8, SN_DW, 14, 14, 512, 512); 
    FILL_SDW_CFG(PWCONV8, SN_PW, 14, 14, 614, 512); 
    FILL_SDW_CFG(DWCONV9, SN_DW, 14, 14, 512, 512); 
    FILL_SDW_CFG(PWCONV9, SN_PW, 14, 14, 614, 512); 
    FILL_SDW_CFG(DWCONV10, SN_DW, 14, 14, 512, 512); 
    FILL_SDW_CFG(PWCONV10, SN_PW, 14, 14, 614, 512); 
    FILL_SDW_CFG(DWCONV11, SN_DW, 14, 14, 512, 512); 
    FILL_SDW_CFG(PWCONV11, SN_PW, 14, 14, 614, 512); 
    FILL_SDW_CFG(DWCONV12, SN_DW, 7, 7, 512, 512); 
    FILL_SDW_CFG(PWCONV12, SN_PW, 7, 7, 1228, 1024); 
    FILL_SDW_CFG(DWCONV13, SN_DW, 7, 7, 1024, 1024); 
    FILL_SDW_CFG(PWCONV13, SN_PRED, 7, 7, 1228, 1024); 
    FILL_SDW_CFG(RESULTS, SN_RET, 1, 1, 1200, 1000); 
}

SHADOWNET_CFG_IDX get_cfg_idx(char *position) {
    if (strcmp(position, "conv1") == 0) 
        return CONV1;
    else if (strcmp(position, "dwconv1") == 0)
        return DWCONV1;
    else if (strcmp(position, "pwconv1") == 0)
        return PWCONV1;
    else if (strcmp(position, "dwconv2") == 0)
        return DWCONV2;
    else if (strcmp(position, "pwconv2") == 0)
        return PWCONV2;
    else if (strcmp(position, "dwconv3") == 0)
        return DWCONV3;
    else if (strcmp(position, "pwconv3") == 0)
        return PWCONV3;
    else if (strcmp(position, "dwconv4") == 0)
        return DWCONV4;
    else if (strcmp(position, "pwconv4") == 0)
        return PWCONV4;
    else if (strcmp(position, "dwconv5") == 0)
        return DWCONV5;
    else if (strcmp(position, "pwconv5") == 0)
        return PWCONV5;
    else if (strcmp(position, "dwconv6") == 0)
        return DWCONV6;
    else if (strcmp(position, "pwconv6") == 0)
        return PWCONV6;
    else if (strcmp(position, "dwconv7") == 0)
        return DWCONV7;
    else if (strcmp(position, "pwconv7") == 0)
        return PWCONV7;
    else if (strcmp(position, "dwconv8") == 0)
        return DWCONV8;
    else if (strcmp(position, "pwconv8") == 0)
        return PWCONV8;
    else if (strcmp(position, "dwconv9") == 0)
        return DWCONV9;
    else if (strcmp(position, "pwconv9") == 0)
        return PWCONV9;
    else if (strcmp(position, "dwconv10") == 0)
        return DWCONV10;
    else if (strcmp(position, "pwconv10") == 0)
        return PWCONV10;
    else if (strcmp(position, "dwconv11") == 0)
        return DWCONV11;
    else if (strcmp(position, "pwconv11") == 0)
        return PWCONV11;
    else if (strcmp(position, "dwconv12") == 0)
        return DWCONV12;
    else if (strcmp(position, "pwconv12") == 0)
        return PWCONV12;
    else if (strcmp(position, "dwconv13") == 0)
        return DWCONV13;
    else if (strcmp(position, "pwconv13") == 0)
        return PWCONV13;
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
    return sdw_cfg[idx].h * sdw_cfg[idx].w * sdw_cfg[idx].c *sizeof(float);
}
static inline int get_sn_output_len (int idx){
    if (idx != PWCONV13)
	    return sdw_cfg[idx].h * sdw_cfg[idx].w * sdw_cfg[idx].units *sizeof(float);
    else
	    return sdw_cfg[idx].units * sizeof(float);
}
static inline int get_sn_outputs (int idx){
	return sdw_cfg[idx].h * sdw_cfg[idx].w * sdw_cfg[idx].units;
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
#ifdef NO_MASK
        // ignore sn_idx param, init all weights in TA
        init_shadownet_tee_ca_cmd(0);
        // load all layer weights
        for (i = CONV1; i < SHADOWNET_CFG_NUM; ++i)
	        load_weights(i);
#endif
	}

    //LOGD("load weights begin!");
	sn_idx = get_cfg_idx(pos);
    if (sn_idx == -1) 
        LOGD("get_cfg_idx failed! pos:%s", pos);

#ifndef NO_MASK
    init_shadownet_tee_ca_cmd(sn_idx);
	load_weights(sn_idx);
#endif
	
    //LOGD("load weights done!");

    in_len = get_sn_input_len(sn_idx);
    out_len = get_sn_output_len(sn_idx);
	network_predict_ca_cmd(sn_idx, input, in_len, output, out_len);

	return;
}

void load_weights(SHADOWNET_CFG_IDX sn_idx) {
	switch(sn_idx) {
		case CONV1:
			load_sn_conv_weights(sn_idx);
			break;
		case DWCONV1:
		case DWCONV2:
		case DWCONV3:
		case DWCONV4:
		case DWCONV5:
		case DWCONV6:
		case DWCONV7:
		case DWCONV8:
		case DWCONV9:
		case DWCONV10:
		case DWCONV11:
		case DWCONV12:
		case DWCONV13:
			load_sn_dw_weights(sn_idx);
			break;
		case PWCONV1:
		case PWCONV2:
		case PWCONV3:
		case PWCONV4:
		case PWCONV5:
		case PWCONV6:
		case PWCONV7:
		case PWCONV8:
		case PWCONV9:
		case PWCONV10:
		case PWCONV11:
		case PWCONV12:
			load_sn_pw_weights(sn_idx);
			break;
		case PWCONV13:
			load_sn_pred_weights(sn_idx);
			break;
		case RESULTS:
			load_sn_res_weights(sn_idx);
			break;
		default:
			LOGD("Load weights Error!");
	}	
	return;
} 

#define LOAD_WEIGHTS_CA(type, idx, size)                        \
         load_weights_ca_cmd(type, addr, idx, size, sn_idx);    \
         offset += size;                                        \
         addr = (char *) addr + offset


int load_linear_transform_weights(void *addr, SHADOWNET_CFG_IDX sn_idx) {
    int units = sdw_cfg[sn_idx].units;
    int offset = 0;
    // obfw
    LOAD_WEIGHTS_CA(LT, 1, units*2*sizeof(int));
    // rbias
    LOAD_WEIGHTS_CA(LT, 2, units*sizeof(float));

    return offset;
}

int load_shuffle_channel_weights(void *addr, SHADOWNET_CFG_IDX sn_idx) {
    int units = sdw_cfg[sn_idx].units;
    int offset = 0;
    // obfw
    LOAD_WEIGHTS_CA(SF, 1, units*sizeof(int));
    // rbias
    LOAD_WEIGHTS_CA(SF, 2, units*sizeof(float));

    return offset;
}

int load_batchnorm_weights(void *addr, SHADOWNET_CFG_IDX sn_idx) {
    int units = sdw_cfg[sn_idx].units;
    int offset = 0;
    // scales 
    LOAD_WEIGHTS_CA(BN, 1, units*sizeof(float));
    // bias 
    LOAD_WEIGHTS_CA(BN, 2, units*sizeof(float));
    // mean 
    LOAD_WEIGHTS_CA(BN, 3, units*sizeof(float));
    // var 
    LOAD_WEIGHTS_CA(BN, 4, units*sizeof(float));

    return offset;
}

int load_add_mask_a_weights(void *addr, SHADOWNET_CFG_IDX sn_idx) {
    int h= sdw_cfg[sn_idx].h;
    int w= sdw_cfg[sn_idx].w;
    int units = sdw_cfg[sn_idx].units;
    int outputs = h*w*units;
    int offset = 0;

#ifdef NO_MASK
    offset = outputs * sizeof(float) + sizeof(float);
#else
    // weights 
    LOAD_WEIGHTS_CA(AM_A, 1, outputs*sizeof(float));
    // rscalar 
    LOAD_WEIGHTS_CA(AM_A, 2, sizeof(float));
#endif


    return offset;
}

int load_add_mask_b_weights(void *addr, SHADOWNET_CFG_IDX sn_idx) {
    int h= sdw_cfg[sn_idx].h;
    int w= sdw_cfg[sn_idx].w;
    int units = sdw_cfg[sn_idx].units;
    int outputs = h*w*units;
    int offset = 0;

    // weights 
    if (sn_idx != PWCONV13) { // PWCONV13 is the PRED layer 
#ifdef NO_MASK
        offset += outputs*sizeof(float);
#else
        // weights 
        LOAD_WEIGHTS_CA(AM_B, 1, outputs*sizeof(float));
#endif
    } else {
#ifdef NO_MASK
        offset += units*sizeof(float);
#else
        // PRED layer, pooling
        LOAD_WEIGHTS_CA(AM_B, 1, units*sizeof(float));
#endif
    }

#ifdef NO_MASK
        offset += sizeof(float);
#else
    // rscalar 
    LOAD_WEIGHTS_CA(AM_A, 2, sizeof(float));
#endif

    return offset;
}

#define LOAD_WEIGHTS(func)          \
         addr = (char*) base + offset;      \
         len = func(addr, sn_idx); \
         offset += len

// SN_CONV : LT, BN, ReLU, AM, SF;
void load_sn_conv_weights(SHADOWNET_CFG_IDX sn_idx){
	void *base = weights_addrs[sn_idx];
	void *addr;
	int offset = WEIGHTS_HEADER;
	int len = 0;

	// linear transform
    LOAD_WEIGHTS(load_linear_transform_weights);

    // batchnorm
    LOAD_WEIGHTS(load_batchnorm_weights);

    // add mask a 
    LOAD_WEIGHTS(load_add_mask_a_weights);

    // shuffle channel 
    LOAD_WEIGHTS(load_shuffle_channel_weights);

	return;
} 

// SN_DW: SF, AM, BN, ReLU, AM;
void load_sn_dw_weights(SHADOWNET_CFG_IDX sn_idx){
	void *base = weights_addrs[sn_idx];
	void *addr;
	int offset = WEIGHTS_HEADER;
	int len = 0;

    // shuffle channel 
    LOAD_WEIGHTS(load_shuffle_channel_weights);

    // add mask a 
    LOAD_WEIGHTS(load_add_mask_a_weights);

    // batchnorm
    LOAD_WEIGHTS(load_batchnorm_weights);

    // add mask b 
    LOAD_WEIGHTS(load_add_mask_b_weights);

	return;
} 

// SN_PW: LT, AM, BN, ReLU, AM, SF;
void load_sn_pw_weights(SHADOWNET_CFG_IDX sn_idx){
	void *base = weights_addrs[sn_idx];
	void *addr;
	int offset = WEIGHTS_HEADER;
	int len = 0;

	// linear transform
    LOAD_WEIGHTS(load_linear_transform_weights);

    // add mask a 
    LOAD_WEIGHTS(load_add_mask_a_weights);

    // batchnorm
    LOAD_WEIGHTS(load_batchnorm_weights);

    // add mask b 
    LOAD_WEIGHTS(load_add_mask_b_weights);

    // shuffle channel 
    LOAD_WEIGHTS(load_shuffle_channel_weights);

	return;
} 

// SN_PRED: LT, AM, BN, ReLU, PL, RS, DO, AM;
void load_sn_pred_weights(SHADOWNET_CFG_IDX sn_idx){
	void *base = weights_addrs[sn_idx];
	void *addr;
	int offset = WEIGHTS_HEADER;
	int len = 0;

	// linear transform
    LOAD_WEIGHTS(load_linear_transform_weights);

    // add mask a 
    LOAD_WEIGHTS(load_add_mask_a_weights);

    // batchnorm
    LOAD_WEIGHTS(load_batchnorm_weights);

    // add mask b 
    LOAD_WEIGHTS(load_add_mask_b_weights);

	return;
} 

// SN_RET: LT, AM, AM, RS, SM;
void load_sn_res_weights(SHADOWNET_CFG_IDX sn_idx){
	void *base = weights_addrs[sn_idx];
	void *addr;
	int offset = WEIGHTS_HEADER;
	int len = 0;

	// linear transform
    LOAD_WEIGHTS(load_linear_transform_weights);

    // add mask a 
    LOAD_WEIGHTS(load_add_mask_a_weights);

    // add mask b 
    LOAD_WEIGHTS(load_add_mask_b_weights);

	return;
} 
