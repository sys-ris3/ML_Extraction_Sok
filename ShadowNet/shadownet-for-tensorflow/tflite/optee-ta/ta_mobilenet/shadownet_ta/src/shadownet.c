#include <shadownet.h>

shadownet_config sdw_cfg[SHADOWNET_CFG_NUM];
void *mem = NULL;
mempool mpool = {0};

void *big_output = NULL;
int flag_bigoutput = 0;

#ifdef NO_MASK // without mask layer
int nets_weight_initialized = 0;
int nets_initialized = 0;
shadownet sdw_nets[SHADOWNET_CFG_NUM];
#else
int mempool_initialized = 0;
shadownet sdw_net = {0};
#endif

#define SELECT_OUTPUT_BUF(func, param1, param2)     \
    if (flag_bigoutput == 0)                            \
        func(net, param1, input);                       \
    else {                                              \
        func(net, param1, param2);                       \
        big_output = param2;                            \
    }


#define FILL_SDW_CFG(idx, T, H, W, C, U)    \
    sdw_cfg[idx].type = T;                  \
    sdw_cfg[idx].h = H;                     \
    sdw_cfg[idx].w = W;                     \
    sdw_cfg[idx].c = C;                     \
    sdw_cfg[idx].units = U

int get_input_len(SHADOWNET_CFG_IDX sn_idx) {
    return sdw_cfg[sn_idx].h * sdw_cfg[sn_idx].w * sdw_cfg[sn_idx].c* sizeof(float);
}

int get_output_len(SHADOWNET_CFG_IDX sn_idx) {
  if (sn_idx != PWCONV13)
    return sdw_cfg[sn_idx].h * sdw_cfg[sn_idx].w * sdw_cfg[sn_idx].units * sizeof(float);
  else // pool layer
    return sdw_cfg[sn_idx].units * sizeof(float);
}

void init_shadownet_config(void) {
    FILL_SDW_CFG(CONV1, SN_CONV, 112, 112, 38, 32); 
    FILL_SDW_CFG(DWCONV1, SN_DW, 112, 112, 32, 32); 
    FILL_SDW_CFG(PWCONV1, SN_PW, 112, 112, 38, 64); 
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

#define ALLOC_WEIGHTS_FBUF(buf, len)                           \
                    buf = (float *)((char *)mem + offset);     \
                    offset += len; 

//                    DMSG("pool fbuf x :%p", buf)

#define ALLOC_WEIGHTS_IBUF(buf, len)                           \
                    buf = (int *)((char *)mem + offset);        \
                    offset += len; 

//                    DMSG("pool ibuf x :%p", buf)


#ifdef NO_MASK
int init_all_weights() {
    int i;
    int offset = 0;
    mem = (void *)calloc(ALL_WEIGHTS_SIZE + MAX_INPUT_LEN * 2, sizeof(char)); 
    if (mem == NULL) {
        EMSG("calloc mem failed!");
        return -1;
    }

    // OUTPUT 
    ALLOC_WEIGHTS_FBUF(mpool.OUTPUT_A, MAX_INPUT_LEN);
    ALLOC_WEIGHTS_FBUF(mpool.OUTPUT_B, MAX_INPUT_LEN);

    // alloc weights for CONV1
    // Linear Transform weights
    ALLOC_WEIGHTS_IBUF(sdw_nets[CONV1].lt_obfweights, MAX_CHN_NUM*2*4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].lt_rbias, MAX_CHN_NUM*4);

    // Batchnorm weights
    ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].bn_biases, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].bn_scales, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].bn_rolling_mean, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].bn_rolling_variance,MAX_CHN_NUM * 4);

    // Shuffle Channel weights
    ALLOC_WEIGHTS_IBUF(sdw_nets[CONV1].sf_obfweights, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].sf_rbias, MAX_CHN_NUM * 4);

    // alloc weights for dw/pw subnets
    for (i = DWCONV1; i < PWCONV13; ++i) {
        if ((i - DWCONV1)%2 == 0) { // DW w/o mask:SF, BN, ReLU
            // Shuffle Channel weights
            ALLOC_WEIGHTS_IBUF(sdw_nets[i].sf_obfweights, MAX_CHN_NUM * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].sf_rbias, MAX_CHN_NUM * 4);

            // Batchnorm weights
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_biases, MAX_CHN_NUM * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_scales, MAX_CHN_NUM * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_rolling_mean, MAX_CHN_NUM * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_rolling_variance,MAX_CHN_NUM * 4);
            
        } else { // PW w/o mask: LT, BN, ReLU, SF
            // Linear Transform weights
            ALLOC_WEIGHTS_IBUF(sdw_nets[i].lt_obfweights, MAX_CHN_NUM*2*4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].lt_rbias, MAX_CHN_NUM*4);

            // Batchnorm weights
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_biases, MAX_CHN_NUM * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_scales, MAX_CHN_NUM * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_rolling_mean, MAX_CHN_NUM * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_rolling_variance,MAX_CHN_NUM * 4);

            // Shuffle Channel weights
            ALLOC_WEIGHTS_IBUF(sdw_nets[i].sf_obfweights, MAX_CHN_NUM * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].sf_rbias, MAX_CHN_NUM * 4);
        }
    } 

    // alloc weights for PRED/PWCONV13: LT, BN, ReLU, PL, RS, DO
    // Linear Transform Weights
    ALLOC_WEIGHTS_FBUF(sdw_nets[PWCONV13].lt_obfweights, MAX_CHN_NUM*2*4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[PWCONV13].lt_rbias, MAX_CHN_NUM*4);

    // Batchnorm weights
    ALLOC_WEIGHTS_FBUF(sdw_nets[PWCONV13].bn_biases, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[PWCONV13].bn_scales, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[PWCONV13].bn_rolling_mean, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[PWCONV13].bn_rolling_variance,MAX_CHN_NUM * 4);

    // alloc weights for RET:LT, AM, RS, SM
    // Linear Transform Weights
    ALLOC_WEIGHTS_FBUF(sdw_nets[RESULTS].lt_obfweights, MAX_CHN_NUM*2*4);
    ALLOC_WEIGHTS_FBUF(sdw_nets[RESULTS].lt_rbias, MAX_CHN_NUM*4);

    // Add Mask/Bias weights
    ALLOC_WEIGHTS_FBUF(sdw_nets[RESULTS].am_weights_a, MAX_CHN_NUM*4); 

    
    // make sure allocation is correct
    if (offset > ALL_WEIGHTS_SIZE + 2*MAX_INPUT_LEN){ 
        EMSG("offset:%d > ALL_WEIGHTS_SIZE:%d",offset,ALL_WEIGHTS_SIZE);
        return -1;
    }
    return 0;
}

int init_shadownets(void) {
    int ret = 0;
    int sn_idx;
    SHADOWNET_TYPE type;

    // init mempool
    if (nets_weight_initialized == 0) {
        ret = init_all_weights(); 
        if (ret == 0) 
            nets_weight_initialized = 1;
        else
            return -1;
        
        init_shadownet_config(); 
    }

    for (sn_idx = 0; sn_idx <= RESULTS; ++sn_idx) {
        type = get_sn_type_from_cfg_idx(sn_idx); 
        sdw_nets[sn_idx].h = sdw_cfg[sn_idx].h;
        sdw_nets[sn_idx].w = sdw_cfg[sn_idx].w;
        sdw_nets[sn_idx].c = sdw_cfg[sn_idx].c;
        sdw_nets[sn_idx].units = sdw_cfg[sn_idx].units;
        sdw_nets[sn_idx].inputs = sdw_nets[sn_idx].h * sdw_nets[sn_idx].w * sdw_nets[sn_idx].units;
        sdw_nets[sn_idx].temperature = 1;
        sdw_nets[sn_idx].type = type;
    }

    return 0;
}

 // SN_CONV-nomask : LT, BN, ReLU, SF;
int forward_sn_conv(shadownet *net, float *input) {
#ifdef FUSE_LAYERS
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_bn_ac_layers(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_shuffle_channel_layer(net, mpool.OUTPUT_B, input);
#else
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_batchnorm_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_relu6_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_shuffle_channel_layer(net, mpool.OUTPUT_A, input);
#endif
    return 0;
}

// SN_DW-nomask: SF, BN, ReLU;
int forward_sn_dw(shadownet *net, float *input) {
#ifdef FUSE_LAYERS
    forward_shuffle_channel_layer(net, input, mpool.OUTPUT_A);
    //forward_bn_ac_layers(net, mpool.OUTPUT_A, input);
    SELECT_OUTPUT_BUF(forward_bn_ac_layers, mpool.OUTPUT_A, mpool.OUTPUT_B);
#else
    forward_shuffle_channel_layer(net, input, mpool.OUTPUT_A);
    forward_batchnorm_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    //forward_relu6_layer(net, mpool.OUTPUT_B, input);
    SELECT_OUTPUT_BUF(forward_relu6_layer, mpool.OUTPUT_B, mpool.OUTPUT_A);
#endif
    return 0;
}

 // SN_PW-nomask: LT, BN, ReLU, SF;
int forward_sn_pw(shadownet *net, float *input) {
#ifdef FUSE_LAYERS
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_bn_ac_layers(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    //forward_shuffle_channel_layer(net, mpool.OUTPUT_B, input);
    SELECT_OUTPUT_BUF(forward_shuffle_channel_layer, mpool.OUTPUT_B, mpool.OUTPUT_A);
#else
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_batchnorm_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_relu6_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    //forward_shuffle_channel_layer(net, mpool.OUTPUT_A, input);
    SELECT_OUTPUT_BUF(forward_shuffle_channel_layer, mpool.OUTPUT_A, mpool.OUTPUT_B);
#endif
    return 0;
}


// SN_PRED-nomask: LT, BN, ReLU, PL, RS, DO;
int forward_sn_pred(shadownet *net, float *input) {
#ifdef FUSE_LAYERS
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_bn_ac_layers(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_avgpool_layer(net, mpool.OUTPUT_B, input);
#else
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_batchnorm_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_relu6_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    // avgpool update net->inputs
    forward_avgpool_layer(net, mpool.OUTPUT_A, input);
#endif
    return 0;
}

 // SN_RET-nomask: LT, AM, RS, SM;
int forward_sn_ret(shadownet *net, float *input){
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_add_mask_layer_a(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_softmax_layer(net, mpool.OUTPUT_B, input);
    return 0;
}

#else // With Mask layer

int init_mempool(mempool* pool) {
    int offset = 0; // in bytes
    mem = (void *)calloc(MEMPOOL_SIZE, sizeof(char)); 
    if (mem == NULL) {
        return -1;
    }

    // Add Mask weights
    ALLOC_WEIGHTS_FBUF(pool->AM_WEIGHTS_A, MAX_AM_WEIGHTS_LEN); 
    ALLOC_WEIGHTS_FBUF(pool->AM_WEIGHTS_B, MAX_AM_WEIGHTS_LEN); 

#ifdef TEST_DYN_SHM
    // OUTPUT 
    ALLOC_WEIGHTS_FBUF(pool->OUTPUT_A, MAX_INPUT_LEN);
    pool->OUTPUT_B = pool->OUTPUT_A; 
#else
    // OUTPUT 
    ALLOC_WEIGHTS_FBUF(pool->OUTPUT_A, MAX_AM_WEIGHTS_LEN);
    ALLOC_WEIGHTS_FBUF(pool->OUTPUT_B, MAX_INPUT_LEN);
#endif
    
    // Linear Transform weights
    ALLOC_WEIGHTS_IBUF(pool->LT_OBF_WEIGHTS, MAX_CHN_NUM * 2 * 4);
    ALLOC_WEIGHTS_FBUF(pool->LT_BIAS_WEIGHTS, MAX_CHN_NUM * 4);
 
    // Shuffle Channel weights
    ALLOC_WEIGHTS_IBUF(pool->SF_OBF_WEIGHTS, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(pool->SF_BIAS_WEIGHTS, MAX_CHN_NUM * 4);
    
    // Batchnorm weights
    ALLOC_WEIGHTS_FBUF(pool->BN_SCALE_WEIGHTS, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(pool->BN_BIAS_WEIGHTS, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(pool->BN_MEAN_WEIGHTS, MAX_CHN_NUM * 4);
    ALLOC_WEIGHTS_FBUF(pool->BN_VAR_WEIGHTS, MAX_CHN_NUM * 4);

    // make sure allocation is correct
    if (offset != MEMPOOL_SIZE) 
        EMSG("offset:%d != MEMPOOL_SIZE:%d",offset,MEMPOOL_SIZE);
    
    return 0;
}

void init_sn_weights(shadownet *net) {
    // LT
    net->lt_obfweights = mpool.LT_OBF_WEIGHTS; 
    net->lt_rbias = mpool.LT_BIAS_WEIGHTS; 

    // BN
    net->bn_biases = mpool.BN_BIAS_WEIGHTS;
    net->bn_scales= mpool.BN_SCALE_WEIGHTS;
    net->bn_rolling_mean = mpool.BN_MEAN_WEIGHTS;
    net->bn_rolling_variance = mpool.BN_VAR_WEIGHTS;

    // AM 
    net->am_weights_a = mpool.AM_WEIGHTS_A;

    // SF 
    net->sf_obfweights = mpool.SF_OBF_WEIGHTS;
    net->sf_rbias = mpool.SF_BIAS_WEIGHTS;

    // AM 
    net->am_weights_b = mpool.AM_WEIGHTS_B;
}

int init_shadownet(shadownet *net) {
    int ret = 0;
    SHADOWNET_TYPE type;

    // init mempool
    if (mempool_initialized == 0) {
        ret = init_mempool(&mpool); 
        if (ret == 0) 
            mempool_initialized = 1;
        else
            return -1;
        
        init_shadownet_config(); 
        init_sn_weights(net);
    }

    type = get_sn_type_from_cfg_idx(net->sn_idx); 
    net->h = sdw_cfg[net->sn_idx].h;
    net->w = sdw_cfg[net->sn_idx].w;
    net->c = sdw_cfg[net->sn_idx].c;
    net->units = sdw_cfg[net->sn_idx].units;
    net->inputs = net->h * net->w * net->units;
    net->temperature = 1;
    net->type = type;

    return 0;
}

 // SN_CONV : LT, BN, ReLU, AM, SF;
int forward_sn_conv(shadownet *net, float *input) {
#ifdef FUSE_LAYERS
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_bn_ac_ama_layers(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_shuffle_channel_layer(net, mpool.OUTPUT_B, input);
#else
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_batchnorm_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_relu6_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_add_mask_layer_a(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_shuffle_channel_layer(net, mpool.OUTPUT_B, input);
#endif
    return 0;
}

// SN_DW: SF, AM, BN, ReLU, AM;
int forward_sn_dw(shadownet *net, float *input) {
#ifdef FUSE_LAYERS
    forward_shuffle_channel_layer(net, input, mpool.OUTPUT_A);
    //forward_ama_bn_ac_amb_layers(net, mpool.OUTPUT_A, input);
    SELECT_OUTPUT_BUF(forward_ama_bn_ac_amb_layers, mpool.OUTPUT_A, mpool.OUTPUT_B);
#else
    forward_shuffle_channel_layer(net, input, mpool.OUTPUT_A);
    forward_add_mask_layer_a(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_batchnorm_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_relu6_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    //forward_add_mask_layer_b(net, mpool.OUTPUT_B, input); 
    SELECT_OUTPUT_BUF(forward_add_mask_layer_b, mpool.OUTPUT_B, mpool.OUTPUT_A);
#endif
    return 0;
}

 // SN_PW: LT, AM, BN, ReLU, AM, SF;
int forward_sn_pw(shadownet *net, float *input) {
#ifdef FUSE_LAYERS
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_ama_bn_ac_amb_layers(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    //forward_shuffle_channel_layer(net, mpool.OUTPUT_B, input);
    SELECT_OUTPUT_BUF(forward_shuffle_channel_layer, mpool.OUTPUT_B, mpool.OUTPUT_A);
#else
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_add_mask_layer_a(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_batchnorm_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_relu6_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_add_mask_layer_b(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    //forward_shuffle_channel_layer(net, mpool.OUTPUT_A, input);
    SELECT_OUTPUT_BUF(forward_shuffle_channel_layer, mpool.OUTPUT_A, mpool.OUTPUT_B);
#endif
    return 0;
}


// SN_PRED: LT, AM, BN, ReLU, PL, RS, DO, AM;
int forward_sn_pred(shadownet *net, float *input) {
#ifdef FUSE_LAYERS
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_ama_bn_ac_layers(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_avgpool_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_add_mask_layer_b(net, mpool.OUTPUT_A, input);
#else
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_add_mask_layer_a(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_batchnorm_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_relu6_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    // avgpool update net->inputs
    forward_avgpool_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_add_mask_layer_b(net, mpool.OUTPUT_A, input);
#endif
    return 0;
}

 // SN_RET: LT, AM, AM, RS, SM;
int forward_sn_ret(shadownet *net, float *input){
#ifdef FUSE_LAYERS
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_ama_amb_layers(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_softmax_layer(net, mpool.OUTPUT_B, input);
#else
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_add_mask_layer_a(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_add_mask_layer_b(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_softmax_layer(net, mpool.OUTPUT_A, input);
#endif
    return 0;
}

#endif // NO_MASK

SHADOWNET_TYPE get_sn_type_from_cfg_idx(SHADOWNET_CFG_IDX sn_idx) {
    if (sn_idx == CONV1)
        return SN_CONV;
    else if (sn_idx == PWCONV13)
        return SN_PRED;
    else if (sn_idx == RESULTS)
        return SN_RET;
    else if (sn_idx % 2 == 0) // PWCONVx
        return SN_PW;
    else
        return SN_DW;
}


int network_predict(shadownet *net, float *inout) {
    switch(net->type) {
        case SN_CONV:
            forward_sn_conv(net, inout);
            break;
        case SN_DW:
            forward_sn_dw(net, inout);  
            break;
        case SN_PW:
            forward_sn_pw(net, inout);  
            break;
        case SN_PRED:
            forward_sn_pred(net, inout);  
            break;
        case SN_RET:
            forward_sn_ret(net, inout);  
            break;
        default:
            EMSG("%s type not recognized!",__func__);
            return -1;
    }
    return 0;
}
