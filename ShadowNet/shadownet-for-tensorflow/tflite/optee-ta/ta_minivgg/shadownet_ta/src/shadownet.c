#include <shadownet.h>

shadownet_config sdw_cfg[SHADOWNET_CFG_NUM];
void *mem = NULL;
mempool mpool = {0};

int nets_initialized;
shadownet sdw_nets[SHADOWNET_CFG_NUM] = {0};

#define FILL_SDW_CFG(idx, MAXPOOL, H, OH, C, U)    \
    sdw_cfg[idx].has_maxpool = MAXPOOL;                     \
    sdw_cfg[idx].h = H;                     \
    sdw_cfg[idx].w = H;                     \
    sdw_cfg[idx].oh = OH;                     \
    sdw_cfg[idx].ow = OH;                     \
    sdw_cfg[idx].c = C;                     \
    sdw_cfg[idx].units = U

void init_shadownet_config(void) {
    FILL_SDW_CFG(CONV1, 1, 64, 32, 38, 32); 
    FILL_SDW_CFG(CONV2, 1, 32, 16, 76, 64); 
    FILL_SDW_CFG(CONV3, 0, 16, 16, 153, 128); 
    FILL_SDW_CFG(CONV4, 0, 16, 16, 153, 128); 
    FILL_SDW_CFG(CONV5, 1, 16, 8, 153, 128); 
    FILL_SDW_CFG(CONV6, 0, 1, 1, 614, 512); 
    FILL_SDW_CFG(RESULTS, 0, 1, 1, 1200, 1000); 
}

#define ALLOC_WEIGHTS_FBUF(buf, len)                           \
                    buf = (float *)((char *)mem + offset);     \
                    offset += len; 

//                    DMSG("pool fbuf x :%p", buf)

#define ALLOC_WEIGHTS_IBUF(buf, len)                           \
                    buf = (int *)((char *)mem + offset);        \
                    offset += len; 

//                    DMSG("pool ibuf x :%p", buf)


int init_shadownets() {
    int i;
    int offset = 0;
    int bufsize;

    // initialize config
    init_shadownet_config();

    // initialize mempool
    mem = (void *)calloc(MEMPOOL_SIZE, sizeof(char)); 
    if (mem == NULL) {
        EMSG("calloc mem failed! %d", MEMPOOL_SIZE);
        return -1;
    }

    // OUTPUT 
    ALLOC_WEIGHTS_FBUF(mpool.OUTPUT_A, MAX_INPUT_LEN);
    ALLOC_WEIGHTS_FBUF(mpool.OUTPUT_B, MAX_INPUT_LEN);

    // initialize weights
    for (i = 0; i <= RESULTS; i++) {
        // Linear Transform Generic weights
        ALLOC_WEIGHTS_IBUF(sdw_nets[i].lt_obfweights, sdw_cfg[i].units * 2 * 4);
        ALLOC_WEIGHTS_FBUF(sdw_nets[i].lt_rscale, sdw_cfg[i].units * 4);
        ALLOC_WEIGHTS_FBUF(sdw_nets[i].lt_rbias, sdw_cfg[i].units * 4);

    
        if ( i == CONV1 || i == RESULTS) { // only one mask layer
            if (i == CONV1) {
                // Batchnorm weights
                ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].bn_biases, sdw_cfg[i].units * 4);
                ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].bn_scales, sdw_cfg[i].units * 4);
                ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].bn_rolling_mean, sdw_cfg[i].units * 4);
                ALLOC_WEIGHTS_FBUF(sdw_nets[CONV1].bn_rolling_variance, sdw_cfg[i].units * 4);
            }

            bufsize = sdw_cfg[i].oh * sdw_cfg[i].ow * sdw_cfg[i].units * 4;
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].am_weights_a, bufsize); 
        } else {
            bufsize = sdw_cfg[i].h * sdw_cfg[i].w * sdw_cfg[i].units * 4;
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].am_weights_a, bufsize); 

            // Batchnorm weights
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_biases, sdw_cfg[i].units * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_scales, sdw_cfg[i].units * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_rolling_mean, sdw_cfg[i].units * 4);
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].bn_rolling_variance, sdw_cfg[i].units * 4);

            bufsize = sdw_cfg[i].oh * sdw_cfg[i].ow * sdw_cfg[i].units * 4;
            ALLOC_WEIGHTS_FBUF(sdw_nets[i].am_weights_b, bufsize); 
        }
        
    }

    // make sure allocation is correct
    if (offset > MEMPOOL_SIZE){ 
        EMSG("offset:%d > MEMPOOL_SIZE :%d",offset,MEMPOOL_SIZE);
        return -1;
    }
    return 0;
}

int forward_minivgg_conv_wo_maxpool(shadownet *net, float *input) {
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_add_mask_layer_a(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_relu_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_batchnorm_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_add_mask_layer_b(net, mpool.OUTPUT_B, input);
}

int forward_minivgg_conv_w_maxpool(shadownet *net, float *input) {
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_add_mask_layer_a(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_relu_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_batchnorm_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_maxpool_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_add_mask_layer_b(net, mpool.OUTPUT_A, input);
}

int forward_minivgg_conv1(shadownet *net, float *input) {
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_relu_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_batchnorm_layer(net, mpool.OUTPUT_B, mpool.OUTPUT_A);
    forward_maxpool_layer(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_add_mask_layer_a(net, mpool.OUTPUT_B, input);
}

int forward_minivgg_results(shadownet *net, float *input) {
    forward_linear_transform_layer(net, input, mpool.OUTPUT_A);
    forward_add_mask_layer_a(net, mpool.OUTPUT_A, mpool.OUTPUT_B);
    forward_softmax_layer(net, mpool.OUTPUT_B, input);
}


int network_predict(shadownet *net, float *inout) {
    if (net->sn_idx == CONV1)
        forward_minivgg_conv1(net, inout);
    else if (net->sn_idx == RESULTS)
        forward_minivgg_results(net, inout);
    else { // CONV2 - CONV8
        if (sdw_cfg[net->sn_idx].has_maxpool == 1)
            forward_minivgg_conv_w_maxpool(net, inout);
        else
            forward_minivgg_conv_wo_maxpool(net, inout);
    }

    return 0;
}
