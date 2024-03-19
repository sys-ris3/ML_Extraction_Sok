#include <stdio.h>
#include <stdlib.h>
#include "shadownet_ca.h"
#include "ca_cmds.h"


model_type_t model_type;
int shadownet_initialized = 0;
clock_t t;
double time_taken;
double total_time = 0.0; 


int get_resnet_execution_plan_id(char *position) {
    if (strcmp(position,"results") == 0)
        return 44;
    else
        return atoi(position+7);
}

int get_cfg_idx(char *position) {
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


model_type_t get_model_type(char *pos) {
    LOGD("%s pos:%s",__func__, pos);
    if (pos[4] == '1')
        return MOBILENET;
	else if (pos[5] == 'd' && pos[6] == '_')
        return RESNET;
	else
        return UNSUPPORTED;
}


int shadownet_predict(char *pos, unsigned input_size, void *input, unsigned output_size, void *output)
{
    int ret, sn_idx;
    unsigned net_id, op_id;
    int mobilenet_operator_idx[28] = {1, 7, 13, 21, 27, 34, 40, 48, 54, 61, 67, 75, 81, 88, 94,101, 107, 114, 120,127, 133, 140, 146, 154, 160,     167, 173, 182};

	if (shadownet_initialized == 0) {
		model_type = get_model_type(pos);
		if (model_type == MOBILENET) 
        	ret = load_weights_ca_cmd(MOBILENET_MODEL_PATH, TA_SHADOWNET_CMD_LOAD_MODEL); 
		else if (model_type == RESNET)
        	ret = load_weights_ca_cmd(RESNET_MODEL_PATH, TA_SHADOWNET_CMD_LOAD_MODEL); 
		else
            LOGD("Unsupported model type: %u!\n", model_type);
			
        if (ret == -1) {
            LOGD("load_model failed!\n");
            return -1;
        }
		shadownet_initialized = 1; 
	}

	if (model_type == MOBILENET)
		net_id = get_cfg_idx(pos);
	else if (model_type == RESNET) 
		net_id = get_resnet_execution_plan_id(pos); 
	else
    	LOGD("Unsupported model type!");

    if (net_id == -1) 
    	LOGD("get net id failed! pos:%s", pos);

    if (net_id == 0) { // update masks for each new round of inference
        CLOCK_START;
		if (model_type == MOBILENET){
    		ret = load_weights_ca_cmd(MOBILENET_MASKS_PATH, TA_SHADOWNET_CMD_LOAD_MASKS); 
		} else if (model_type == RESNET) {
    		ret = load_weights_ca_cmd(RESNET_MASKS_PATH, TA_SHADOWNET_CMD_LOAD_MASKS); 
		}
        CLOCK_END(total_time, "load_weights");

    	if (ret != 0) 
    		LOGD("load_weights_ca_cmd failed! model type %u\n", model_type);
    }

	if (model_type == MOBILENET)
		op_id = mobilenet_operator_idx[net_id];
	else if (model_type == RESNET) 
		op_id = net_id;

    //CLOCK_START;
    network_predict_ca_cmd(op_id, model_type, input_size, input, output_size, output);
    //CLOCK_END(total_time, "predict");

	return 0;
}


void shadownet_predict_multinputs(int num_inputs, char* pos, unsigned input_sizes[],void * inputs[], unsigned output_size, void *output) {
	int ret, sn_idx;
    unsigned net_id, op_id;
    int mobilenet_operator_idx[28] = {1, 7, 13, 21, 27, 34, 40, 48, 54, 61, 67, 75, 81, 88, 94,101, 107, 114, 120,127, 133, 140, 146, 154, 160,     167, 173, 182};

	if (shadownet_initialized == 0) {
		model_type = get_model_type(pos);
		if (model_type == MOBILENET) 
        	ret = load_weights_ca_cmd(MOBILENET_MODEL_PATH, TA_SHADOWNET_CMD_LOAD_MODEL); 
		else if (model_type == RESNET)
        	ret = load_weights_ca_cmd(RESNET_MODEL_PATH, TA_SHADOWNET_CMD_LOAD_MODEL); 
		else
            LOGD("Unsupported model type: %u!\n", model_type);
			
        if (ret == -1) {
            LOGD("load_model failed!\n");
            return -1;
        }
		shadownet_initialized = 1; 
	}

	if (model_type == MOBILENET)
		net_id = get_cfg_idx(pos);
	else if (model_type == RESNET) 
		net_id = get_resnet_execution_plan_id(pos); 
	else
    	LOGD("Unsupported model type!");

    if (net_id == -1) 
    	LOGD("get net id failed! pos:%s", pos);

    if (net_id == 0) { // update masks for each new round of inference
        CLOCK_START;
		if (model_type == MOBILENET){
    		ret = load_weights_ca_cmd(MOBILENET_MASKS_PATH, TA_SHADOWNET_CMD_LOAD_MASKS); 
		} else if (model_type == RESNET) {
    		ret = load_weights_ca_cmd(RESNET_MASKS_PATH, TA_SHADOWNET_CMD_LOAD_MASKS); 
		}
        CLOCK_END(total_time, "load_weights");

    	if (ret != 0) 
    		LOGD("load_weights_ca_cmd failed! model type %u\n", model_type);
    }

	if (model_type == MOBILENET)
		op_id = mobilenet_operator_idx[net_id];
	else if (model_type == RESNET) 
		op_id = net_id;

    //CLOCK_START;
    network_predict_ca_cmd_multinputs(op_id, model_type, num_inputs, input_lens, inputs, output_size, output);
    //CLOCK_END(total_time, "predict");

	return 0;
}
