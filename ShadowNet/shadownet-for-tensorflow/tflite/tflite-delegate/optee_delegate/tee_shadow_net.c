#include <string.h>
#include <time.h>
//#include <android/log.h>
//#include "shadownet_ca/include/shadownet_ca.h"
//#include "shadownet_ca/shadownet_ca.h"
//#include "minivgg_ca/include/shadownet_ca.h"
#include"tee_shadow_net.h"

//#define LOGD
//#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "com.tflitedemo ZC native", __VA_ARGS__)

void darknet_predict(const char *position, unsigned input_size,  const void *input, unsigned output_size, void *output) {
#if 0
    network *net;
    char cfg_path[256]={0}; 
    char weights_path[256]={0}; 

    if (strlen(position) > 20) {
        LOGD("Err! position not known!");
        return;
    }

    LOGD("ZC position :%s !\n",position);
    strcat(cfg_path, "/data/data/android.example.com.tflitecamerademo/cache/mobilenet-subnets/");
    strcat(cfg_path, position);
    strcat(cfg_path, ".cfg");

    strcat(weights_path, "/data/data/android.example.com.tflitecamerademo/cache/mobilenet-submodels/");
    strcat(weights_path, position);
    strcat(weights_path, ".weights");


    // TODO optimize, allow network_predict to accept output buf to reduce copy
    net = load_network_ca(cfg_path, weights_path, 0);
    LOGD("load network return net:%p",net);
    if (net == NULL)
        return; 
    float *predictions = network_predict_ca(net, (float *)input);
    copy_cpu(net->outputs, predictions, 1, output, 1);
#endif
	clock_t start, end;
	double cpu_time_used;
	
	start = clock();
    shadownet_predict(position, input_size, input, output_size, output);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
//	LOGD("pos:%s, time:%f ms", position, cpu_time_used);

    return;
}


void darknet_predict_multinputs(int input_num, const char *position, unsigned input_sizes[],  void *inputs[], unsigned output_size, void *output) {
	clock_t start, end;
	double cpu_time_used;

	start = clock();
    shadownet_predict_multinputs(input_num, position,input_sizes, inputs, output_size, output);
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;

    return;
}
