#include<string.h>
#include <time.h>
#include<android/log.h>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "shadownet_ca ", __VA_ARGS__)

#define MOBILENET_MODEL_PATH "/data/data/android.example.com.tflitecamerademo/cache/mobilenet.tflite"
#define MOBILENET_MASKS_PATH "/data/data/android.example.com.tflitecamerademo/cache/masks.weights"
#define RESNET_MODEL_PATH "/data/data/android.example.com.tflitecamerademo/cache/resnet.tflite"
#define RESNET_MASKS_PATH "/data/data/android.example.com.tflitecamerademo/cache/resnet.masks"

//#define DEBUG_TIME

#ifdef DEBUG_TIME
#define CLOCK_START     \
    t = clock()

#define CLOCK_END(whichtime, fname)                            \
    t = clock() - t;                                \
    time_taken = ((double)t)*1000/CLOCKS_PER_SEC;   \
    whichtime += time_taken;                       \
    LOGD("%s takes %.2f ms", fname, time_taken)

#define TOTAL_TIME(whichtime)                                  \
        LOGD(" total_time : %.0f", whichtime);     \
        whichtime = 0.0
#else
#define CLOCK_START     \
    time_taken = 0.0

#define CLOCK_END(whichtime, fname)                            \
    time_taken = 0.0

#define TOTAL_TIME(whichtime)                                  \
    time_taken = 0.0
#endif

typedef enum model_type_t {
    MOBILENET,
    MINIVGG,
    RESNET,
    ALEXNET,
	UNSUPPORTED
}model_type_t;

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

int shadownet_predict(char *pos, unsigned input_size, void *input, unsigned output_size, void *output);

int load_weights_ca_cmd(char *model_path, unsigned CMD); 
int network_predict_ca_cmd(unsigned op_id, model_type_t mtype,  unsigned input_size, void *input, unsigned output_size, void *output);
int network_predict_ca_cmd_multinputs(unsigned op_id, model_type_t model_type, int num_inputs, unsigned input_lens[] , void * inputs[],  unsigned output_size, void *output);