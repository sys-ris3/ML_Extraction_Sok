#include <string.h>
#include <time.h>
//#include <android/log.h>
#include<stdio.h>
#include<stdlib.h>
#include <shadownet_ca.h>

//#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "com.tflitedemo ZC native", __VA_ARGS__)
//#define LOGD(...) printf(__VA_ARGS)
#define LOGD printf

#define MAX_INPUT_LEN   (112*112*76*4) 
#define MAX_OUTPUT_LEN  (112*112*64*4) 



int main(){
    int i, j;
	clock_t start, end;
	double cpu_time_used;
    char pos_array[28][12]={
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
        "results",
    };
    float *input, *output;
    float times[28][10] = {0};
    float total, avg, tee_total;
    input = (float *)malloc(MAX_INPUT_LEN);
    output = (float *)malloc(MAX_OUTPUT_LEN); 
   	
    for (j = 0; j < 10; j++){
        total = 0.0;
        for (i = 0; i < 28; i++){
            if (i == 2)
                continue;
    	    start = clock();
            shadownet_predict(pos_array[i], input, output);
    	    end = clock();
    	    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
    	    //LOGD("pos:%s, time:%f ms", pos_array[i], cpu_time_used);
    	    printf("pos:%s, time:%f ms\n", pos_array[i], cpu_time_used);
            times[i][j] = cpu_time_used;
            total += cpu_time_used;
        }
        printf("total tee time: %f\n\n", total);
    }
    tee_total = 0.0;
    for (i = 0; i < 28; i++) {
        total = 0.0;
        for (j = 2; j < 10; j++) total += times[i][j]; 
        avg = total/8.0;
        printf("pos:%s, avg time:%f ms\n", pos_array[i], avg);
        tee_total += avg;
    }

    printf("\n");
    printf("tee total avg time:%f ms\n", tee_total);

    return 0;
}
