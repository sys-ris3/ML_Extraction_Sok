#ifndef TEE_SHADOW_NET_H
#define TEE_SHADOW_NET_H

#ifdef __cplusplus
extern "C" {
#endif

void darknet_predict(const char *position,unsigned input_size, const void *input, unsigned output_size, void *output); 
void darknet_predict_multinputs(int input_num, const char *position, unsigned input_sizes[],  void *inputs[], unsigned output_size, void *output);

void shadownet_predict(const char *position,unsigned input_size, const void *input, unsigned output_size, void *output); 
void shadownet_predict_multinputs(int num_inputs, char* pos, unsigned input_sizes[],void * inputs[], unsigned output_size, void *output);




#ifdef __cplusplus
}
#endif
#endif
