#ifndef SHADOWNET_H
#define SHADOWNET_H

#define MAX_MODEL_SIZE (62*1024*1024)
#define MAX_MASKS_SIZE (38543264)
void shadownet_inference(unsigned op_id, float* input, float *output);
void resnet_inference(unsigned plan_id, float* input, float *output);
void init_output_buf(void); 
void init_resnet_plan(void); 
int initialize_tflite_model_from_buffer(void *buffer);
int update_masks(uint8_t *masks, unsigned mask_size);
int fake_update_masks(uint8_t *masks, unsigned mask_size);
int neon_muladd(float *x, float *s, float *y, float *z, int size); 
int neon_muladd_fixed_scalar(float *x, float s, float *y, float *z, int size);
int neon_relu6(float *input, float *output, int size); 

#endif
