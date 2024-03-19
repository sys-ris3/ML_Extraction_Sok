#ifndef TEE_SHADOW_NET_H
#define TEE_SHADOW_NET_H

#ifdef __cplusplus
extern "C" {
#endif

void darknet_predict(const char *position, const void *input, void *output); 

#ifdef __cplusplus
}
#endif
#endif
