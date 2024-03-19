#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
void activate_array(float *x, const int n, const ACTIVATION a);

static inline float relu_activate(float x){return x*(x>0);}
static inline float relu6_activate(float x){return (x < 0.) ? 0 : (6.0 < x) ? 6.0: x;}
/* output = input; no activation */
static inline float noact_activate(float x){return x;}
#endif

