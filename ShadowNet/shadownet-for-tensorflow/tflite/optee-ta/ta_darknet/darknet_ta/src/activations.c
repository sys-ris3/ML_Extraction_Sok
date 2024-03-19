#include "activations.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tee_internal_api.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case RELU:
            return (char*)"relu";
        case RELU6:
            return (char*)"relu6";
        case NOACT:
            return (char*)"noact";
        default:
            break;
    }
    return (char*)"relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "relu6")==0) return RELU6;
    if (strcmp(s, "noact")==0) return NOACT;
    //fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    DMSG("Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case RELU:
            return relu_activate(x);
        case RELU6:
            return relu6_activate(x);
        case NOACT:
            return noact_activate(x);
        default:
            return 0;
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}
