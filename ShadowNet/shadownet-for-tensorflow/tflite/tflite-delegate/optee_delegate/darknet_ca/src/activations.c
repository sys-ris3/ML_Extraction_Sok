#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case RELU:
            return "relu";
        case RELU6:
            return "relu6";
        case NOACT:
            return "noact";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "relu6")==0) return RELU6;
    if (strcmp(s, "noact")==0) return NOACT;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
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
