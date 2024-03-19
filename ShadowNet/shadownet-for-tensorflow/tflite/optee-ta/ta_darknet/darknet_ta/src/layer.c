#include "layer.h"

#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        return;
    }
    if(l.biases)             free(l.biases);
    if(l.scales)             free(l.scales);
    if(l.weights)            free(l.weights);
    //if(l.output)             free(l.output);
    if(l.rolling_mean)       free(l.rolling_mean);
    if(l.rolling_variance)   free(l.rolling_variance);
    if(l.rscalar)            free(l.rscalar);
    if(l.rbias)              free(l.rbias);
    if(l.obfweights)         free(l.obfweights);

}
