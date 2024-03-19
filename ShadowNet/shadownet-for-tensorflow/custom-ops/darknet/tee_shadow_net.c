#include <string.h>
#include"darknet.h"
#include"tee_shadow_net.h"

void darknet_predict(const char *position, const void *input, void *output) {
    network *net;
    char cfg_path[256]={0}; 
    char weights_path[256]={0}; 

    if (strlen(position) > 20) {
        fprintf(stderr, "ERR! darknet_predict() position not known! %s\n", position);
        return;
    }

    strcat(cfg_path, "/root/darknet/cfg/mobilenet-subnets/");
    strcat(cfg_path, position);
    strcat(cfg_path, ".cfg");

    strcat(weights_path, "/root/darknet/data/mobilenet-submodels/");
    strcat(weights_path, position);
    strcat(weights_path, ".weights");

    // TODO optimize, allow network_predict to accept output buf to reduce copy
    net = load_network(cfg_path, weights_path, 0);
    set_batch_network(net, 1);
    float *predictions = network_predict(net, (float *)input);
    copy_cpu(net->outputs, predictions, 1, output, 1);

    return;
}
