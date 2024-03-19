#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);
list *read_cfg_from_buf(char * cfg_buf);
network *parse_network_cfg_wrapper(char *filename);

#endif
