#!/usr/bin/env python

import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def print_layer_idx_and_name(model):
    index = None
    for idx, layer in enumerate(model.layers):
        print ("%3d %s"%(idx, layer.name))

MobileNet = module_from_file("MobileNet", "./mobilenet_def.py").MobileNet

if __name__ == '__main__':
    model = MobileNet()
    model.summary()
    model.save('mobilenet.h5')
    print_layer_idx_and_name(model)
    