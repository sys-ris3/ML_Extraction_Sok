import itertools
# part of the code here borrowed from slalom https://github.com/ftramer/slalom/blob/master/python/slalom/quant_layers.py

def get_all_layers(model):
    all_layers = [[l] for l in model.layers]
    all_layers = list(itertools.chain.from_iterable(all_layers))
    return all_layers

def get_first_linear_layer_idx(layers):
    for i in range(len(layers)):
        if isinstance(layers[i], Conv2D) or isinstance(layers[i],Dense):
            return i
    return -1

def get_last_linear_layer_idx(layers):
    num = len(layers)
    for i in range(num-1, -1, -1):
        if isinstance(layers[i], Conv2D) or isinstance(layers[i],Dense):
            return i
    return -1
# transform a model into a (quantized) shadownet model
def transform(model, obf_ratio, bits_x, bits_w, quantize=True):
    layers = model.layers
    layer_map = {} 

    def transform_layer(layer, is_first_linear, is_last_linear):
        print("transform {} (next = {})".format(layer, next_layer))
        new_layers = []
        skip_next = False

        if isinstance(layer, InputLayer):
            new_layers.append(InputLayer.from_config(layer.get_config()))

        elif isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            conf = layer.get_config()

            if not is_first_linear: # push mask if not first linear layer 
                new_layers.append(AddMask(random_scalar=1.0))

            filters = conf['filters']
            obf_filters = int(filters*obf_ratio)
            conf['filters'] = obf_filters

            act = conf['activation']
            conf['activation'] = "linear"

            # TODO: handle Conv2D with bias
            new_layer = Conv2D.from_config(conf)
            new_layers.append(new_layer)
            layer_map[new_layer] = layer

            new_layer = LinearTransformGeneric(filters)
            new_layers.append(new_layer)
            layer_map[new_layer] = layer

            if not is_first_linear: # pop mask if not first linear layer 
                new_layers.append(AddMask(random_scalar=1.0))

            if act != 'linear':
                new_layers.append(Activation(act))

        elif isinstance(layer, Dense):
            conf = layer.get_config()

            if not is_first_linear: # push mask if not first linear layer 
                new_layers.append(AddMask(random_scalar=1.0))

            del conf['units']
            filters = layer.units
            obf_filters = int(filters*obf_ratio)
            conf['filters'] = obf_filters 
            conf['kernel_size'] = 1

            act = conf['activation']
            conf['activation'] = 'linear' 

            h_in = int(layer.input_spec.axes[-1])
            new_layers.append(Reshape((1, 1, h_in)))
            flattened = True
            new_layer = Conv2D.from_config(conf)
            new_layers.append(new_layer)
            layer_map[new_layer] = layer

            new_layer = LinearTransformGeneric(filters)
            new_layers.append(new_layer)
            layer_map[new_layer] = layer

            if not is_first_linear: # pop mask if not first linear layer 
                new_layers.append(AddMask(random_scalar=1.0))

            if act != 'linear':
                new_layers.append(Activation(act))

            new_layers.append(Reshape((filters,)))

        elif isinstance(layer, BatchNormalization):
            new_layers.append(BatchNormalization.from_config(layer.get_config()))

        elif isinstance(layer, MaxPooling2D):
            new_layers.append(MaxPooling2D.from_config(layer.get_config()))

        elif isinstance(layer, AveragePooling2D):
            new_layers.append(AveragePooling2D.from_config(layer.get_config()))

        elif isinstance(layer, Activation):
            new_layers.append(Activation.from_config(layer.get_config()))

        elif isinstance(layer, ZeroPadding2D):
            new_layers.append(ZeroPadding2D.from_config(layer.get_config()))

        elif isinstance(layer, Flatten):
            new_layers.append(Flatten.from_config(layer.get_config()))

        elif isinstance(layer, GlobalAveragePooling2D):
            conf = layer.get_config()
            new_layers.append(GlobalAveragePooling2D.from_config(conf))

        elif isinstance(layer, Reshape):
            new_layers.append(Reshape.from_config(layer.get_config()))

        elif isinstance(layer, Dropout):
            pass

        return new_layers

    new_model = Sequential()

    first_linear_idx = get_first_linear_layer_idx(layers)
    last_linear_idx = get_last_linear_layer_idx(layers)
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        new_layers = transform_layer(layer, layer_idx == first_linear_idx, layer_idx == last_linear_idx)
        for new_layer in new_layers:
            new_model.add(new_layer)

    print(new_model.summary())

"""
def buffer_todo():
    # copy over (and potentially quantize) all the weights
    new_layers = get_all_layers(new_model)

    for layer in new_layers:
        if layer in layer_map:
            src_layer = layer_map[layer]

            weights = src_layer.get_weights()
            kernel = weights[0]
            bias = weights[1]

            if quantize:
                range_w = 2**bits_w
                range_x = 2**bits_x
                kernel_q = np.round(range_w * kernel)
                bias_q = np.round(range_w * range_x * bias)
                if slalom_privacy:

                    if isinstance(layer, DepthwiseConv2DQ):
                        bias_q = bias_q.astype(np.float64)
                        kernel_q = kernel_q.astype(np.float64)

                layer._trainable_weights = layer._trainable_weights[2:]

                if isinstance(src_layer, Dense) and verif_preproc:
                    kernel_q = np.reshape(kernel_q, (1, 1, kernel_q.shape[0], kernel_q.shape[1]))

                layer.set_weights((kernel_q, bias_q))
            else:
                layer._trainable_weights = layer._trainable_weights[2:]
                layer.set_weights((kernel, bias))

    # find all the TensorFlow ops that correspond to inputs/outputs of linear operators
    new_ops = [op for op in K.get_session().graph.get_operations() if op not in old_ops]
    linear_ops_in = [tf.reshape(op.inputs[0], [-1]) for op in new_ops if op.type in ['Conv2D', 'MatMul', 'DepthwiseConv2dNative']]
    linear_ops_out = [tf.reshape(op.outputs[0], [-1]) for op in new_ops if op.type in ['BiasAdd']]

    return new_model, linear_ops_in, linear_ops_out
"""
