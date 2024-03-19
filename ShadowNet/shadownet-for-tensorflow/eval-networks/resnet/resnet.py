
## https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_06_3_resnet.ipynb

import tensorflow as tf
import tensorflow.keras
from tensorflow.python.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.engine import training
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, Input,AveragePooling2D,Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.activations import relu, softmax

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

import numpy as np
import random as rand
import json, os, sys

from IPython import embed


cur_dir = os.path.dirname(__file__)
resnet_package_path = os.path.join(cur_dir,"..","..","quantization")
sys.path.append(resnet_package_path)
from add_mask_layer import AddMask
from linear_transform_generic_layer import LinearTransformGeneric 

'''
from tee_shadow_generic_layer import TeeShadowGeneric
from add_mask_layer import AddMask
from linear_transform_generic_layer import LinearTransformGeneric 
from shuffle_channel_layer import ShuffleChannel
'''

# quantized layer for any non-linear computation
class ActivationQ(Layer):

    def __init__(self, activation, bits_w, bits_x, quantize=True,
                 **kwargs):
        super(ActivationQ, self).__init__(**kwargs)
        self.bits_w = bits_w 
        self.bits_x = bits_x
        self.range_w = 2**bits_w
        self.range_x = 2**bits_x
        self.quantize = quantize
        self.activation = activation
        assert activation in ["relu", "relu6", "softmax"]

    def activation_name(self):
        return self.activation

    def call(self, inputs):
        #inputs = tf.Print(inputs, [tf.reduce_sum(tf.abs(tf.cast(inputs, tf.float64)))], message="relu input: ")
        #inputs = tf.Print(inputs, [], message="in ActivationQ with input shape: {}".format(inputs.get_shape().as_list()))

        if self.quantize:
            inputs_dq = inputs/(self.range_x * self.range_w)
            if self.activation in ["relu", "relu6"]:
                if self.activation.endswith("relu6"):
                    act = relu(inputs, max_value=6 * self.range_x * self.range_w)
                else:
                    act = relu(inputs)

                outputs = tf.math.round(act / self.range_w)
            else: # softmax
                outputs = tf.nn.softmax(inputs_dq)
        else:
            if self.activation == "relu":
                outputs = tf.nn.relu(inputs)
            elif self.activation == "relu6":
                outputs = tf.nn.relu6(inputs)
            else:
                outputs = tf.nn.softmax(inputs)
        return outputs 

    def get_config(self):
        config = {
            'bits_w': self.bits_w,
            'bits_x': self.bits_x,
            'quantize': self.quantize,
            'activation': self.activation,
        }
        base_config = super(ActivationQ, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

(train_imgs, train_labels), (test_imgs, test_labels) = cifar10.load_data()
conv_map = {} 
depth_map = {} 
dense_map = {} 
mask_box = []
conv_id = 0

def transform_resnet_block_layer(layer, layer_idx, is_first_linear, is_last_linear, obf_ratio = 1.2, quantize=True, keep_dense=True, bits_w = 8, bits_x = 8):
    print("transform {})".format(layer))
    new_layers = []
    orig_block = []
    conv_trans_block = []
    mask_trans_block = []



    if isinstance(layer, Conv2D):
        conf = layer.get_config()
        useBias = False
        isDense = False

        # push mask if not first linear layer 
        if not is_first_linear:
            push_mask = AddMask(random_scalar=1.0)
            new_layers.append(push_mask)

        act = conf['activation']
        conf['activation'] = "linear"
        global conv_id
        conf['name'] = "conv2d_{}".format(conv_id)
        conv_id += 1
        filters = conf['filters']
        obf_filters = int(np.ceil(filters*obf_ratio))
        conf['filters'] = obf_filters

        conf['use_bias'] = False
        # bias is added at LinearTransformGeneric layer
        if layer.use_bias:
            useBias = True

        new_conv = Conv2D.from_config(conf)
        new_layers.append(new_conv)

        lt = LinearTransformGeneric(filters)
        new_layers.append(lt)

        # weights transformation
        conv_map[layer] = (new_conv, lt, isDense, useBias) 

        if not is_first_linear: # pop mask (salar=-1) if not first linear layer 
            pop_mask = AddMask(random_scalar= -1.0)
            new_layers.append(pop_mask)
            mask_box.append((new_layers, len(new_layers))) 

        if act != 'linear':
            act_layer = ActivationQ(act, bits_w, bits_x, quantize, name='activation_%d'%layer_idx)
            new_layers.append(act_layer)

        # prepare conv_trans and mask_trans for block comparison
        orig_block.append(layer) 

        conv_trans_block.append(new_conv)
        conv_trans_block.append(lt)
        if act != 'linear':
            conv_trans_block.append(act_layer) 

        mask_trans_block.extend(new_layers)

    elif isinstance(layer, Dense):
        conf = layer.get_config()
        useBias = False
        isDense = True 

        if not is_first_linear: # push mask if not first linear layer 
            push_mask = AddMask(random_scalar=1.0)
            new_layers.append(push_mask)

        conf['use_bias'] = False
        if layer.use_bias and not keep_dense:
            useBias = True 

        filters = layer.units
        obf_filters = int(np.ceil(filters*obf_ratio))
        if keep_dense:
            conf['filters'] = filters 
        else:
            conf['filters'] = obf_filters 
        del conf['units']

        conf['kernel_size'] = 1

        act = conf['activation']
        conf['activation'] = 'linear' 

        h_in = int(layer.input_spec.axes[-1])
        inp_reshape = Reshape((1, 1, h_in))
        new_layers.append(inp_reshape)
        new_conv = Conv2D.from_config(conf)
        new_layers.append(new_conv)

        if keep_dense:
            dense_map[layer] = new_conv 
        else:
            lt = LinearTransformGeneric(filters)
            new_layers.append(lt)

            # transform conv weights
            conv_map[layer] = (new_conv, lt, isDense, useBias)

        if not is_first_linear: # pop mask if not first linear layer 
            pop_mask = AddMask(random_scalar=-1.0)
            new_layers.append(pop_mask)
            # convert_mask_weights(new_layers)
            mask_box.append((new_layers,len(new_layers)))

        if act != 'linear':
            act_layer = ActivationQ(act, bits_w, bits_x, quantize, name='activation_%d'%layer_idx)
            new_layers.append(act_layer)
        else:
            act_layer = None

        reshape = Reshape((filters,))
        new_layers.append(reshape)

        orig_block.append(layer)
        if act_layer != None:
            if keep_dense:
                conv_trans_block.extend([inp_reshape, new_conv, act_layer, reshape])
            else:
                conv_trans_block.extend([inp_reshape, new_conv, lt, act_layer, reshape])
        else:
            if keep_dense:
                conv_trans_block.extend([inp_reshape, new_conv, reshape])
            else:
                conv_trans_block.extend([inp_reshape, new_conv, lt, reshape])
        mask_trans_block.extend(new_layers)

    elif isinstance(layer, BatchNormalization):
        pass

    elif isinstance(layer, AveragePooling2D):
        new_layers.append(AveragePooling2D.from_config(layer.get_config()))
        #new_layers.append(Lambda(tf.round))


    elif isinstance(layer, Activation):
        print(layer.activation)
        assert layer.activation in [relu, softmax]
        act_func = "relu" if layer.activation == relu else "softmax"
        new_layers.append(ActivationQ(act_func, bits_w, bits_x, quantize, name = "activation_%d"%layer_idx))



    elif isinstance(layer, Flatten):
        new_layers.append(Flatten.from_config(layer.get_config()))


    elif tf.python.keras.layers.merge.add.__code__.co_code == layer.__code__.co_code:
        new_layers.append(tf.keras.layers.add)
    else:
        print("unsupported layer", layer)
        embed()
        exit(1)

    return new_layers, orig_block, conv_trans_block, mask_trans_block

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    x = conv(x)
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)

    return x
def resnet_layer_functors(num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))
    functors = []
    functors.append(conv)
    if batch_normalization:
        functors.append(BatchNormalization())
    if activation is not None:
        functors.append(Activation(activation))

    return functors

def get_random_scalar(scalar_stack, isPush):
    if isPush:
        random_scalar = rand.random()
        scalar_stack.append(random_scalar)
    else:
        random_scalar = scalar_stack.pop() * (-1.)
    return random_scalar

def resnet_layer_obf(inputs,
                 block_id,
                 scalar_stack = [],
                 num_filters=16,
                 obf_ratio = 1.2,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 use_mask=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    x = inputs
    if (use_mask is True):
        random_scalar = get_random_scalar(scalar_stack, True)
        x = AddMask(random_scalar, name = 'push_mask_%d'%(block_id))(x)

    obf_filters = int(num_filters*obf_ratio)

    conv = Conv2D(obf_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  use_bias=False,
                  kernel_regularizer=l2(1e-4))

    x = conv(x)
    x = LinearTransformGeneric(num_filters,name="linear_transform_%d"%block_id)(x)
    
    if (use_mask is True):
        random_scalar = get_random_scalar(scalar_stack, False)
        x = AddMask(random_scalar, name = 'pop_mask_%d'%(block_id))(x)
    
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def get_next_conv_shape(conv_name,obf_model):
    name_split = conv_name.split('_')
    assert(name_split[0] == 'conv2d')
    exceptions = {16:18,31:33,44:-1}
    
    next_conv_idx = None
    if(len(name_split) == 1):
        next_conv_idx = 1
    else:
        this_conv_idx = int(name_split[1])
        if this_conv_idx in exceptions:
            next_conv_idx = exceptions[this_conv_idx]
        else:
            next_conv_idx = this_conv_idx + 1
    
    if next_conv_idx == -1:
        ## reach the end
        return [None,1,1,10]
    else:
        next_conv_name = 'conv2d_{}'.format(next_conv_idx)
        candidates = [v for k,v in obf_model.items() if k == next_conv_name]
        assert(len(candidates) == 1)
        return candidates[0]

    


def resnet_layer_split(inputs,
                 block_id,
                 json_obj,
                 json_dict,
                 obf_model,
                 scalar_stack = [],
                 num_filters=16,
                 obf_ratio = 1.2,
                 kernel_size=3,
                 strides=1,
                 activation=None,
                 batch_normalization=False,
                 use_mask=False):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    x = inputs
    ## if (use_mask is True):
    ##     random_scalar = get_random_scalar(scalar_stack, True)
    ##     x = AddMask(random_scalar, name = 'push_mask_%d'%(block_id))(x)

    obf_filters = int(num_filters*obf_ratio)

    conv = Conv2D(obf_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  use_bias=False,
                  kernel_regularizer=l2(1e-4))

    x = conv(x)
    conv_name = x.name.split('/')[0]
    
    if conv_name not in json_obj:
        json_obj[conv_name] = json_dict
        

    # x = LinearTransformGeneric(num_filters,name="linear_transform_%d"%block_id)(x)
    
    ## if (use_mask is True):
    ##    random_scalar = get_random_scalar(scalar_stack, False)
    ##    x = AddMask(random_scalar, name = 'pop_mask_%d'%(block_id))(x)
    
    next_conv_shape = get_next_conv_shape(conv_name,obf_model)
    if next_conv_shape == [None,1,1,10]:
        x = TeeShadowGeneric(next_conv_shape[1],next_conv_shape[2],next_conv_shape[3], position="results",  name="ts_conv_{}".format(block_id))(x)
    else:
        x = TeeShadowGeneric(next_conv_shape[1],next_conv_shape[2],next_conv_shape[3], position="conv2d_{}".format(block_id),  name="ts_conv_{}".format(block_id))(x)
    return x


class ResNetBlock(Layer):
    def __init__(self, stack, res_block, num_filters=16, strides = 1, transformed = False, quant = False, layer_idx = 0, is_first_linear= False, is_last_linear = False, **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)

        self.stack = stack
        self.res_block = res_block
        self.num_filters = num_filters
        self.strides = strides

        self.transformed = transformed
        self.quant = quant
        self.layer_idx = layer_idx
        self.is_first_linear = is_first_linear
        self.is_last_linear = is_last_linear
        
        layer1_functors = resnet_layer_functors(num_filters=self.num_filters,
                                                    strides=self.strides)
        layer2_functors = resnet_layer_functors(num_filters=self.num_filters,
                                                    activation=None)
        if self.stack > 0 and self.res_block == 0:
            layer3_functors = resnet_layer_functors(num_filters=self.num_filters,
                                                    kernel_size=1,
                                                    strides=self.strides,
                                                    activation=None,
                                                    batch_normalization=False)
        
        layer4_functors = [tf.keras.layers.add, Activation('relu')]
        if(transformed == False and quant == True):
            print("quant but not transform?")
            assert(False)
        if not transformed:
            self.layer1_functors = layer1_functors
            self.layer2_functors = layer2_functors
            if self.stack > 0 and self.res_block == 0:
                self.layer3_functors = layer3_functors
            self.layer4_functors = layer4_functors
        else:
            ## layer1
            self.layer1_functors = []
            for each_layer1_functor in layer1_functors:
                transformed_layers, _ , _ , _ = transform_resnet_block_layer(each_layer1_functor,layer_idx, is_first_linear, is_last_linear,quantize=quant)
                for each_transformed_layer in transformed_layers:
                    self.layer1_functors.append(each_transformed_layer)
            ##layer2
            self.layer2_functors = []
            for each_layer2_functor in layer2_functors:
                transformed_layers, _ , _ , _ = transform_resnet_block_layer(each_layer2_functor, layer_idx, is_first_linear, is_last_linear,quantize=quant)
                for each_transformed_layer in transformed_layers:
                    self.layer2_functors.append(each_transformed_layer)
            ## layer3
            if self.stack > 0 and self.res_block == 0:
                self.layer3_functors = []
                for each_layer3_functor in layer3_functors:
                    transformed_layers, _ , _ , _ = transform_resnet_block_layer(each_layer3_functor, layer_idx, is_first_linear, is_last_linear,quantize=quant)
                    for each_transformed_layer in transformed_layers:
                        self.layer3_functors.append(each_transformed_layer)
            ## layer4
            self.layer4_functors = []
            for each_layer4_functor in layer4_functors:
                transformed_layers , _ , _ , _ = transform_resnet_block_layer(each_layer4_functor, layer_idx, is_first_linear, is_last_linear,quantize=quant)
                for each_transformed_layer in transformed_layers:
                    self.layer4_functors.append(each_transformed_layer)
    def call(self, inputs):
        x = inputs
        y = inputs
        range_w = 2**8
        for each_functor in self.layer1_functors:
            y = each_functor(y)
        for each_functor in self.layer2_functors:
            y = each_functor(y)
        
        if self.stack > 0 and self.res_block == 0:  
            for each_functor in self.layer3_functors:
                x = each_functor(x)
        ## don't fucking modify this line
        elif self.quant:
            assert(not hasattr(self,'layer3_functors'))
            x = x * range_w
        x = self.layer4_functors[0]([x, y])
        x = self.layer4_functors[1](x)
        return x
    def get_config(self):
        config = {
            "num_filters":self.num_filters,
            "stack":self.stack,
            "res_block":self.res_block,
            "strides":self.strides,
            "transformed":self.transformed,
            "quant": self.quant,
            "layer_idx":self.layer_idx,
            "is_first_linear":self.is_first_linear,
            "is_last_linear": self.is_last_linear
        }

        base_config = super(ResNetBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature 
    map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of 
    filters is
    doubled. Within each stage, the layers have the same number 
    filters and the same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:  
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            # first layer but not first stack
            if stack > 0 and res_block == 0:  
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v1_obf(input_shape, depth,obf_scalar_stack = [], num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature 
    map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of 
    filters is
    doubled. Within each stage, the layers have the same number 
    filters and the same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    
    counter = 0
    x = resnet_layer_obf(inputs=inputs,
                        block_id= counter,
                        scalar_stack = obf_scalar_stack,use_mask=False)
    counter += 1

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:  
                strides = 2  # downsample
            y = resnet_layer_obf(inputs=x,
                             block_id = counter,
                             scalar_stack = obf_scalar_stack,
                             num_filters = num_filters,
                             strides = strides)
            counter += 1
            y = resnet_layer_obf(inputs=y,
                             block_id = counter, 
                             num_filters = num_filters,
                             activation = None)
            counter += 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:  
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer_obf(inputs=x,
                                 block_id = counter,
                                 scalar_stack = obf_scalar_stack,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
                counter += 1
            x = tf.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v1_split(json_obj,input_shape = train_imgs.shape[1:], depth = 44,obf_scalar_stack = [], num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    with open("./obf_model_conv_input_dict.json",'r') as rfile:
        obf_model = json.load(rfile)
    inputs = Input(shape=input_shape)
    counter = 0
    x = resnet_layer_split(inputs=inputs,
                        block_id= counter,
                        json_obj=json_obj,
                        json_dict={'cache_ac':True,'cache_conv':False},
                        obf_model=obf_model,
                        scalar_stack = obf_scalar_stack,use_mask=False)
    counter += 1

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:  
                strides = 2  # downsample
                ## correct the last conv json dict's cache ac to False
                last_conv_name = list(json_obj.keys())[-1]
                assert(json_obj[last_conv_name]['cache_ac'] == True)
                json_obj[last_conv_name]['cache_ac'] = False
            
            
            y = resnet_layer_split(inputs=x,
                             block_id = counter,
                             json_obj=json_obj,json_dict={'cache_ac':False,'cache_conv':False},
                             obf_model=obf_model,
                             scalar_stack = obf_scalar_stack,
                             num_filters = num_filters,
                             strides = strides)
            counter += 1
            if stack > 0 and res_block == 0:
                ## y is conv_16
                y = resnet_layer_split(inputs=y,
                             block_id = counter, 
                             json_obj=json_obj,
                             json_dict={'cache_ac':True,'cache_conv':False},
                             obf_model=obf_model,
                             num_filters = num_filters)
                counter += 1

                ## x is conv_17
                x = resnet_layer_split(inputs=x,
                                 block_id = counter,
                                 json_obj=json_obj,json_dict={'cache_ac':False,'cache_conv':True},
                                 obf_model=obf_model,
                                 scalar_stack = obf_scalar_stack,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides)
                counter += 1
                ## x first
                y = tf.keras.layers.add([x, y])
            else:
                y = resnet_layer_split(inputs=y,
                             block_id = counter, 
                             json_obj=json_obj,json_dict={'cache_ac':True,'cache_conv':False},
                             obf_model=obf_model,
                             num_filters = num_filters)
                counter += 1
            # first layer but not first stack
            
            x = y
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    '''
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    '''
    outputs = [x]
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:  
                strides = 2  # downsample
            x = ResNetBlock(stack=stack, res_block=res_block,num_filters=num_filters, strides=strides)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train():
    ## data_set
    
    
    global train_imgs,train_labels,test_imgs,test_labels 
    # Training parameters
    BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128
    EPOCHS = 200 # 200
    USE_AUGMENTATION = True
    NUM_CLASSES = np.unique(train_labels).shape[0] # 10
    COLORS = train_imgs.shape[3]
    input_shape = train_imgs.shape[1:]
    SUBTRACT_PIXEL_MEAN = True
    DEPTH = 404
    checkpoint_path = "training_"+str(DEPTH)+"_orig/cp-{epoch:04d}.ckpt"
    VERSION = 1
    
    


    # Normalize data.
    train_imgs = train_imgs.astype('float32') / 255
    test_imgs = test_imgs.astype('float32') / 255
    train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
    test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)

    model = resnet_v1(input_shape=input_shape, depth=DEPTH)


    ## prepare to train
    from tf.keras.optimizers import SGD
    from tf.keras.callbacks import LearningRateScheduler
    from tf.keras.callbacks import ReduceLROnPlateau
    from tf.keras.preprocessing.image import ImageDataGenerator
    model.compile(loss='categorical_crossentropy', 
        optimizer=SGD(lr=0.001), metrics=['accuracy'])
    #model.summary()

    

    # Prepare callbacks for model saving and for learning rate adjustment.
    #lr_scheduler = LearningRateScheduler(lr_schedule)
    #lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                           cooldown=0,
    #                           patience=5,
    #                           min_lr=0.5e-6)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=5*BATCH_SIZE)
    #callbacks = [lr_reducer, lr_scheduler,cp_callback]
    callbacks = [cp_callback]
    if not USE_AUGMENTATION:
        print('Not using data augmentation.')
        history=model.fit(train_imgs, train_labels,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(test_imgs, test_labels),
                  shuffle=True,
                  callbacks=callbacks)

    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
         # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(train_imgs)

        # Fit the model on the batches generated by datagen.flow().
        history=model.fit_generator(datagen.flow(train_imgs, train_labels, batch_size=BATCH_SIZE),
                            validation_data=(test_imgs, test_labels),
                            epochs=EPOCHS, verbose=1, workers=4,
                            callbacks=callbacks)
    scores = model.evaluate(test_imgs, test_labels, verbose=1)
    
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.save("resnet-{}.h5".format(DEPTH))

def train_new_resnet():
    ## data_set
    
    
    global train_imgs,train_labels,test_imgs,test_labels 
    # Training parameters
    BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128
    EPOCHS = 200 # 200
    USE_AUGMENTATION = True
    NUM_CLASSES = np.unique(train_labels).shape[0] # 10
    COLORS = train_imgs.shape[3]
    input_shape = train_imgs.shape[1:]
    SUBTRACT_PIXEL_MEAN = True
    DEPTH = 404
    checkpoint_path = "training_"+str(DEPTH)+"_block/cp-{epoch:04d}.ckpt"
    VERSION = 1
    
    


    # Normalize data.
    train_imgs = train_imgs.astype('float32') / 255
    test_imgs = test_imgs.astype('float32') / 255
    train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
    test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)

    model = resnet_v2(input_shape=input_shape, depth=DEPTH)


    ## prepare to train
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.callbacks import LearningRateScheduler
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    model.compile(loss='categorical_crossentropy', 
        optimizer=SGD(lr=0.001), metrics=['accuracy'])
    #model.summary()

    

    # Prepare callbacks for model saving and for learning rate adjustment.
    #lr_scheduler = LearningRateScheduler(lr_schedule)
    #lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                           cooldown=0,
    #                           patience=5,
    #                           min_lr=0.5e-6)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=5*BATCH_SIZE)
    #callbacks = [lr_reducer, lr_scheduler,cp_callback]
    callbacks = [cp_callback]
    if not USE_AUGMENTATION:
        print('Not using data augmentation.')
        history=model.fit(train_imgs, train_labels,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(test_imgs, test_labels),
                  shuffle=True,
                  callbacks=callbacks)

    else:
        print("v2 using data augmentation")
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
         # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(train_imgs)

        # Fit the model on the batches generated by datagen.flow().
        history=model.fit_generator(datagen.flow(train_imgs, train_labels, batch_size=BATCH_SIZE),
                            validation_data=(test_imgs, test_labels),
                            epochs=EPOCHS, verbose=1, workers=4,
                            callbacks=callbacks)
    scores = model.evaluate(test_imgs, test_labels, verbose=1)
    
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    model.save("resnet-{}.h5".format(DEPTH))

if __name__ == '__main__':
    ## be nice to others
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    train_new_resnet()
