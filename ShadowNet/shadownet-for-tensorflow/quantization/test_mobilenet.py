#!/usr/bin/env python
import tensorflow as tf
from tensorflow.python.keras.engine import training
from tensorflow.python.keras import layers
from tensorflow.keras import initializers, Model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Layer,Conv2D,Dense,InputLayer,AveragePooling2D,MaxPooling2D,Activation,Flatten,Reshape,Dropout,ZeroPadding2D,GlobalAveragePooling2D,BatchNormalization,Lambda,DepthwiseConv2D,ReLU, Input
from tensorflow.keras.activations import relu, softmax
from tensorflow.nn import relu6

from add_mask_layer import AddMask
from linear_transform_generic_layer import LinearTransformGeneric 
from linear_transform_layer import LinearTransform
from tee_shadow_generic_layer import TeeShadowGeneric
from shuffle_channel_layer import ShuffleChannel 
from tee_shadow_layer import TeeShadow
from tee_shadow_generic_layer import TeeShadowGeneric
from tee_shadow_generic_2inputs import TeeShadowGeneric_2Inputs

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

#model_name="mobilenet_auto_obf_quant.h5"
#model = tf.keras.models.load_model(model_name, custom_objects={'ActivationQ':ActivationQ,'TeeShadow':TeeShadow,'TeeShadowGeneric':TeeShadowGeneric,'AddMask':AddMask,'ShuffleChannel':ShuffleChannel,'LinearTransformGeneric':LinearTransformGeneric,'TeeShadowGeneric_2Inputs':TeeShadowGeneric_2Inputs})

def FakeMobileNet(input_shape):
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(img_input)
    x = layers.Conv2D(
        38,
        (3,3),
        padding='valid',
        use_bias=False,
        strides=(2,2),
        name='conv1_obf')(x)
    # layers below supposed to be in TEE
    x = LinearTransform(32, name="tee_linear_transform")(x)
    x = layers.ReLU(6., name='tee_conv1_relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1,1,32), name='tee_reshape_1')(x)
    x = layers.Conv2D(1000, (1, 1), padding='same', name='obf_conv_preds', use_bias=False)(x)
    x = layers.Reshape((1000,), name='tee_reshape_2')(x)
    #imagenet_utils.imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Activation(activation='softmax',
                          name='tee_predictions')(x)
    model = training.Model(img_input, x, name='mobilenet_fake')
    return model


fake_model = FakeMobileNet((224,224,3))
fake_model.save('mobilenet_fake_lt.h5')
