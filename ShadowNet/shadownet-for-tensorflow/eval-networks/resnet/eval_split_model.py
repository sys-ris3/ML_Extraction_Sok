from tee_shadow_generic_layer import TeeShadowGeneric
from add_mask_layer import AddMask
from linear_transform_generic_layer import LinearTransformGeneric 
from shuffle_channel_layer import ShuffleChannel

import tensorflow as tf
from IPython import embed

model = tf.keras.models.load_model('./resnet-44-split.h5',custom_objects={'AddMask':AddMask, 'LinearTransformGeneric':LinearTransformGeneric,'ShuffleChannel':ShuffleChannel,'TeeShadowGeneric':TeeShadowGeneric})
embed()
