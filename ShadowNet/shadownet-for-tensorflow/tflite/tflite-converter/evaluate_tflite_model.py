#!/usr/bin/env python
from numpy import asarray
from numpy import save
from numpy import load
import numpy as np
import tensorflow as tf
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform 
from shuffle_channel_layer import ShuffleChannel 



# mobilenet convert pair
#ORIG_MODEL='mobilenet_obf_filled_plan_A.h5'
#TF_MODEL='mobilenet_model.tflite'

#ORIG_MODEL='mobilenet_obf_filled_plan_A.h5'
#ORIG_MODEL='mobilenet_obf_custom.h5'
#ORIG_MODEL='testnet_obf_filled.h5'
#ORIG_MODEL='mobilenet_obf_custom_filled.h5'
#ORIG_MODEL='mobilenet_split.h5'
#model = tf.keras.models.load_model(ORIG_MODEL, custom_objects={'AddMask':AddMask,'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
#model.summary()

TF_MODEL='mobilenet_split.tflite'
#TF_MODEL='mobilenet_model.tflite'
#TF_MODEL='testnet_obf.tflite'
#TF_MODEL='testnet.tflite'
#TF_MODEL='mobilenet_v1_1.0_224.tflite.backup'

# Load the MobileNet tf.keras model.
#model = tf.keras.applications.MobileNetV2(
#            weights="imagenet", input_shape=(224, 224, 3))
#model = tf.keras.models.load_model(ORIG_MODEL)

# Convert the model.
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=TF_MODEL)
#interpreter.resizeInput(tensor_index, [num_batch, 100, 100, 3]);
#print(interpreter.get_tensor_details())
interpreter.allocate_tensors()

# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_content=tflite_model)
#interpreter.allocate_tensors()
tensor_details = interpreter.get_tensor_details()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
fh = open("test.input","b+w")
print("input_shape")
print(input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
#save('input_data.npy',input_data)
input_data.tofile(fh)
print("input_data %s, %s"%(input_data[0][0][0][0], input_data[0][0][0][1]))
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
tflite_results = interpreter.get_tensor(output_details[0]['index'])
save('tflite_output.npy',tflite_results)

#print("len of tensor_details: %s" % len(tensor_details))
#for i in range(len(tensor_details)):
#    idx = tensor_details[i]['index']
#    t = interpreter.get_tensor(idx)
#    print ("tensor name: %s, idx: %s" % (tensor_details[i]['name'], idx))
#    bs = t.tobytes()
#    if (len(bs) > 0 and len(bs) <= 4):
#        print ("bs[0-7]:(%s,%s,%s,%s)" %(bs[0],bs[1],bs[2],bs[3]))
#    elif (len(bs) > 4):
#        print ("bs[0-7]:(%s,%s,%s,%s,%s,%s,%s,%s)" %(bs[0],bs[1],bs[2],bs[3],bs[4],bs[5],bs[6],bs[7]))
#    else:
#        print("bs len 0")
#
#    if (type(t) == np.float32):
#        print ("float32 t[0]:(%s)" %(t))
#    else:
#        print("t.shape: " )
#        print(t.shape)
#        if len(t.shape) == 4:
#            print("t[0-2] %s %s %s" % (t[0][0][0][0], t[0][0][0][1],t[0][0][0][2]))
#        elif len(t.shape) == 3:
#            print("t[0-2] %s %s %s" % (t[0][0][0], t[0][0][1],t[0][0][2]))

# Test the TensorFlow model on random input data.
tf_results = model(tf.constant(input_data))

print ((tflite_results.shape))
#print ((tflite_results))
# Compare the result.
for tf_result, tflite_result in zip(tf_results, tflite_results):
    np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
    print ((tf_result.shape))
    print("a0: %f, b0:%f" % (tf_result[0], tflite_result[0]))
    print("a1: %f, b1:%f" % (tf_result[1], tflite_result[1]))
    #print("a0: %f, b0:%f" % (tf_result[0][0][0], tflite_result[0][0][0]))
    #print("a1: %f, b1:%f" % (tf_result[0][0][1], tflite_result[0][0][1]))
    #print("a2: %f, b2:%f" % (tf_result[0][0][2], tflite_result[0][0][2]))
    #print("a10,0: %f, b10,0:%f" % (tf_result[0][10][0], tflite_result[0][10][0]))
