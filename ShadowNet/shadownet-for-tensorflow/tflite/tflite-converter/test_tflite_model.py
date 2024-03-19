#!/usr/bin/env python
import tensorflow as tf

#model_path="alexnetsplit.tflite"
#model_path="minivggsplit_new.tflite"
#model_path="mobilenet_obf_split.tflite"
#model_path="minivggsplit_new.tflite"
model_path="resnet_obf.tflite"
interpreter = tf.lite.Interpreter(model_path)
#interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tensor_details = interpreter.get_tensor_details()
print("input_details")
print(input_details)
print("outut_details")
print(output_details)
print("tensor_details")
print(tensor_details)
