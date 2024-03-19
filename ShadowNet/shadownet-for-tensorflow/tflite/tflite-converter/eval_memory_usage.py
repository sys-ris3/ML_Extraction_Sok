#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform 
from shuffle_channel_layer import ShuffleChannel 

import os
import psutil

def measure_mem_usage(model_name):
    pid = os.getpid()
    python_process = psutil.Process(pid)
    
    memoryUse = python_process.memory_info()[0]/2.**20  # memory use in GB...I think
    print('memory use:', memoryUse, 'MB\n')
    
    interpreter = tf.lite.Interpreter(model_path=model_name)
    memoryUse = python_process.memory_info()[0]/2.**20  # memory use in GB...I think
    print(' load memory use:', memoryUse, 'MB\n')

    interpreter.allocate_tensors()

    memoryUse = python_process.memory_info()[0]/2.**20  # memory use in GB...I think
    print('allocate memory use:', memoryUse, 'MB\n')
    
    tensor_details = interpreter.get_tensor_details()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    #fh = open("test.input","b+w")
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    memoryUse = python_process.memory_info()[0]/2.**20  # memory use in GB...I think
    print('invoke memory use:', memoryUse, 'MB\n')

    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    
    print('\n')
    memoryUse = python_process.memory_info()[0]/2.**20  # memory use in GB...I think
    print('result memory use:', memoryUse, 'MB\n')

if __name__ == '__main__':
    model_list = ['mobilenet.tflite', 'resnet-44.tflite', 'minivgg.tflite']
    for m in model_list:
        print ("=============== measure ",m,"==============")
        measure_mem_usage(m)
        print("\n")
