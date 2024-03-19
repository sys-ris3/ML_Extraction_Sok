#!/usr/bin/env python
import tensorflow as tf
from add_mask_layer import AddMask
from linear_transform_layer import LinearTransform 
from shuffle_channel_layer import ShuffleChannel 
from tee_shadow_layer import TeeShadow 
from tee_shadow_generic_layer import TeeShadowGeneric 
from AlexNet import AlexNet
has_mask = True
if has_mask is True:
    #MODEL="mobilenet.h5"
    #TFMODEL="mobilenet.tflite"
    #MODEL="mobilenet_obf_filled_plan_B.h5"
    #MODEL="mobilenet_obf_filled_plan_A.h5"
    #TFMODEL="mobilenet_obf.tflite"
    #MODEL="testnet_obf_filled.h5"
    #TFMODEL="testnet_obf.tflite"
    #MODEL="mobilenet_obf_split_filled.h5"
    #TFMODEL="mobilenet_obf_split.tflite"
    #MODEL="alexnetobf.h5"
    #TFMODEL="alexnetobf.tflite"
    #MODEL="minivgg.h5"
    #TFMODEL="minivgg.tflite"
    #MODEL="inception_v3.h5"
    #TFMODEL="inception_v3.tflite"
    #MODEL="alexnetsplit.h5"
    #TFMODEL="alexnetsplit.tflite"
    #MODEL="minivggsplit.h5"
    #TFMODEL="minivggsplit.tflite"
    MODEL="resnetsplit.h5"
    TFMODEL="resnetsplit.tflite"
else:
    MODEL="model.h5"
    TFMODEL="model.tflite"

#model = tf.keras.models.load_model(MODEL, custom_objects={'AlexNet':AlexNet, 'TeeShadow':TeeShadow,'AddMask':AddMask,'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel})
model = tf.keras.models.load_model(MODEL, custom_objects={'TeeShadowGeneric':TeeShadowGeneric})
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = False 
#converter.experimental_new_converter = True 
#converter.enable_mlir_converter=True
converter.allow_custom_ops = True
#converter.conversion_summary_dir = "summary" 
#converter.custom_opdefs="name: 'AddMask' input\_arg: { name: 'input' type: DT\_FLOAT } input\_arg: { name: 'weights' type: DT\_FLOAT } input\_arg: {name: 'rscalar' type:DT\_FLOAT} output\_arg: { name: 'masked' type: DT\_FLOAT } attr : { name: 'DT\_FLOAT' type:'float'}"
converter.dump_graphviz_dir="/tmp/graph_dir"
converter.dump_graphviz_video=True
converter.conversion_summary_dir = "/tmp/summary"
#custom_opdefs_str = (
#        'name: \'AddMask\' '
#        'input_arg: { name: \'input\' type: DT_FLOAT } '
#        'input_arg: { name: \'weights\' type: DT_FLOAT } '
#        'input_arg: { name: \'rscalar\' type: DT_FLOAT } '
#        'output_arg: { name: \'masked\' type: DT_FLOAT } '
#        'name: \'LinearTransform\' '
#        'input_arg: { name: \'input\' type: DT_FLOAT } '
#        'input_arg: { name: \'weights\' type: DT_INT32} '
#        'input_arg: { name: \'rscalar\' type: DT_FLOAT } '
#        'output_arg: { name: \'transformed\' type: DT_FLOAT } '
#        'name: \'ShuffleChannel\' '
#        'input_arg: { name: \'input\' type: DT_FLOAT } '
#        'input_arg: { name: \'weights\' type: DT_INT32} '
#        'input_arg: { name: \'rscalar\' type: DT_FLOAT } '
#        'output_arg: { name: \'shuffled\' type: DT_FLOAT } ')

custom_opdefs_str = (
        'name: \'LinearTransform\' '
        'input_arg: { name: \'input\' type: DT_FLOAT } '
        'input_arg: { name: \'weights\' type: DT_INT32} '
        'input_arg: { name: \'rscalar\' type: DT_FLOAT } '
        'output_arg: { name: \'transformed\' type: DT_FLOAT } ')

#custom_opdefs_str = (
#        'name: \'TFLite_Detection_PostProcess\' '
#        'input_arg: { name: \'raw_outputs/box_encodings\' type: DT_FLOAT } '
#        'input_arg: { name: \'raw_outputs/class_predictions\' type: DT_FLOAT } '
#        'input_arg: { name: \'anchors\' type: DT_FLOAT } '
#        'output_arg: { name: \'TFLite_Detection_PostProcess\' type: DT_FLOAT } '
#        'output_arg: { name: \'TFLite_Detection_PostProcess:1\' '
#        'type: DT_FLOAT } '
#        'output_arg: { name: \'TFLite_Detection_PostProcess:2\' '
#        'type: DT_FLOAT } '
#        'output_arg: { name: \'TFLite_Detection_PostProcess:3\' '
#        'type: DT_FLOAT } '
#        'attr : { name: \'h_scale\' type: \'float\'} '
#        'attr : { name: \'max_classes_per_detection\' type: \'int\'} '
#        'attr : { name: \'max_detections\' type: \'int\'} '
#        'attr : { name: \'nms_iou_threshold\' type: \'float\'} '
#        'attr : { name: \'nms_score_threshold\' type: \'float\'} '
#        'attr : { name: \'num_classes\' type: \'int\'} '
#        'attr : { name: \'w_scale\' type: \'int\'} '
#        'attr : { name: \'x_scale\' type: \'int\'} '
#        'attr : { name: \'y_scale\' type: \'int\'}')
opdefs = '{0}'.format(custom_opdefs_str)
print(opdefs)
converter.custom_opdefs=opdefs
tflite_model = converter.convert()
open(TFMODEL, "wb").write(tflite_model)
#print(dir(tflite_model))

interpreter = tf.lite.Interpreter(model_path="alexnetsplit.tflite")
interpreter.allocate_tensors()
