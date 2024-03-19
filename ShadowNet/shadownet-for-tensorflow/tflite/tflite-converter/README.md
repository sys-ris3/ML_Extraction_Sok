## Guidance to use the tools here
The tools here are for converting keras model to tensorflow lite model and 
compare the original model and the converted model on the same input to
make sure they are equivalent.

As the keras model needs to be loaded with tensorflow custom ops support,
this requires install tensorflow from official release with the following
commands:

```
# create python virtualenv ENV1
python3 -m venv ENV1
# activate ENV1
source ENV1/bin/activate
# upgrade pip to 20.0.0 if necessary
# install tensorflow 2.1.0
pip install --upgrade tensorflow
# install our custom_ops
pip install tensorflow_custom_ops-0.0.1-cp36-cp36m-linux_x86_64.whl
# install matplotlib required by testnet drawing
pip install matplotlib
```

Let's call the above tensorflow environment ENV1.

When we evaluate tensorflow lite model, we need to run the model within
a customized tensorflow enviornment where the tensorflow is built from
source with our newly added tensorflow lite kernels to support custom ops.
Let's call it ENV2. It can be created with the following commands:

```
# create python virtualenv ENV2
python3 -m venv ENV2
# activate ENV2
source ENV2/bin/activate
# install our modified tensorflow 2.1.0 with custom tflite ops
pip install tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl
```
For the following list, if marked with `ENV1`, means this program needs
to be run in `ENV1`, the same for `ENV2`.

`convert_h5_to_tflite.py`: [ENV1] convert a given keras model in `h5` format into
tensorflow lite model.

`evaluate_tflite_model.py`: [ENV2] evaluate the tflite model and save the random input and model output.

`compare_models.py`: [ENV1] compare tflite model results with original keras model results.

## Special Guidance to Bypass TFLite Converter Bug

### Problem Description
So far, we can use tflite python interface of model converter to 
convert obfuscated mobilenet keras format model to tflite format model, 
however, the converted model can not be invoked due to assertion fail in
`strided_slice` operation. Please refer to 
[issue#13](https://github.com/RiS3-Lab/ModelSafe-Code/issues/13) for 
details.

### Dirty Hack
After inspecting the issue, I find that the tensorflow2.0+ is using a
different tflite model converter(not the same as tensorflow1.0+). 
However, as we use custom ops in our model which requires tensorflow2.0+,
we can not go back to tensorflow1.0+ to use its model converter.
Luckyly, the tflite converter has a cmd line interface, which allow us
to use tensorflow2.0+ framework but selectively use tensorflow1.0+ model
converter with flag `--enable_v1_converter`.

The only drawback is the cmdline interface doesnot allow me to provide
custom op definition, so the converter will complain. However, internally,
the cmdline interface will call a converter which takes custom op 
definition, which I can't access through the exposed cmdline interface.

The dirty hack I did is to add custom op definition to the internal 
library which the cmdline interface internally use. To achieve that,
I need to modify the tensorflow python library.


### Steps to Use the Dirty Hack

#### Setting up Env
- Build tensorflow2.2 from source. Note, make sure you add flag
`--cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0` to make the tensorflow compatable 
with the custom ops library we are going to build. Example scripts is 
provided `ModelSafe-Code/scripts/build_tensorflow.sh`.

- Build custom ops from source. Refer to example scripts `ModelSafe-Code/scripts/useful_cmds.sh`.

- Create Virtual Python Env `venvtf22`; Install python packages of tensorflow and custom\_ops.

- Hack installed tensorflow library code to support custom ops for tflite
model converter.
  - locate library code `/root/venvtf22/lib/python3.6/site-packages/tensorflow/lite/python/lite.py` 
  - copy custom layer code to library.
```
cp add_mask_layer.py /root/venvtf22/lib/python3.6/site-packages/tensorflow/lite/python/
cp linear_transform_layer.py /root/venvtf22/lib/python3.6/site-packages/tensorflow/lite/python/
cp shuffle_channel_layer.py /root/venvtf22/lib/python3.6/site-packages/tensorflow/lite/python/
```
  - add import code to lite.py
```
 from tensorflow.lite.python.add_mask_layer import AddMask
 from tensorflow.lite.python.linear_transform_layer import LinearTransform
 from tensorflow.lite.python.shuffle_channel_layer import ShuffleChannel
```
  - change function parameter like:

```
 @_tf_export(v1=["lite.TFLiteConverter"])
 class TFLiteConverter(TFLiteConverterBase):
   ...
   def from_keras_model_file(cls,
     model_file,
     input_arrays=None,
     input_shapes=None,
     output_arrays=None,
     custom_objects={'AddMask':AddMask,'LinearTransform':LinearTransform,'ShuffleChannel':ShuffleChannel}):
```

- Run the converter tool `tflite_convert_cmd.py` (modified from `tflite_convert_test.py`)
