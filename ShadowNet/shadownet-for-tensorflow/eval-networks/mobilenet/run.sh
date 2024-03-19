#!/bin/sh

echo "generate imagenet mobilenet model mobilenet.h5"
python mobilenet.py

echo "generate obfuscation template model with custom op mobilenet_obf_custom.h5"
python mobilenet_obf_custom_op.py

echo "convert original model mobilenet.h5 to obfuscated model mobilenet_obf_custom_filled.h5"
echo "Note: takes tens of minutes"
python mobilenet_model_converter.py

echo "evaluate the obfuscation"
python evaluate_obf_enc_scheme_mobilenet.py
