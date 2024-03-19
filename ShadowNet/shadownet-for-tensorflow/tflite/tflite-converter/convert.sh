#!/bin/sh
# arg1: path, like ../alexnet/
# arg2: model name, like alexnet
cp $1/$2".h5" .
./tflite_convert_cmd.py -i $2".h5" -o $2".tflite"
cp $2".tflite" /tmp/

