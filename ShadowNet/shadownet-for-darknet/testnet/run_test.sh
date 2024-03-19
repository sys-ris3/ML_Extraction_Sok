#!/bin/sh

echo "generate original model"
python testnet.py

echo "generate template model"
python testnet_tpl.py

echo "generate converted model with template model"
python model_converter.py

echo "evaluate converted model against original model"
python model_evaluater.py
