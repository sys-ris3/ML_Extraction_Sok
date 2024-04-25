#!/bin/bash
# RUN ML-DOCTOR...  
# Attack Type	Name
#   0	        MemInf
#   1	        ModInv
#   2           AttrInf
#   3           ModSteal

echo "python demo.py --attack_type 3 --dataset_name FMNIST"
python demo.py --attack_type 3 --dataset_name FMNIST
