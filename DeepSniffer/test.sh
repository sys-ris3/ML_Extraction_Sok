#!/bin/bash

# Navigate to the directory
cd $(pwd)/ModelExtraction/scripts

# Layer Sequence Predictor Inference
# RUN... - 
chmod +x infer_predictor_typicalmodels.sh
./infer_predictor_typicalmodels.sh
cd ../../

# The results log files are stored in DeepSniffer/Results/Table4/logs
# To display the final prediction error rate results.

# RUN...  - 
# Navigate to the directory
cd $(pwd)/Results/Table4/logs
echo "python results_analysis.py"
python results_analysis.py
cd ../../../

# Layer Sequence Predictor Training
# RUN... 
# Navigate to the directory
cd $(pwd)/ModelExtraction/scripts
chmod +x train_predictor.sh
./train_predictor.sh
cd ../../

# The training log files are in the following directory: DeepSniffer/Results/Figure6/logs.
