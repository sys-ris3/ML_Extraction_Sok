#!/bin/bash

# Set up Conda path
source /lclhome/tnaya002/.bashrc

 For power measurement
echo "If you want to monitor its power consumption, please run the following cmd in a different terminal"
echo " sudo Evaluation/pcm/build/bin/pcm "
echo "after that, press any key to continue ..."
read -n 1 -s -r

# Test Deep Sniffer
echo -e "\n\n"
echo "*******************************"
echo "Start testing Deep Sniffer ..."
echo "*******************************"
# Setup conda environment
conda activate ../../../anaconda3/envs/deepsniffer
cd DeepSniffer
## Deep Sniffer experiment started ...
echo "Deep Sniffer takes very long to finish. We let it run for 2 minutes, for demonstration purpose" 
echo "For complete experiments, please remove the `timeout` in the script" 
chmod +x test.sh
timeout 20s ./test.sh 
echo "*******************************"
echo "Deep Sniffer finishes with the set timemout"
echo "*******************************"
conda deactivate
sleep 2

 Test ModelXray 
echo -e "\n\n"
echo "*******************************"
echo "Start testing Modelxray ..."
echo "*******************************"
# Setup virtual environment
source  ../../sok_venvs/xray/bin/activate
cd ../ModelXRay
# ModelXray experiment started ...
chmod +x test.sh
./test.sh
echo "*******************************"
echo "ModelXray finishes"
echo "*******************************"
deactivate
sleep 2


# Test ML Doctor
echo -e "\n\n"
echo "*******************************"
echo "Start testing ML Doctor ..."
echo "*******************************"
# Setup conda environment
conda activate ../../../../anaconda3/envs/ml-doctor
cd ../ML-Doctor
# ML Doctor experiment started ...
chmod +x test.sh
timeout 20s ./test.sh
echo "*******************************"
echo "ML Doctor finishes with the set timemout"
echo "*******************************"
echo "press any key to continue with testing a real world model"
read -n 1 -s -r
echo "ML Doctor start testing real-world models"
chmod +x test_real.sh
./test_real.sh
echo "*******************************"
echo "ML Doctor failed testing real-world models"
echo "*******************************"
conda deactivate
sleep 2


 Test Adaptive Misinformation
echo -e "\n\n"
echo "*******************************"
echo "Start testing Adaptive Misinformation ..."
echo "*******************************"
# Setup conda environment
conda activate ../../../../anaconda3/envs/admis
cd ../adaptive_misinformation
# Adaptive Misinformation experiment started
chmod +x evaluate.sh
timeout 20s ./evaluate.sh
echo "*******************************"
echo "Adaptive Misinformation finishes with the set timemout"
echo "*******************************"
echo "press any key to continue with testing a real world model"
read -n 1 -s -r
echo "Adaptive Misinformation testing a real world model"
chmod +x ./evaluate_real.sh
./evaluate_real.sh
echo "*******************************"
echo "Adaptive Misinformation failed loading a real world model"
echo "*******************************"
sleep 2
conda deactivate
sleep 2


# Test Prediction Poison
echo -e "\n\n"
echo "*******************************"
echo "Start testing Prediction Poison ..."
echo "*******************************"
# Setup conda environment
conda activate ../../../../anaconda3/envs/predpoison
cd ../prediction-poisoning
# Prediction poison experiment started
chmod +x test.sh
timeout 20s ./test.sh
echo "Prediction Poison finishes with the set timeout"
sleep 2
conda deactivate