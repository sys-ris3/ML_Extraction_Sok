#!/bin/bash

# Navigate to the adaptive_misinformation directory
cd $(pwd)/adaptive_misinformation
# Create conda environment from environment.yml for adaptive_misinformation
echo "Creating conda environment for adaptive_misinformation...."
if ! conda env create -f environment.yml; then
    echo "Error: Environment creation failed."
    exit 1
fi
echo "Installing other packages and dependencies using setup.sh..."
if [ -f setup.sh ]; then 
    echo "Executing setup.sh..."
    chmod +x ./setup.sh
    bash setup.sh
fi
echo "Environment creation completed for adaptive_misinformation."
# Activate conda environment
echo "Activate conda environment..."


# Navigate to the prediction-poisoning directory
cd ../prediction-poisoning
# Create conda environment from environment.yml for prediction-poisoning
echo "Creating conda environment for prediction-poisoning..."
if ! conda env create -f environment.yml; then
    echo "Error: Environment creation failed."
    exit 1
fi
echo "Installing other packages and dependencies using setup.sh..."
if [ -f setup.sh ]; then
    echo "Executing setup.sh..."
    chmod +x ./setup.sh 
    bash setup.sh
fi
echo "Environment creation completed for prediction-poisoning."
# Activate conda environment
echo "Activate conda environment..."


# Navigate to the DeepSniffer directory
cd ../DeepSniffer
# Create conda environment from environment.yml for DeepSniffer
echo "Creating conda environment for DeepSniffer...."
if ! conda env create -f environment.yml; then
    echo "Error: Environment creation failed."
    exit 1
fi

echo "Environment creation completed for DeepSniffer."
# Activate conda environment
echo "Activate conda environment..."
# Navigate to the specified directory
cd ../ML-Doctor

# Create conda environment from environment.yml for ML-Doctor
echo "Creating conda environment for ML-Doctor..."
if ! conda env create -f environment.yml; then
    echo "Error: Environment creation failed."
    exit 1
fi
echo "Environment creation completed for ML-Doctor."
# Activate conda environment
echo "Activate conda environment..."


