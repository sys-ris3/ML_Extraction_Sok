# DeepSniffer
DeepSniffer is a model extraction framework that predicts the model architecture of the victim models based on the architecture hints during their execution. Specifically, this project mainly provides the most important part: layer sequence prediction. The key concept of DeepSniffer is to transform the layer sequence to a sequence-to-sequence prediction problem.

## Setup & Installation

You can use the `env-setup.sh` file to setup the virtual environment, followed by this step to enter into virtual environment-

```bash
$ conda activate deepsniffer
```

1) If failed to use `env-setup.sh`, try running the `chmod +x setup.sh` followed by `./setup.sh` file  provied in the dir ./ml-doctor or to install everything manually follow the steps provided below.
    ```
        conda create --name deepsniffer python=3.6.2
        conda activate deepsniffer
        conda install tensorflow<2.0.0
        conda install scipy
        conda install matplotlib
        conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
    ```
2) Download the model checkpoint files from the [google drive](https://drive.google.com/drive/folders/1JrTkT9C0klWFMK4x-KSMqvvPJ7k3TL6U?usp=sharing).

`Note: Pytorch: 1.8.0 (Recommended) - Pytorch Installation [Refer to this](https://pytorch.org/get-started/previous-versions/)`

## Running Experiments

The below content is the part of original README.md file which was used to reproduce the result as claimed by [author](https://github.com/xinghu7788/DeepSniffer). During testing with other real-world model checkpoints, DeepSniffer was unable to infer layer sequences due to incompatible log files. Unfortunately, the model we used cannot be added to this repository due to limited storage capacity.

### Evaluation & Workflow

This project comprises of two parts: 1) Model extraction part: we provide the source code and data set for training and testing the layer sequence predictor which is the fundamental step for model extraction. 
2) Adversarial attack example: In the further step, we also provide the source code and trained substitute model checkpoints to evaluate the effectiveness of the extracted models on adversarial attacks (Step 2 is not our scope for evaluation).
#### Model Extraction
##### Layer Sequence Predictor Inference 
* **Predictors**: We provide the trained layer sequence predictor in /DeepSniffer/ModelExtraction/validate_deepsniffer/predictors， which can be used for predicting the layer sequence of the victim models with their architecture hints. 
* **Dataset**: We provide architecture hint feature file of several commonly-used DNN models (profiling on K40), in the following directory: DeepSniffer/ModelExtraction/dataset/typicalModels.
* **Scripts**: To infer the layer sequence of these victim models, run 
DeepSniffer/ModelExtraction/scripts/.infer_predictor_typicalmodels.sh. The results log files are stored in DeepSniffer/Results/Table4/logs. Run DeepSniffer/Results/Table4/results_analysis.py to display the final prediction error rate results.

##### Layer Sequence Predictor Training
* **Dateset**: We randomly generate computational graphs and profile the GPU performance counter information (kernel latency, read volume, write volume) during their execution to train the layer sequence predictor. The training and testing dataset is in the following directory: DeepSniffer/ModelExtraction/dataset/training_randomgraphs.

* **Scripts**: To train the layer sequence predictor for model extraction, run DeepSniffer/ModelExtraction/scripts/train_predictor.sh. The trained model is in the directory of DeepSniffer/ModelExtraction/training_deepsniffer and training log is under the model checkpoint file directory.

* **Results**: The training log files are in the following directory: DeepSniffer/Results/Figure6/logs.

### Energy consumption

- Use [PCM](https://github.com/intel/pcm) tool in an another terminal to monitor the energy consumption in real-time while running project before and after the attack/defense is applied.
- For energy consumption in real-time, you need to run the experiments in two scenarios, during predictor inference, and during pedictor training.
- We recommend referring to the `orig_README.md` file for detailed instructions on how to execute the two scenarios.
