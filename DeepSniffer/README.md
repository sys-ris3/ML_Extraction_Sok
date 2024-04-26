# PROJECT: DeepSniffer: A DNN Model Extraction Framework Based on Learning Architectural Hints

## 1. Setup & Installation
- You can use the `env-setup.sh` file to setup the virtual environments and then used `quick-test.sh` to quickly run the projects (all projects) or,
- Use `environment.yml` provided in the directory `ML_Extraction_Sok/DeepSniffer` to create the virtual environment.
```bash
$ cd ML_Extraction_Sok/DeepSniffer
$ conda env create -f environment.yml
$ conda activate deepsniffer
```

## 2.  Running Experiments 
- The evaluated projects can work on sample models 
- The evaluated projects may not work on real world ML models
- Use the script file `test.sh`  to reproduce the results the author claims.
```bash
$ chmod +x test.sh
$ ./test.sh 
```
#### NOTE:
We suggest referring to the README file provied here `ML_Extraction_Sok/adaptive_misinformation/orig_README.md` for additional insights and the overall project workflow.

## 3. Power consumption
- The power consumption of the evaluated projects is consistent with those reported in Section 4.4.2 of our paper.
- Use [PCM](https://github.com/intel/pcm) tool in a parallel terminal to monitor the power consumption in real-time while running projects.
- To use the PCM tool for power consumption, please check the directory `ML_Extraction_Sok/Evaluation`


We recommend consulting the [PAPER](https://dl.acm.org/doi/10.1145/3373376.3378460) for more information on the attack strategy utilized and other relevant details.

### Testing Steps involved 

The `test.sh` file contains the following instructions - 

#####  Running Experiments

The below content is the part of original README.md file which was used to reproduce the result as claimed by [author](https://github.com/xinghu7788/DeepSniffer). During testing with other real-world model checkpoints, DeepSniffer was unable to infer layer sequences due to incompatible log files. Unfortunately, the model we used cannot be added to this repository due to limited storage capacity.

###### Evaluation & Workflow

This project comprises of two parts: 1) Model extraction part: we provide the source code and data set for training and testing the layer sequence predictor which is the fundamental step for model extraction. 
2) Adversarial attack example: In the further step, we also provide the source code and trained substitute model checkpoints to evaluate the effectiveness of the extracted models on adversarial attacks. 
##### Model Extraction
###### Layer Sequence Predictor Inference 
* **Predictors**: We provide the trained layer sequence predictor in /DeepSniffer/ModelExtraction/validate_deepsniffer/predictors， which can be used for predicting the layer sequence of the victim models with their architecture hints. 
* **Dataset**: We provide architecture hint feature file of several commonly-used DNN models (profiling on K40), in the following directory: DeepSniffer/ModelExtraction/dataset/typicalModels.
* **Scripts**: To infer the layer sequence of these victim models, run 
DeepSniffer/ModelExtraction/scripts/.infer_predictor_typicalmodels.sh. The results log files are stored in DeepSniffer/Results/Table4/logs. Run DeepSniffer/Results/Table4/results_analysis.py to display the final prediction error rate results.

###### Layer Sequence Predictor Training
* **Dateset**: We randomly generate computational graphs and profile the GPU performance counter information (kernel latency, read volume, write volume) during their execution to train the layer sequence predictor. The training and testing dataset is in the following directory: DeepSniffer/ModelExtraction/dataset/training_randomgraphs.

* **Scripts**: To train the layer sequence predictor for model extraction, run DeepSniffer/ModelExtraction/scripts/train_predictor.sh. The trained model is in the directory of DeepSniffer/ModelExtraction/training_deepsniffer and training log is under the model checkpoint file directory.

* **Results**: The training log files are in the following directory: DeepSniffer/Results/Figure6/logs.
