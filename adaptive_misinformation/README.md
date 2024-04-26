# PROJECT: Defending Against Model Stealing Attacks with Adaptive Misinformation

## 1. Setup & Installation
- You can use the `env-setup.sh` file to setup the virtual environments and then used `quick-test.sh` to quickly run the projects (all projects) or,
- Use `environment.yml` provided in the directory `ML_Extraction_Sok/adaptive_misinformation` to create the virtual environment.
```bash
$ cd ML_Extraction_Sok/adaptive_misinformation
$ conda env create -f environment.yml
$ conda activate admis
$ chmod +x setup.sh
$ ./setup.sh
$ export PYTHONPATH="$PYTHONPATH:<PATH>/knockoffnets:<PATH>/adaptivemisinformation" # Add KnockoffNets and AdaptiveMisinformation to PYTHONPATH; Replace <PATH> with the path containing knockoffnets/adaptivemisinformation dirs
```

## 2.  Running Experiments 
- The evaluated projects can work on sample models 
- The evaluated projects may not work on real world ML models
- Use the script file `evaluate.sh`  to reproduce the results the author claims. Also, with slights modifcations to `evaluate.sh`  file you can try out your own model.
```bash
$ chmod +x evaluate.sh
$ ./evaluate.sh 
```
#### NOTE:
You can modify the `evaluate.sh` file and test its effectiveness with different model architecture.

## 3. Power consumption
- The power consumption of the evaluated projects is consistent with those reported in Section 4.4.2 of our paper.
- Use [PCM](https://github.com/intel/pcm) tool in a parallel terminal to monitor the power consumption in real-time while running projects.
- To use the PCM tool for power consumption, please check the directory `ML_Extraction_Sok/Evaluation`


We suggest referring to the README file provied here `ML_Extraction_Sok/adaptive_misinformation/orig_README.md` for additional insights and the overall project workflow.


### Testing Steps involved 

The `evaluate.sh` file contains the following instructions - 

### Train Defender Model
#### Selective Misinformation

python admis/defender/train.py MNIST lenet -o models/defender/mnist -e 20 --lr 0.1 --lr-step 10 --log-interval 200 -b 128 --defense=SM --oe_lamb 1 -doe KMNIST


### Evaluate Attacks

### Benign User

python admis/benign_user/test.py MNIST models/defender/mnist --defense SM --defense_levels 0.99

#### KnockoffNets Attack

python admis/adv_user/transfer.py models/defender/mnist --out_dir models/adv_user/mnist --budget 50000 --queryset EMNISTLetters --defense SM --defense_levels 0.99

python ./admis/adv_user/train_knockoff.py models/adv_user/mnist lenet MNIST --budgets 50000 --batch-size 128 --log-interval 200 --epochs 20 --lr 0.1 --lr-step 10 --defense SM --defense_level 0.99

#### JBDA Attack

python admis/adv_user/train_jbda.py ./models/defender/mnist/ ./models/adv_user/mnist/ lenet MNIST --defense=SM --aug_rounds=6 --epochs=10 --substitute_init_size=150 --defense_level=0.99 --lr 0.01

Note:
1. '--defense_levels' refers to the values of tau in the context of Selective Misinformation.
2. Varying the value of --defense_levels can be used to obtain the defender accuracy vs clone accuracy trade-off curve
