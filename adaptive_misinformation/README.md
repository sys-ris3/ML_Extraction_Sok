# Defending Against Model Stealing Attacks with Adaptive Misinformation

## Setup & Installation
You can use the `env-setup.sh` file to setup the virtual environment, if failed use the `adaptive_misinformation/environment.yml` file to create the virtual environment.
```bash
$ conda activate admis
$ export PYTHONPATH="$PYTHONPATH:<PATH>/knockoffnets:<PATH>/adaptivemisinformation" # Add KnockoffNets and AdaptiveMisinformation to PYTHONPATH; Replace <PATH> with the path containing knockoffnets/adaptivemisinformation dirs
```

## Running Experiments 

### Evaluation with different model architecture -
- Use the script file `evaluate.sh`  to reproduce the results the author claims. Also, with slights modifcations to `evaluate.sh`  file you can try out your own model.

```bash
$ chmod + ./evaluate.sh
$ ./evaluate.sh 
```
#### NOTE:
You can modify the `evaluate.sh` file and test its effectiveness with different model architecture.

### Energy consumption
- Use [PCM](https://github.com/intel/pcm) tool in an another terminal to monitor the energy consumption in real-time while running project before and after the attack/defense is applied.
- For energy consumption in real-time, you need to run the experiments in two scenarios, one when the model is vulnerable (before defense) and another, when the defense is applied to the model.
- We recommend referring to the `orig_README.md` file for detailed instructions on how to execute the two scen

The `evaluate.sh` file contains the following instructions - 

## Train Defender Model

### Selective Misinformation

python admis/defender/train.py MNIST lenet -o models/defender/mnist -e 20 --lr 0.1 --lr-step 10 --log-interval 200 -b 128 --defense=SM --oe_lamb 1 -doe KMNIST


## Evaluate Attacks

### Benign User

python admis/benign_user/test.py MNIST models/defender/mnist --defense SM --defense_levels 0.99

### KnockoffNets Attack

python admis/adv_user/transfer.py models/defender/mnist --out_dir models/adv_user/mnist --budget 50000 --queryset EMNISTLetters --defense SM --defense_levels 0.99

python ./admis/adv_user/train_knockoff.py models/adv_user/mnist lenet MNIST --budgets 50000 --batch-size 128 --log-interval 200 --epochs 20 --lr 0.1 --lr-step 10 --defense SM --defense_level 0.99

### JBDA Attack

python admis/adv_user/train_jbda.py ./models/defender/mnist/ ./models/adv_user/mnist/ lenet MNIST --defense=SM --aug_rounds=6 --epochs=10 --substitute_init_size=150 --defense_level=0.99 --lr 0.01

Note:
1. '--defense_levels' refers to the values of tau in the context of Selective Misinformation.

2. Varying the value of --defense_levels can be used to obtain the defender accuracy vs clone accuracy trade-off curve


## Credits

Parts of this repository have been adapted from https://github.com/tribhuvanesh/knockoffnets


