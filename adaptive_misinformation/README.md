# Defending Against Model Stealing Attacks with Adaptive Misinformation

Implementation of the paper "Defending Against Model Stealing Attacks with Adaptive Misinformation".

## Setup

1. `git clone https://github.com/sanjaykariyappa/adaptive_misinformation.git` or use the current repository `<path>/adaptive-misinformation`.

- Use the the PATCH file `adaptive-misinformation.patch` to update the code or, 
 
2. `conda env create -f environment.yml --name misinformation` # Creates Anaconda env with requirements

3. `conda activate misinformation`

3. `git clone https://github.com/tribhuvanesh/knockoffnets.git ` # Download KnockoffNets repository

4. export PYTHONPATH="$PYTHONPATH:<PATH>/knockoffnets:<PATH>/adaptivemisinformation" # Add KnockoffNets and AdaptiveMisinformation to PYTHONPATH; Replace <PATH> with the path containing knockoffnets/adaptivemisinformation dirs

5. Run `./evaluate.sh` to evaluate the results or follow the below steps to test AM.

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


