# Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks, ICLR '20

**Tribhuvanesh Orekondy<sup>1</sup>, Bernt Schiele<sup>1</sup>, Mario Fritz<sup>2</sup>**   

<sup>1</sup> Max Planck Institute for Informatics   
<sup>2</sup> CISPA Helmholtz Center for Information Security

----

High-performance Deep Neural Networks (DNNs) are increasingly deployed in many real-world applications e.g., cloud prediction APIs. 
Recent advances in model functionality stealing attacks via black-box access (i.e., inputs in, predictions out) threaten the business model of such applications, which require a lot of time, money, and effort to develop. 
Existing defenses take a passive role against stealing attacks, such as by truncating predicted information. 
We find such passive defenses ineffective against DNN stealing attacks. 
In this paper, we propose the first defense which actively perturbs predictions targeted at poisoning the training objective of the attacker. 
We find our defense effective across a wide range of challenging datasets and DNN model stealing attacks, and additionally outperforms existing defenses. 
Our defense is the first that can withstand highly accurate model stealing attacks for tens of thousands of queries, amplifying the attacker's error rate up to a factor of 85× with minimal impact on the utility for benign users.

**tl; dr:** We propose the first approach that can resist DNN model stealing attacks

**Project webpage: [url](https://resources.mpi-inf.mpg.de/d2/orekondy/predpoison/)**

## Installation

You can use the current repository or clone the repository to test [prediction-poisoning](https://github.com/tribhuvanesh/knockoffnets.git):

`We have included a patch file that can be used to update the code or track the changes we made.`

```bash
$ cd prediction-posioning ,or
$ git clone --recursive https://github.com/tribhuvanesh/knockoffnets.git
$ conda create --name prediction python=3.7.16
$ conda activate prediction
$ conda env create -f environment.yml ,or
$ pip install -r requirements.txt
$ git clone https://github.com/tribhuvanesh/knockoffnets.git
$ cd knockoffnets
$ pip install -e .
```

Please refer to `orig_README.md` for more details about Datasets, Victim Models, Attack Models, Surrogate Models used by author. 

### Running Experiments

Use the script file (test.sh) to reproduce the results the author claims. Also, with slights modifcations to test.sh file you can try out your own model.

```bash
./test.sh
```

Use the test.sh file to run the experiment or follow the steps below.

The instructions below will execute experiments with the following setting:
 * Defense = MAD
 * Attack = Knockoff
 * Dataset (Victim model) = MNIST
 * Queryset = EMNISTLetters (i.e., images queried by the attacker)

Most of these parameters can be changed by simply substituting the variables with the one you want.

#### Step 1: Setting up experiment variables or use the tesh.sh script file (change the victim model's directory with our own model files)

The configuration for experiments is provided primarily via command-line arguments. 
Since some of these arguments are re-used between experiments (e.g., attack and defense models), it's convenient to assign the configuration in shell variables and just reference them in the command-line arguments (which you will see in the next steps).
To do this, copy-paste the block below into command-line.

```bash
### If you have multiple GPUs on the machine, use this to select the specific GPU
dev_id=0
### Metric for perturbation ball dist(y, y'). Supported = L1, L2, KL
ydist=l1
### Perturbation norm
eps=0.5
### p_v = victim model dataset
p_v=MNIST
### f_v = architecture of victim model
f_v=lenet 
### queryset = p_a = image pool of the attacker 
queryset=EMNISTLetters
### Path to victim model's directory (the one downloded earlier)
vic_dir=models/victim/${p_v}-${f_v}-train-nodefense;
### No. of images queried by the attacker. With 60k, attacker obtains 99.05% test accuracy on MNIST at eps=0.0.
budget=60000 
### Initialization to the defender's surrogate model. 'scratch' refers to random initialization.
proxystate=scratch;
### Path to surrogate model
proxydir=models/victim/${p_v}-${f_v}-train-nodefense-${proxystate}-advproxy
### Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}-mad_${ydist}-eps${eps}-${queryset}-B${budget}-proxy_${proxystate}-random
### Defense strategy
strat=mad
### Parameters to defense strategy, provided as a key:value pair string. 
defense_args="epsilon:${eps},objmax:True,ydist:${ydist},model_adv_proxy:${proxydir},out_path:${out_dir}"
### Batch size of queries to process for the attacker
batch_size=1
```

It is vital to retain these variables when running the subsequent commands when executes the model stealing attack under the configured defense.

#### Step 2: Simulate Attacker Interactions

The command below constructs the attacker's transfer set i.e., images and their corresponding pseudo-labels (perturbed posteriors) obtained by querying the defended blackbox.
The defense is configured by `strat` and `defense_args` variables.

```bash
$ python defenses/adversary/transfer.py random ${vic_dir} ${strat} ${defense_args} \
    --out_dir ${out_dir} \
    --batch_size ${batch_size} \
    -d ${dev_id} \
    --queryset ${queryset} \
    --budget ${budget}
```

The command produces a `${out_dir}/queries.pickle` file containing the image-(perturbed) prediction pairs.
Additionally, the file `${out_dir}/distancetransfer.log.tsv` logs the mean and standard deviations of L1, L2, and KL between the original and perturbed predictions.

#### Step 3: Train + Evaluate Attacker

After the transfer set (i.e., attacker's training set) is constructed, the command below trains multiple attack models for various choices of sizes of transfer sets (specified by `budgets`).
During training, the model is simulatenously evaluated during each epoch. 

```bash
python knockoff/adversary/train.py ${out_dir} ${f_v} ${p_v} \
    --budgets 50,100,500,1000,10000,60000 \
    --log-interval 500 \
    --epochs 50 \
    -d ${dev_id}
``` 

The train and test accuracies of the attack model (against MAD defense@eps) are logged to `${out_dir}/train.<budget>.log.tsv`.

#### Step 4: Evaluate Blackbox utility

The utility of the defended blackbox is evaluated by computing 
  * the test-set accuracy (i.e., ) with perturbed predictions on the test image set
  * perturbation magnitude norms introduced as a result

```bash
python defenses/adversary/eval_bbox.py ${vic_dir} ${strat} ${defense_args} \
    --out_dir ${out_dir} \
    --batch_size ${batch_size} \
    -d ${dev_id}
```

The utility metrics will be logged to `${out_dir}/bboxeval.<testsetsize>.log.tsv` (test accuracies) and `${out_dir}/distancetest.log.tsv` (perturbation magnitudes).
