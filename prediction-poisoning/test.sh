#!/bin/bash
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
vic_dir=knockoffnets/models/victim/${p_v}-${f_v}-train-nodefense;
#vic_dir=knockoffnets/models/victim/
### No. of images queried by the attacker. With 60k, attacker obtains 99.05% test accuracy on MNIST at eps=0.0.
budget=1000 
### Initialization to the defender's surrogate model. 'scratch' refers to random initialization.
proxystate=scratch;
### Path to surrogate model
proxydir=knockoffnets/models/victim/${p_v}-${f_v}-train-nodefense-${proxystate}-advproxy
### Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}-mad_${ydist}-eps${eps}-${queryset}-B${budget}-proxy_${proxystate}-random
### Defense strategy
strat=mad
### Parameters to defense strategy, provided as a key:value pair string. 
defense_args="epsilon:${eps},objmax:True,ydist:${ydist},model_adv_proxy:${proxydir},out_path:${out_dir}"
### Batch size of queries to process for the attacker
batch_size=1

# Surrogate Models
echo python knockoffnets/knockoff/victim/train.py MNIST lenet --out_path models/victim/MNIST-lenet-train-nodefense-scratch-advproxy --device_id 1 --epochs 1 --train_subset 10 --lr 0.0
python knockoffnets/knockoff/victim/train.py MNIST lenet --out_path models/victim/MNIST-lenet-train-nodefense-scratch-advproxy --device_id 1 --epochs 1 --train_subset 10 --lr 0.0

#echo python knockoff/victim/train.py CUBS200 vgg16_bn --out_path models/victim/CUBS200-vgg16_bn-train-nodefense-scratch-advproxy --device_id 1 --epochs 1 --train_subset 10 --lr 0.0
#python knockoff/victim/train.py CUBS200 vgg16_bn --out_path models/victim/CUBS200-vgg16_bn-train-nodefense-scratch-advproxy --device_id 1 --epochs 1 --train_subset 10 --lr 0.0

#Simulate Attacker Interactions
echo $ python defenses/adversary/transfer.py random ${vic_dir} ${strat} ${defense_args} --out_dir ${out_dir} --batch_size ${batch_size} -d ${dev_id} --queryset ${queryset} --budget ${budget}
python defenses/adversary/transfer.py random ${vic_dir} ${strat} ${defense_args} --out_dir ${out_dir} --batch_size ${batch_size} -d ${dev_id} --queryset ${queryset} --budget ${budget}

#Train + Evaluate Attacker
echo $ python3.7 knockoffnets/knockoff/adversary/train.py ${out_dir} ${f_v} ${p_v} --budgets 50,100,500,1000,10000,60000 --log-interval 500 --epochs 50 -d ${dev_id}
python3.7 knockoffnets/knockoff/adversary/train.py ${out_dir} ${f_v} ${p_v} --budgets 50,100,500,1000,10000,60000 --log-interval 500 --epochs 50 -d ${dev_id}

#Evaluate Blackbox utility
echo $ python defenses/adversary/eval_bbox.py ${vic_dir} ${strat} ${defense_args} --out_dir ${out_dir} --batch_size ${batch_size} -d ${dev_id}
python defenses/adversary/eval_bbox.py ${vic_dir} ${strat} ${defense_args} --out_dir ${out_dir} --batch_size ${batch_size} -d ${dev_id}


## Test with different models and dataset
#CUBS200-vgg16_bn-train-nodefense
#CIFAR10-vgg16_bn-train-nodefense-scratch-advproxy
#CUBS200-vgg16_bn-train-nodefense