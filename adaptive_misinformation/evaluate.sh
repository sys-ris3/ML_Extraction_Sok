### p_v = victim model dataset
p_v=MNIST
### f_v = architecture of victim model
f_v=lenet 

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/knockoffnets


echo "python admis/defender/train.py ${p_v} ${f_v} -o models/defender/mnist -e 20 --lr 0.1 --lr-step 10 --log-interval 200 -b 128 --defense=SM --oe_lamb 1 -doe KMNIST"
python admis/defender/train.py ${p_v} ${f_v} -o models/defender/mnist -e 20 --lr 0.1 --lr-step 10 --log-interval 200 -b 128 --defense=SM --oe_lamb 1 -doe KMNIST

## Evaluate Attacks

### Benign User
echo "python admis/benign_user/test.py MNIST models/defender/mnist --defense SM --defense_levels 0.99"
python admis/benign_user/test.py MNIST models/defender/mnist --defense SM --defense_levels 0.99


### KnockoffNets Attack
echo "python admis/adv_user/transfer.py models/defender/mnist --out_dir models/adv_user/mnist --budget 50000 --queryset EMNISTLetters --defense SM --defense_levels 0.99"
python admis/adv_user/transfer.py models/defender/mnist --out_dir models/adv_user/mnist --budget 50000 --queryset EMNISTLetters --defense SM --defense_levels 0.99

echo "python ./admis/adv_user/train_knockoff.py models/adv_user/mnist lenet MNIST --budgets 50000 --batch-size 128 --log-interval 200 --epochs 20 --lr 0.1 --lr-step 10 --defense SM --defense_level 0.99"

### JBDA Attack

echo "python admis/adv_user/train_jbda.py ./models/defender/mnist/ ./models/adv_user/mnist/ lenet MNIST --defense=SM --aug_rounds=6 --epochs=10 --substitute_init_size=150 --defense_level=0.99 --lr 0.01"
