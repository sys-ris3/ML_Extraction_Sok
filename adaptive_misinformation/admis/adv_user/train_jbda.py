#!/usr/bin/python

import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime
import time
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from torchvision import transforms

import knockoff.config as cfg
import admis.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
import admis.utils.datasets as datasets
import admis.utils.zoo as zoo
from admis.utils.blackbox import Blackbox
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
from admis.utils.model import test_step
from jbda_utils import *

def get_labels(X_sub, blackbox, defense_level, params):
    scores = []
    label_batch = 64
    X_sub_len = X_sub.shape[0]
    num_splits = 1 + int(X_sub_len / label_batch)
    splits = np.array_split(X_sub, num_splits, axis=0)

    # with tqdm(total=num_splits) as pbar:
    for x_sub in splits:
        score_batch = blackbox(to_var(torch.from_numpy(x_sub)))[defense_level]
        score_batch = score_batch.data.cpu().numpy()
        scores.append(score_batch)
        # pbar.update(1)
    scores = np.concatenate(scores)
    print('done labeling')

    if params['argmaxed']:
        y_sub = np.argmax(scores, axis=1)
    else:
        y_sub = scores

    blackbox.defense_fn.get_stats()
    blackbox.defense_fn.reset_stats()
    return y_sub, scores

def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('victim_out_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('out_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')


    # Optional arguments
    parser.add_argument('--substitute_init_size', metavar='S', type=int,
                        help='Number of examples in the initial substitute dataset (round 0)', default=128)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('-ae', '--aug_rounds', type=int, default=5, metavar='N',
                        help='number of data augmentation rounds (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)

    # Attacker's defense
    attack='jbda'
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))

    # Defender's Defense
    parser.add_argument('--defense', metavar='DEF', type=str,
                        help='No Defense(ND)/ Prediction Poisoning(PP)/ Selective Misinformation (SM)', default='ND')
    parser.add_argument('--defense_level', metavar='TYPE', type=str,
                        help='defense levels: epsilon(for PP)/delta(SM)',
                        default='0.')

    args = parser.parse_args()
    params = vars(args)
    attack='jbda'

    defense_level = float(params['defense_level'])
    defense_levels = [defense_level]

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    out_dir = '{}/{}/{}/{}/'.format(params['out_dir'], attack, params['defense'], defense_level)
    knockoff_utils.create_dir(out_dir)

    # ----------- Set up testset
    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    #trainset = dataset(train=True, transform=transforms.ToTensor())
    trainset = dataset(train=True, transform=transform)

    sub_init_size = params['substitute_init_size']
    print('sub_init_size: ', sub_init_size)
    train_loader = DataLoader(trainset, batch_size=sub_init_size, shuffle=False, num_workers=params['num_workers'])
    test_loader = DataLoader(testset, batch_size=sub_init_size, shuffle=False, num_workers=params['num_workers'])
    num_classes = len(testset.classes)

    # ----------- Initialize Blackbox
    blackbox_dir = '{}/{}/'.format(params['victim_out_dir'], params['defense'])
    blackbox = Blackbox.from_modeldir(blackbox_dir, params['defense'], defense_levels, device)

    # ----------- Set up Clone Model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model_clone = zoo.get_net(model_name, modelfamily, pretrained, dataset_name, num_classes=num_classes)
    model_clone = model_clone.to(device)

    #  Label seed data
    data_iter = iter(train_loader)
    X_sub, _ = data_iter.next()
    X_sub = X_sub.numpy()
    y_sub, _ = get_labels(X_sub, blackbox, defense_level, params)

    rng = np.random.RandomState()
    criterion_test = nn.CrossEntropyLoss(reduction='mean')
    if params['argmaxed']:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = model_utils.soft_cross_entropy
        #y_sub_oh = np.zeros([sub_init_size, num_classes], dtype='float32')
        #y_sub_oh[range(sub_init_size), y_sub] = 1.0
        #y_sub = y_sub_oh
    optimizer = optim.SGD(model_clone.parameters(), lr=params['lr'], momentum=params['momentum'], weight_decay=5e-4)

    # Train the substitute and augment dataset alternatively
    for aug_round in range(params['aug_rounds']):
        # model training
        # Indices to shuffle training set
        index_shuf = list(range(len(X_sub)))
        rng.shuffle(index_shuf)

        for epoch in range(params['epochs']):
            nb_batches = int(np.ceil(float(len(X_sub)) /
                                     params['batch_size']))
            assert nb_batches * params['batch_size'] >= len(X_sub)

            for batch in range(nb_batches):
                start, end = batch_indices(batch, len(X_sub),
                                           params['batch_size'])
                x = X_sub[index_shuf[start:end]]
                y = y_sub[index_shuf[start:end]]
                scores = model_clone(to_var(torch.from_numpy(x)))
                if params['argmaxed']:
                    loss = criterion(scores, to_var(torch.from_numpy(y).long()))
                else:
                    loss = criterion(scores, to_var(torch.from_numpy(y)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            test_loss, test_acc = test_step(model_clone, test_loader, None, criterion_test, device, epoch=epoch)

        # If we are not at last substitute training iteration, augment dataset
        if aug_round < params['aug_rounds'] - 1:
            print("[{}] Augmenting substitute training data.".format(aug_round))
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(model_clone, X_sub, y_sub, nb_classes=num_classes)


            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            y_sub, scores = get_labels(X_sub, blackbox, defense_level, params)


            #if aug_round == params['aug_rounds'] - 2:
            pred_max = np.max(scores, axis=1)
            np.savez_compressed('./logs/max_probs_stats', a=pred_max)



    print('final test acc: {}'.format(test_acc))
    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)



if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    seconds = end-start
    hour = int(seconds/(60*60))
    min = int(seconds/60) % 60
    sec = int(seconds) % 60
    print('Runtime: {}:{}:{}'.format(hour, min, sec))

