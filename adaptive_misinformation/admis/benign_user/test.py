import numpy as np
import argparse
import os
import os.path as osp
from tqdm import tqdm
import torch
import pandas as pd

from torch.utils.data import DataLoader

import admis.utils.datasets as datasets
from admis.utils.blackbox import Blackbox

def CXE(predicted, target):
    target = target.float()
    predicted = predicted.float()
    eps = 1e-7
    return -(target * torch.log(predicted+eps)).sum(dim=1)

def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)

    # ----------- Defense
    parser.add_argument('--defense', metavar='DEF', type=str,
                        help='No Defense(ND)/ Prediction Poisoning(PP)/ Selective Misinformation (SM)', default='ND')
    parser.add_argument('--defense_levels', metavar='TYPE', type=str,
                        help='comma separated values specifying defense levels: epsilon(for PP)/delta(SM)',
                        default='0.')
    parser.add_argument('-T', '--temp', metavar='T', type=float, help='Temperature Scaling', default=1.)
    parser.add_argument('--no_adaptive', action='store_true', help='Use simple misinformation', default=False)
    parser.add_argument('--rand_fhat', action='store_true', help='Use randomly initialized misinformation function',
                        default=False)

    args = parser.parse_args()
    params = vars(args)
    if params['defense'] == 'ND':
        defense_levels = [0]
    else:
        defense_levels = [float(val) for val in params['defense_levels'].split(',')]

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    testset = dataset(train=False, transform=test_transform)
    num_classes = len(testset.classes)
    params['num_classes'] = num_classes
    batch_size = params['batch_size']
    if params['defense'] == 'PP':
        batch_size = 1
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=params['num_workers'])

    # ----------- Initialize blackbox
    blackbox_dir = '{}/{}/'.format(params['victim_model_dir'], params['defense'])
    use_adaptive = not params['no_adaptive']
    blackbox = Blackbox.from_modeldir(blackbox_dir, params['defense'], defense_levels, device=device,rand_fhat=params['rand_fhat'], use_adaptive=use_adaptive)

    # ----------- Evaluate Model
    total = 0
    correct = {}
    for def_level in defense_levels:
        correct[def_level] = 0
    print('defense_levels: ', defense_levels )
    correct_probs_list = []
    max_probs_list = []

    with tqdm(total=len(test_loader.dataset)) as pbar:
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = blackbox(inputs, T=params['temp'])
            total += targets.size(0)
            for i, def_level in enumerate(defense_levels):
                y = outputs[def_level]
                max_probs, predicted = y.max(1)
                if def_level == 0:
                    max_probs_list.append(max_probs)
                correct[def_level] += predicted.eq(targets).sum().cpu().item()
            pbar.update(inputs.size(0))
    test_acc = {}
    for def_level in defense_levels:
        test_acc[def_level] = 100. * float(correct[def_level]) / float(total)
    print('Test Accuracy: ', test_acc[defense_levels[0]])
    df = pd.DataFrame(test_acc.items(), index=None)
    print(df)
    if not osp.exists('./logs'):
        os.mkdir('./logs')
    df.to_csv('logs/acc_{}_{}.csv'.format(params['dataset'], params['defense']), index=None, header=None)
    blackbox.defense_fn.get_stats()

if __name__ == '__main__':
    main()
