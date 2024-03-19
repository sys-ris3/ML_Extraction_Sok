import argparse
import os.path as osp
import os
import json

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from knockoff.utils.type_checks import TypeCheck
import admis.utils.zoo as zoo
import admis.utils.datasets as datasets
from admis.utils.defense import selectiveMisinformation, noDefense


class Blackbox(object):
    def __init__(
        self,
        model,
        model_def,
        defense,
        defense_levels,
        device=None,
        num_classes=10,
        rand_fhat=False,
        use_adaptive=True,
    ):
        self.device = torch.device("cuda") if device is None else device
        self.__model = model.to(device)
        self.__model.eval()
        self.__call_count = 0
        self.defense = defense
        model_def = model_def.to(device)
        self.num_classes = num_classes

        if self.defense == "SM":
            self.defense_fn = selectiveMisinformation(
                model_def, defense_levels, self.num_classes, rand_fhat, use_adaptive
            )
        else:
            self.defense_fn = noDefense()

    @classmethod
    def from_modeldir(
        cls,
        model_dir,
        defense,
        defense_levels,
        device=None,
        rand_fhat=False,
        use_adaptive=True,
    ):
        device = torch.device("cuda") if device is None else device
        # What was the model architecture used by this model?
        params_path = osp.join(model_dir, "params.json")
        with open(params_path) as jf:
            params = json.load(jf)
        model_arch = params["model_arch"]
        num_classes = params["num_classes"]
        victim_dataset = params.get("dataset", "imagenet")
        modelfamily = datasets.dataset_to_modelfamily[victim_dataset]

        # Instantiate the model
        model = zoo.get_net(
            model_arch,
            modelfamily,
            pretrained=None,
            dataset=victim_dataset,
            num_classes=num_classes,
        )
        model_def = zoo.get_net(
            model_arch,
            modelfamily,
            pretrained=None,
            dataset=victim_dataset,
            num_classes=num_classes,
        )
        model = model.to(device)

        # Load weights
        checkpoint_path = osp.join(model_dir, "model_best.pth.tar")
        if not osp.exists(checkpoint_path):
            checkpoint_path = osp.join(model_dir, "checkpoint.pth.tar")
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint["epoch"]
        best_test_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc)
        )

        if defense == "SM":
            model_def_path = model_dir + "/model_poison.pt"
            if not rand_fhat:
                print("loading misinformation model")
                model_def.load_state_dict(torch.load(model_def_path))
            model_def = model_def.to(device)
            print(
                "=> loaded checkpoint for Misinformation model (used for Selective Misinformation)"
            )

        blackbox = cls(
            model,
            model_def,
            defense,
            defense_levels,
            device,
            num_classes,
            rand_fhat,
            use_adaptive,
        )
        return blackbox

    def __call__(self, x, T=1):
        TypeCheck.multiple_image_blackbox_input_tensor(x)
        with torch.no_grad():
            x = x.to(self.device)
            y = self.__model(x)
            self.__call_count += x.shape[0]
            y = F.softmax(y / T, dim=1)
        # print('y_before defense: ', y)
        y_mod = self.defense_fn(x, y)
        return y_mod
