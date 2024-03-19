import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import admis.utils.datasets as datasets
import knockoff.utils.utils as knockoff_utils
from admis.utils.blackbox import Blackbox
import knockoff.config as cfg


class RandomAdversary(object):
    def __init__(self, blackbox, queryset, defense_levels, batch_size=8):
        self.blackbox = blackbox
        self.queryset = queryset
        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()
        self.transferset = {}
        self._restart()
        self.defense_levels = defense_levels
        for def_level in self.defense_levels:
            self.transferset[def_level] = []

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)
        self.idx_set = set(range(len(self.queryset)))

    def get_transferset(self, budget, temp=1):
        start_B = 0
        end_B = budget
        pred_max_list = []

        with tqdm(total=budget) as pbar:
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                # idxs= range(t*self.batch_size, (t+1)*self.batch_size)
                idxs = np.random.choice(
                    list(self.idx_set),
                    replace=False,
                    size=min(
                        self.batch_size,
                        budget - len(self.transferset[self.defense_levels[0]]),
                    ),
                )
                self.idx_set = self.idx_set - set(idxs)

                if len(self.idx_set) == 0:
                    print("=> Query set exhausted. Now repeating input examples.")
                    self.idx_set = set(range(len(self.queryset)))

                x_t = torch.stack(
                    [self.queryset[i][0] for i in idxs]
                )  # .to(self.blackbox.device)
                y_mod = self.blackbox(x_t, T=temp)

                if self.blackbox.defense == None:
                    pred_max, _ = torch.max(y_mod, dim=1)
                else:
                    pred_max, _ = torch.max(y_mod[self.defense_levels[0]], dim=1)

                pred_max_list.append(pred_max)

                if hasattr(self.queryset, "samples"):
                    # Any DatasetFolder (or subclass) has this attribute
                    # Saving image paths are space-efficient
                    img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                else:
                    # Otherwise, store the image itself
                    # But, we need to store the non-transformed version
                    img_t = [self.queryset.data[i] for i in idxs]
                    if isinstance(self.queryset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                for def_level, y_t in y_mod.items():
                    for i in range(x_t.size(0)):
                        img_t_i = (
                            img_t[i].squeeze()
                            if isinstance(img_t[i], np.ndarray)
                            else img_t[i]
                        )
                        self.transferset[def_level].append(
                            (img_t_i, y_t[i].cpu().squeeze())
                        )

                pbar.update(x_t.size(0))

        return self.transferset


def main():
    parser = argparse.ArgumentParser(description="Construct transfer set")
    parser.add_argument(
        "victim_model_dir",
        metavar="PATH",
        type=str,
        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"',
    )
    parser.add_argument(
        "--out_dir",
        metavar="PATH",
        type=str,
        help="Destination directory to store transfer set",
        required=True,
    )
    parser.add_argument(
        "--budget",
        metavar="N",
        type=int,
        help="Size of transfer set to construct",
        required=True,
    )
    parser.add_argument(
        "--queryset",
        metavar="TYPE",
        type=str,
        help="Adversary's dataset (P_A(X))",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        metavar="TYPE",
        type=int,
        help="Batch size of queries",
        default=8,
    )

    # ----------- Defense
    parser.add_argument(
        "--defense",
        metavar="DEF",
        type=str,
        help="No Defense(ND)/ Selective Misinformation (SM)",
        default="ND",
    )
    parser.add_argument(
        "--defense_levels",
        metavar="TYPE",
        type=str,
        help="comma separated values specifying defense levels: delta(SM)",
        default="0.0",
    )
    parser.add_argument(
        "-T", "--temp", metavar="T", type=float, help="Temperature Scaling", default=1.0
    )

    # ----------- Other params
    parser.add_argument(
        "-d", "--device_id", metavar="D", type=int, help="Device id", default=0
    )
    parser.add_argument(
        "-w",
        "--nworkers",
        metavar="N",
        type=int,
        help="# Worker threads to load data",
        default=10,
    )

    args = parser.parse_args()
    params = vars(args)
    attack = "knockoff"
    if params["defense"] == "ND":
        defense_levels = [0.0]
    else:
        defense_levels = [float(val) for val in params["defense_levels"].split(",")]

    out_path = params["out_dir"]
    knockoff_utils.create_dir(out_path)
    torch.manual_seed(cfg.DEFAULT_SEED)
    if params["device_id"] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params["device_id"])
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ----------- Set up queryset
    queryset_name = params["queryset"]
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError(
            "Dataset not found. Valid arguments = {}".format(valid_datasets)
        )
    modelfamily = datasets.dataset_to_modelfamily[queryset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]["test"]
    queryset = datasets.__dict__[queryset_name](train=True, transform=transform)

    # ----------- Initialize blackbox
    blackbox_dir = "{}/{}/".format(params["victim_model_dir"], params["defense"])
    blackbox = Blackbox.from_modeldir(
        blackbox_dir, params["defense"], defense_levels, device
    )

    # ----------- Initialize adversary
    batch_size = params["batch_size"]
    nworkers = params["nworkers"]
    transfer_out_path = osp.join(out_path, "transferset.pickle")
    adversary = RandomAdversary(
        blackbox, queryset, defense_levels, batch_size=batch_size
    )

    if params["defense"] == "SM":
        print("=> constructing transfer set with selective misinformation...")
    else:
        print("=> constructing transfer set without defense...")

    transferset = adversary.get_transferset(params["budget"], temp=params["temp"])

    for def_level in defense_levels:
        out_path_eps = "{}/{}/{}/{}".format(
            out_path, attack, params["defense"], def_level
        )
        knockoff_utils.create_dir(out_path_eps)
        transfer_out_path = out_path_eps + "/transferset.pickle"
        with open(transfer_out_path, "wb") as wf:
            pickle.dump(transferset[def_level], wf)
        print(
            "=> transfer set ({} samples) written to: {}".format(
                len(transferset[def_level]), transfer_out_path
            )
        )

    blackbox.defense_fn.get_stats()

    # Store arguments
    params["created_on"] = str(datetime.now())
    params_out_path = osp.join(out_path, "params_transfer.json")
    with open(params_out_path, "w") as jf:
        json.dump(params, jf, indent=True)


if __name__ == "__main__":
    main()
