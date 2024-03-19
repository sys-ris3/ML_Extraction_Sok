import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import uniform


def compute_hellinger(y_a, y_b):
    """
    :param y_a: n x K dim
    :param y_b: n x K dim
    :return: n dim vector of hell dist between elements of y_a, y_b
    """
    diff = torch.sqrt(y_a) - torch.sqrt(y_b)
    sqr = torch.pow(diff, 2)
    hell = torch.sqrt(0.5 * torch.sum(sqr, dim=1))
    return hell.cpu().detach().numpy()


class selectiveMisinformation:
    def __init__(
        self, model_mis, delta_list, num_classes=10, rand_fhat=False, use_adaptive=True
    ):
        self.delta_list = delta_list
        self.input_count = 0
        self.ood_count = {}
        self.num_classes = num_classes
        self.correct_class_rank = np.zeros(self.num_classes)
        self.mis_correct_count = 0
        self.hell_dist = {}
        self.max_probs = {}
        self.alpha_vals = {}
        self.reset_stats()
        self.model_mis = model_mis
        self.use_adaptive = use_adaptive
        self.rand_fhat = rand_fhat

    def __call__(self, x, y):
        probs = y  # batch x 10
        probs_max, probs_max_index = torch.max(probs, dim=1)  # batch
        batch = probs_max.size(0)
        self.input_count += batch
        y_mis = self.model_mis(x)
        y_mis = F.softmax(y_mis, dim=1)
        probs_mis_max, probs_mis_max_index = torch.max(y_mis, dim=1)  # batch
        self.mis_correct_count += (probs_mis_max_index == probs_max_index).sum().item()
        y_mis_dict = {}
        for delta in self.delta_list:
            y_mis = y_mis.detach()
            if self.use_adaptive:
                h = 1 / (1 + torch.exp(-10000 * (delta - probs_max.detach())))
            else:
                h = delta * torch.ones_like(probs_max.detach())

            h = h.unsqueeze(dim=1).float()
            mask_ood = probs_max <= delta
            self.ood_count[delta] += np.sum(mask_ood.cpu().detach().numpy())
            y_mis_dict[delta] = ((1.0 - h) * y) + (h * y_mis.float())
            probs_mis_max, _ = torch.max(y_mis_dict[delta], dim=1)
            self.max_probs[delta].append(probs_mis_max.cpu().detach().numpy())
            self.alpha_vals[delta].append(h.squeeze(dim=1).cpu().detach().numpy())

            hell = compute_hellinger(y_mis_dict[delta], y)
            self.hell_dist[delta].append(hell)
        return y_mis_dict

    def get_stats(self):
        rejection_ratio = {}
        for delta in self.delta_list:
            rejection_ratio[delta] = float(self.ood_count[delta]) / float(
                self.input_count
            )
            print("Delta: {} Rejection Ratio: {}".format(delta, rejection_ratio[delta]))
            self.hell_dist[delta] = np.array(np.concatenate(self.hell_dist[delta]))
            self.max_probs[delta] = np.array(np.concatenate(self.max_probs[delta]))
            self.alpha_vals[delta] = np.array(np.concatenate(self.alpha_vals[delta]))
        print(
            "miss_correct_ratio: ",
            float(self.mis_correct_count) / float(self.input_count),
        )
        np.savez_compressed("./logs/hell_dist_sm", a=self.hell_dist)
        np.savez_compressed("./logs/max_probs", a=self.max_probs)
        np.savez_compressed("./logs/alpha_vals", a=self.alpha_vals)
        return rejection_ratio

    def reset_stats(self):
        for delta in self.delta_list:
            self.ood_count[delta] = 0
            self.hell_dist[delta] = []
            self.max_probs[delta] = []
            self.alpha_vals[delta] = []
        self.input_count = 0
        self.mis_correct_count = 0
        self.correct_class_rank = np.zeros(self.num_classes)


class noDefense:
    def __call__(self, x, y):
        y_noDef = {}
        y_noDef[0] = y
        return y_noDef

    def print_stats(self):
        return
