#!/usr/bin/python
from torch.autograd import Variable
import torch
import numpy as np

def batch_indices(batch_nb, data_length, batch_size):
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = to_var(torch.from_numpy(x), requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        x_var_exp = x_var.unsqueeze(0)
        score = model(x_var_exp)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives


def jacobian_augmentation(model, X_sub_prev, Y_sub, lmbda=0.2, nb_classes=10):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_sub = np.vstack([X_sub_prev, X_sub_prev])
    if Y_sub.ndim == 2:
        # Labels could be a posterior probability distribution. Use argmax as a proxy.
        Y_sub = np.argmax(Y_sub, axis=1)

    # For each input in the previous' substitute training iteration
    offset = len(X_sub_prev)
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x, nb_classes)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
        grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        X_sub[offset+ind] = x + lmbda * grad_val #???
    print('jacobian aug max: ', np.max(X_sub))
    print('jacobian aug min: ', np.min(X_sub))


    # Return augmented training data (needs to be labeled afterwards)
    return X_sub

