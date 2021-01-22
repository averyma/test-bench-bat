import os
import copy

import torch
import torch.nn as nn

from tqdm import trange
import numpy as np

from src.attacks import pgd_rand, fgsm
from src.context import ctx_noparamgrad_and_eval
from src.utils_general import ep2itr

def data_init(init, X, y, model):
    if init == "rand":
        delta = torch.empty_like(X.detach(), requires_grad=False).uniform_(-8./255.,8./255.)
        delta.data = (X.detach() + delta.detach()).clamp(min = 0, max = 1.0) - X.detach()
    elif init == "fgsm":
        with ctx_noparamgrad_and_eval(model):
            param = {"ord":np.inf, "epsilon": 2./255.}
            delta = fgsm(**param).generate(model,X,y)
    elif init == "pgd1":
        with ctx_noparamgrad_and_eval(model):
            param = {"ord":np.inf, "epsilon": 8./255., "alpha":2./255., "num_iter": 1, "restart": 1}
            delta = pgd_rand(**param).generate(model,X,y)
    elif init == "none":
        delta = torch.zeros_like(X.detach(), requires_grad=False)

    return delta

def bat_init(init, X, y, bat_k, bat_step, j_dir, itr, model, model_k_list, device):
    model_path = j_dir + "/model/"

    if init == "fgsm": 
        attack = fgsm
        param = {'ord': np.inf,
                 'epsilon': 8./255.}
    elif init == "pgd":
        attack = pgd_rand
        param = {'ord': np.inf,
                 'epsilon': 8./255.,
                 'alpha': 2./255.,
                 'num_iter': 10,
                 'restarts': 1}


    if itr < bat_k*bat_step:
        with ctx_noparamgrad_and_eval(model):
            delta = attack(**param).generate(model, X, y)
            X_bat = X + delta
            return X_bat, y

    """
    I am cheating by directly using the models saved at model_path.
    This could be problematic if checkpoint is not handled properly.
    A more proper way is to compute the valid model idx based on 
    itr, bat_step and bat_k. 
    """
    # curr_model_list = os.listdir(j_dir+"/model/")
    # curr_k = len(curr_model_list)
    # bat_k = curr_k if curr_k < bat_k else bat_k
    # bat_len = int(X.shape[0] // bat_k * bat_k)
    bat_len_per_k = int(X.shape[0]/bat_k)
    X_bat = torch.zeros_like(X, device = device)
    # y_bat = y[:bat_len]

    for i in range(bat_k):

        # model_k_path = model_path + curr_model_list[i]
        # model_k = copy.deepcopy(model)
        # model_k.load_state_dict(torch.load(model_k_path, map_location=device))
        # model_k.to(device)
        model_k = model_k_list[i]
        with ctx_noparamgrad_and_eval(model_k):
            X_attack = X[i*bat_len_per_k : (i+1)*bat_len_per_k,:,:,:]
            y_attack = y[i*bat_len_per_k : (i+1)*bat_len_per_k]

            # fgsm_v1.1
            # X_attack = X[0 : bat_len_per_k,:,:,:]
            # y_attack = y[0 : bat_len_per_k]

            delta = attack(**param).generate(model_k, X_attack, y_attack)

        X_bat[i*bat_len_per_k : (i+1)*bat_len_per_k,:,:,:] = X[i*bat_len_per_k : (i+1)*bat_len_per_k,:,:,:] + delta

    return X_bat, y

def train_bat_fgsm(bat_k, bat_step, j_dir, itr, X, y, model, model_k_list, opt, device):
    model.train()
    X, y = X.to(device), y.to(device)

    X_bat, y_bat = bat_init("fgsm", X, y, bat_k, bat_step, j_dir, itr, model, model_k_list, device)

    yp = model(X_bat)
    loss = nn.CrossEntropyLoss()(yp, y_bat)

    opt.zero_grad()
    loss.backward()
    opt.step()

    batch_correct = (yp.argmax(dim=1) == y_bat).sum().item()
    batch_acc = batch_correct / X_bat.shape[0]

    return batch_acc, loss.item()

def train_bat_pgd(bat_k, bat_step, j_dir, itr, X, y, model, model_k_list, opt, device):
    model.train()
    X, y = X.to(device), y.to(device)

    X_bat, y_bat = bat_init("pgd", X, y, bat_k, bat_step, j_dir, itr, model, model_k_list, device)

    yp = model(X_bat)
    loss = nn.CrossEntropyLoss()(yp, y_bat)

    opt.zero_grad()
    loss.backward()
    opt.step()

    batch_correct = (yp.argmax(dim=1) == y_bat).sum().item()
    batch_acc = batch_correct / X_bat.shape[0]

    return batch_acc, loss.item()

def train_standard(X, y, model, opt, device):
    model.train()
    X, y = X.to(device), y.to(device)

    yp = model(X)
    loss = nn.CrossEntropyLoss()(yp, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    batch_correct = (yp.argmax(dim=1) == y).sum().item()
    batch_acc = batch_correct / X.shape[0]

    return batch_acc, loss.item()

def train_pgd(X, y, model, opt, device):
    model.train()
    X, y = X.to(device), y.to(device)

    attack = pgd_rand
    param = {'ord': np.inf,
             'epsilon': 8./255.,
             'alpha': 2./255.,
             'num_iter': 10,
             'restarts': 1}
    with ctx_noparamgrad_and_eval(model):
        delta = attack(**param).generate(model, X, y)

    yp = model(X+delta)
    loss = nn.CrossEntropyLoss()(yp, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    batch_correct = (yp.argmax(dim=1) == y).sum().item()
    batch_acc = batch_correct / X.shape[0]

    return batch_acc, loss.item()

def train_fgsm(X, y, model, opt, device):
    model.train()
    X, y = X.to(device), y.to(device)

    attack = fgsm
    param = {'ord': np.inf,
             'epsilon': 8./255.}
    with ctx_noparamgrad_and_eval(model):
        delta = attack(**param).generate(model, X, y)

    yp = model(X+delta)
    loss = nn.CrossEntropyLoss()(yp, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    batch_correct = (yp.argmax(dim=1) == y).sum().item()
    batch_acc = batch_correct / X.shape[0]

    return batch_acc, loss.item()

def train_ensemble_pgd(X, y, model_k_list, opt, device):

    model_k_list[0].train()
    model_k_list[1].train()
    

    #check if size x is even
    batch_size = len(X)
    if batch_size % 2 != 0:
        print("batch size is odd number, drop the first input to make it even!")
        X, y = X[1:,:,:,:].to(device), y[1:].to(device)
        batch_size -= 1
    else:
        X, y = X.to(device), y.to(device)
    bs_half = int(batch_size/2)

    # X_0, y_0 = X[:bs_half,:,:,:], y[:bs_half]
    # X_1, y_1 = X[-bs_half:,:,:,:], y[-bs_half:]

    # # sanity check
    # if len(X_0) != bs_half or len(X_1) != batch_size/2:
    #     print(len(X_0), len(X_1))
    #     ValueError("size wrong...")

    attack = pgd_rand
    param = {'ord': np.inf,
             'epsilon': 8./255.,
             'alpha': 2./255.,
             'num_iter': 10,
             'restarts': 1}

    with ctx_noparamgrad_and_eval(model_k_list[0]):
        delta_0 = attack(**param).generate(model_k_list[0], X[:bs_half,:,:,:], y[:bs_half])
    with ctx_noparamgrad_and_eval(model_k_list[1]):
        delta_1 = attack(**param).generate(model_k_list[1], X[-bs_half:,:,:,:], y[-bs_half:])

    delta = torch.cat([delta_0, delta_1], dim = 0)

    yp_0 = model_k_list[0](X+delta)
    loss_0 = nn.CrossEntropyLoss()(yp_0, y)

    opt[0].zero_grad()
    loss_0.backward()
    opt[0].step()

    yp_1 = model_k_list[1](X+delta)
    loss_1 = nn.CrossEntropyLoss()(yp_1, y)

    opt[1].zero_grad()
    loss_1.backward()
    opt[1].step()

    batch_correct = (yp_0.argmax(dim=1) == y).sum().item()
    batch_acc = batch_correct / X.shape[0]

    return batch_acc, loss_0.item()