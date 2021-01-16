import os
import sys
import logging

import torch
import numpy as np

from models import c11
from src.attacks import pgd_rand
from src.train import train_standard, train_pgd, train_fgsm
from src.train import train_bat_fgsm, train_bat_pgd
from src.evaluation import test_clean, test_adv, test_AutoAttack
from src.args import get_args
from src.utils_dataset import load_dataset
from src.utils_log import metaLogger, rotateCheckpoint
from src.utils_general import seed_everything, get_model, get_optim

import torch
import torch.nn as nn
from tqdm import trange

def train(args, logger, X, y, model, opt, itr, model_k_list, device):
    """perform one epoch of training."""
    if args.method == "standard":
        train_log = train_standard(X, y, model, opt, device)

    elif args.method == "pgd":
        train_log = train_pgd(X, y, model, opt, device)

    elif args.method == "fgsm":
        train_log = train_fgsm(X, y, model, opt, device)

    elif args.method == "bat_fgsm":
        train_log = train_bat_fgsm(args.bat_k, 
                                  args.bat_step, 
                                  args.j_dir,
                                  itr, 
                                  X, 
                                  y, 
                                  model,
                                  model_k_list,
                                  opt, 
                                  device)

    elif args.method == "bat_pgd":
        train_log = train_bat_pgd(args.bat_k, 
                                  args.bat_step, 
                                  args.j_dir,
                                  itr, 
                                  X, 
                                  y, 
                                  model,
                                  model_k_list,
                                  opt, 
                                  device)

    else:
        raise  NotImplementedError("Training method not implemented!")

    logger.add_scalar("train/acc_itr", train_log[0], itr+1)
    logger.add_scalar("train/loss_itr", train_log[1], itr+1)

    return train_log

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    attack_param = {"ord":np.inf, "epsilon": 8./255., "alpha":2./255., "num_iter": 20, "restart": 1}

    args = get_args()
    logger = metaLogger(args)
    logging.basicConfig(
        filename=args.j_dir+ "/log/log.txt",
        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    seed_everything(args.seed)
    train_loader, test_loader = load_dataset(args.dataset, args.batch_size)

    model = get_model(args, device)

    if "bat" in args.method:
        if args.batch_size % args.bat_k != 0:
            raise ValueError("mini-batch size must be divisible by bat_k!")

    model_k_list = []
    # print(len(model_list),model_list)

    opt, lr_scheduler = get_optim(model, args)
    ckpt_epoch = 0
    ckpt_itr = 0
    
    ckpt_dir = args.j_dir+"/"+str(args.j_id)+"/"
    ckpt_location = os.path.join(ckpt_dir, "custome_ckpt_"+logger.ckpt_status+".pth")
    if os.path.exists(ckpt_location):
        ckpt = torch.load(ckpt_location)
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        ckpt_epoch = ckpt["epoch"]
        ckpt_itr = ckpt["itr"]
        if lr_scheduler:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

        if "bat" in args.method:
            for i in range(args.bat_k):
                model_k = get_model(args, device)
                model_k_list.append(model_k)
                model_k_list[i].load_state_dict(ckpt["model_k_list_state_dict"][i])
        print("LOADED CHECKPOINT")



    for _epoch in range(ckpt_epoch, args.epoch):
        # train_log = train(args, _epoch, logger, train_loader, model, opt, device)

        with trange(len(train_loader)) as t:
            for X, y in train_loader:

                if "bat" in args.method and X.shape[0] != args.batch_size:
                    t.update()
                    continue

                train_log = train(args, logger, X, y, model, opt, ckpt_itr, model_k_list, device)

                t.set_postfix(loss=train_log[1],
                              acc='{0:.2f}%'.format(train_log[0]*100))
                t.update()

                ckpt_itr += 1

                # The following code is commented since we are now saving the model 
                # in GPU directly. 
                # if "bat" in args.method:
                #     if ckpt_itr == 1 or (ckpt_itr-1) % (args.bat_step) == 0:
                        
                #         # torch.save(model.state_dict(), args.j_dir+"/model/model_"+str(ckpt_itr)+".pt")

                #         if ckpt_itr > (args.bat_k*args.bat_step):
                #             rotateCheckpoint(ckpt_dir, "custome_ckpt", model, opt, _epoch, ckpt_itr, lr_scheduler)
                #             os.remove(args.j_dir+"/model/model_"+str(ckpt_itr-args.bat_k*args.bat_step)+".pt")
                # print(os.listdir(args.j_dir+"/model/"))
                if "bat" in args.method:
                    if ckpt_itr <= args.bat_k*args.bat_step:
                            if ckpt_itr == 1 or (ckpt_itr-1) % (args.bat_step) == 0:
                                model_k_list.append(model)
                                
                    elif (ckpt_itr-1) % (args.bat_step) == 0:
                        model_k_list.pop(0)
                        model_k_list.append(model)

        test_log = test_clean(test_loader, model, device)
        adv_log = test_adv(test_loader, model, pgd_rand, attack_param, device)
        AA_acc = test_AutoAttack(test_loader, model, 100, device)

        logger.add_scalar("pgd20/acc", adv_log[0], _epoch+1)
        logger.add_scalar("pgd20/loss", adv_log[1], _epoch+1)
        logger.add_scalar("test/acc", test_log[0], _epoch+1)
        logger.add_scalar("test/loss", test_log[1], _epoch+1)
        logger.add_scalar("autoattack/acc", AA_acc, _epoch+1)
        logging.info(
            "Test set: Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                loss=test_log[1],
                acc=test_log[0]))
        logging.info(
            "PGD20: Loss: {loss:.6f}\t"
            "Accuracy: {acc:.2f}".format(
                loss=adv_log[1],
                acc=adv_log[0]))

        if lr_scheduler:
            lr_scheduler.step()

        if (_epoch+1) % args.ckpt_freq == 0:
            rotateCheckpoint(args, ckpt_dir, "custome_ckpt", model, opt, _epoch, ckpt_itr, lr_scheduler, model_k_list)

        if (_epoch+1) == args.epoch:
            AA_acc = test_AutoAttack(test_loader, model, 1000, device)
            logger.add_scalar("autoattack/final_acc", AA_acc, _epoch+1)

        logger.save_log()
    logger.close()
    torch.save(model.state_dict(), args.j_dir+"/model/model.pt")

if __name__ == "__main__":
    main()
