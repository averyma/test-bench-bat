import time
import os
from collections import defaultdict
import torch
from src.save_function import checkpoint_save
from torch.utils.tensorboard import SummaryWriter

import wandb

def rotateCheckpoint(args, ckpt_dir, ckpt_name, model, opt, epoch, ckpt_itr, lr_scheduler, model_k_list, ckpt_max_robust_acc):
    ckpt_curr = os.path.join(ckpt_dir, ckpt_name+"_curr.pth")
    ckpt_prev = os.path.join(ckpt_dir, ckpt_name+"_prev.pth")

    # no existing ckpt
    if not (os.path.exists(ckpt_curr) or os.path.exists(ckpt_prev)):
        pass
    elif os.path.exists(ckpt_curr):
        # overwrite ckpt_prev with ckpt_curr
        cmd = "cp -r {} {}".format(ckpt_curr, ckpt_prev)
        os.system(cmd)

    if "bat" in args.method:    
        saveCheckpoint_bat(ckpt_dir,
                           ckpt_name+"_curr.pth",
                           model,
                           opt,
                           epoch,
                           ckpt_itr,
                           model_k_list,
                           args.bat_k,
                           ckpt_max_robust_acc,
                           lr_scheduler)
    elif "ensemble" in args.method:
        saveCheckpoint_ensemble(ckpt_dir,
                               ckpt_name+"_curr.pth",
                               model,
                               opt,
                               epoch,
                               ckpt_itr,
                               model_k_list,
                               args.bat_k,
                               ckpt_max_robust_acc,
                               lr_scheduler)
    else:
        saveCheckpoint(ckpt_dir,
                       ckpt_name+"_curr.pth",
                       model,
                       opt,
                       epoch,
                       ckpt_itr,
                       ckpt_max_robust_acc,
                       lr_scheduler)

def saveCheckpoint(ckpt_dir, ckpt_name, model, opt, epoch, ckpt_itr, ckpt_max_robust_acc, lr_scheduler):
    
    lr_state_dict = lr_scheduler.state_dict() if lr_scheduler else []

    # if lr_scheduler:
    checkpoint_save({"state_dict": model.state_dict(),
                     "optimizer": opt.state_dict(),
                     "epoch": epoch+1,
                     "itr": ckpt_itr, 
                     "lr_scheduler": lr_state_dict,
                     "max_robust_acc": ckpt_max_robust_acc},
                    ckpt_dir, ckpt_name)
    # else:
    #     checkpoint_save({"state_dict": model.state_dict(),
    #                      "optimizer": opt.state_dict(),
    #                      "epoch": epoch+1,
    #                      "itr": ckpt_itr}, ckpt_dir, ckpt_name)

    print("SAVED CHECKPOINT")

def saveCheckpoint_bat(ckpt_dir, ckpt_name, model, opt, epoch, ckpt_itr, model_k_list, bat_k, ckpt_max_robust_acc, lr_scheduler):
    model_k_list_state_dict = []
    lr_state_dict = lr_scheduler.state_dict() if lr_scheduler else []

    # if len(model_k_list) != 0:
    for i in range(len(model_k_list)):
        model_k_list_state_dict.append(model_k_list[i].state_dict()) 

    # if lr_scheduler:
    checkpoint_save({"state_dict": model.state_dict(),
                     "optimizer": opt.state_dict(),
                     "epoch": epoch+1,
                     "itr": ckpt_itr, 
                     "lr_scheduler": lr_state_dict,
                     "model_k_list_state_dict": model_k_list_state_dict,
                     "max_robust_acc": ckpt_max_robust_acc},
                    ckpt_dir, ckpt_name)
    # else:
    #     checkpoint_save({"state_dict": model.state_dict(),
    #                      "optimizer": opt.state_dict(),
    #                      "epoch": epoch+1,
    #                      "itr": ckpt_itr,
    #                      "model_k_list_state_dict": model_k_list_state_dict,
    #                      "max_robust_acc": ckpt_max_robust_acc},
    #                     ckpt_dir, ckpt_name)

    print("SAVED CHECKPOINT")

def saveCheckpoint_ensemble(ckpt_dir, ckpt_name, model, opt, epoch, ckpt_itr, model_k_list, bat_k, ckpt_max_robust_acc, lr_scheduler):
    model_k_list_state_dict = []
    lr_state_dict = [lr_scheduler[0].state_dict(), lr_scheduler[1].state_dict()] if lr_scheduler[0] else []

    for i in range(len(model_k_list)):
        model_k_list_state_dict.append(model_k_list[i].state_dict()) 

    # if lr_scheduler:
    checkpoint_save({"state_dict": model.state_dict(),
                     "optimizer_0": opt[0].state_dict(),
                     "optimizer_1": opt[1].state_dict(),
                     "epoch": epoch+1,
                     "itr": ckpt_itr, 
                     "lr_scheduler_0": lr_state_dict[0],
                     "lr_scheduler_1": lr_state_dict[1],
                     "model_k_list_state_dict": model_k_list_state_dict,
                     "max_robust_acc": ckpt_max_robust_acc},
                    ckpt_dir, ckpt_name)
    # else:
    #     checkpoint_save({"state_dict": model.state_dict(),
    #                      "optimizer_0": opt[0].state_dict(),
    #                      "optimizer_1": opt[1].state_dict(),
    #                      "epoch": epoch+1,
    #                      "itr": ckpt_itr,
    #                      "model_k_list_state_dict": model_k_list_state_dict,
    #                      "max_robust_acc": ckpt_max_robust_acc},
    #                     ckpt_dir, ckpt_name)

    print("SAVED CHECKPOINT")

class metaLogger(object):
    def __init__(self, args, flush_sec=5):
        self.log_path = args.j_dir+"/log/"
        self.tb_path = args.j_dir+"/tb/"
        # self.ckpt_status = "curr"
        self.ckpt_status = self.get_ckpt_status(args.j_dir, args.j_id)
        self.log_dict = self.load_log(self.log_path)
        self.writer = SummaryWriter(log_dir=self.tb_path, flush_secs=flush_sec)
        self.enable_wandb = args.enable_wandb

        if self.enable_wandb:
            self.wandb_log = wandb.init(name=args.j_dir.split("/")[-1],
                                        project=args.wandb_project,
                                        dir=args.j_dir,
                                        id=str(args.j_id),
                                        resume=True)
            self.wandb_log.config.update(args)

    def get_ckpt_status(self, j_dir, j_id):
        
        status = "curr"
        ckpt_dir = j_dir+"/"+str(j_id)+"/"
        ckpt_location_prev = os.path.join(ckpt_dir, "ckpt_prev.pth")
        ckpt_location_curr = os.path.join(ckpt_dir, "ckpt_curr.pth")
        if os.path.exists(ckpt_location_curr) and os.path.exists(ckpt_location_prev):
            try:
                torch.load(ckpt_location_curr)
            except TypeError:
                status = "prev"
        else:
            status = "curr"
        return status
    
    def load_log(self, log_path):
        if self.ckpt_status == "curr":
            try:
                log_dict = torch.load(log_path + "/log_curr.pth")
            except FileNotFoundError:
                log_dict = defaultdict(lambda: list())
            except TypeError:
                log_dict = torch.load(log_path + "/log_prev.pth")
                self.ckpt_status = "prev"
        else:
            log_dict = torch.load(log_path + "/log_prev.pth")
        return log_dict

    def add_scalar(self, name, val, step):
        self.writer.add_scalar(name, val, step)
        # self.log_dict[name] += [(time.time(), int(step), float(val))]
        try:
            self.log_dict[name] += [(time.time(), int(step), float(val))]
        except KeyError:
            self.log_dict[name] = [(time.time(), int(step), float(val))]

        if self.enable_wandb:
            if "_itr" in name:
                self.wandb_log.log({"iteration": step, name: float(val)})
            else:
                self.wandb_log.log({"epoch": step, name: float(val)})

    def add_scalars(self, name, val_dict, step):
        self.writer.add_scalars(name, val_dict, step)
        for key, val in val_dict.items():
            self.log_dict[name+key] += [(time.time(), int(step), float(val))]

    def add_figure(self, name, val, step):
        self.writer.add_figure(name, val, step)
        val.savefig(self.log_path + "/" + name + ".png")

    def save_log(self):
        try:
            os.makedirs(self.log_path)
        except os.error:
            pass

        log_curr = os.path.join(self.log_path, "log_curr.pth")
        log_prev = os.path.join(self.log_path, "log_prev.pth")

        # no existing logs
        if not (os.path.exists(log_curr) or os.path.exists(log_prev)):
            torch.save(dict(self.log_dict), log_curr)

        elif os.path.exists(log_curr):
            # overwrite log_prev with log_curr
            cmd = "cp -r {} {}".format(log_curr, log_prev)
            os.system(cmd)
            torch.save(dict(self.log_dict), log_curr)

    # def log_obj(self, name, val):
        # self.logobj[name] = val

    # def log_objs(self, name, val, step=None):
        # self.logobj[name] += [(time.time(), step, val)]

    # def log_vector(self, name, val, step=None):
        # name += '_v'
        # if step is None:
            # step = len(self.logobj[name])
        # self.logobj[name] += [(time.time(), step, list(val.flatten()))]

    def close(self):
        self.writer.close()
