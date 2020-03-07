import json

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from utils.util import AverageMeter, save, accuracy, save_best_layer
from utils.util import check_tensor_in_list, get_best_structure, save_structure_sort, save_candidate_table
from config import CONFIG


class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, w_scheduler, logger, writer, device):
        self.top1 = AverageMeter()
        self.top3 = AverageMeter()
        self.losses = AverageMeter()
        self.losses_lat = AverageMeter()
        self.losses_ce = AverageMeter()

        self.logger = logger
        self.writer = writer
        self.device = device

        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.w_scheduler = w_scheduler

        self.layers_structure = []
        self.dataset = CONFIG["dataloading"]["dataset"]

        self.cnt_epochs = CONFIG["train_settings"]["cnt_epochs"]
        self.meta_epochs = CONFIG["train_settings"]["meta_epochs"]
        self.warmup_epochs = CONFIG["train_settings"]["warmup_epochs"]
        self.print_freq = CONFIG["train_settings"]["print_freq"]
        self.path_to_save_model = CONFIG["train_settings"]["path_to_save_model"]
        self.path_to_save_structure = CONFIG["train_settings"]["path_to_save_structure"]
        self.path_to_save_acc = CONFIG["train_settings"]["path_to_save_acc"]
        self.path_to_candidate_table = CONFIG["train_settings"]["path_to_candidate_table"]
        self.ngpu = CONFIG["ngpu"]
        self.max_epochs = 0

        self.acc_record = {}
        self.candidate_table = []
        self.layer = 0

        with open(self.path_to_save_acc, "w") as f:
            json.dump(self.acc_record, f)


    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):
        best_top1 = 0.0
        for epoch in range(self.warmup_epochs):
            self.logger.info("Firstly, start to warmup training for epoch %d" % (epoch))
            self._warmup_training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_warmup_step_")
            #self._validate(model, test_loader, 0)
            self.w_scheduler.step()

        model = model.module if self.ngpu > 1 else model
        model.stage_init()
        model.to(self.device)

        if (self.device.type == "cuda") and (self.ngpu > 1):
            model = nn.DataParallel(model, list(range(self.ngpu)))
        
        for layer in range(CONFIG["train_settings"][self.dataset]["maximum_layer"]):
            model_macs = model.module.get_model_info(self.device) if self.ngpu > 1 else model.get_model_info(self.device)
            print("Entire model macs {}".format(model_macs))

            candidate_block_num = model.module.get_block_num() if self.ngpu > 1 else model.get_block_num()
            self.max_epochs = self.cnt_epochs + self.meta_epochs

            self.layer = layer

            self.structure_num = model.module.get_structure_nums() if self.ngpu > 1 else model.get_structure_nums()
            self.voting = [[0] for i in range(self.structure_num)]

            model.module.transfer_weight() if self.ngpu > 1 else model.transfer_weight()
            self._optimizer_init(model)

            for epoch in range(self.meta_epochs):
                self.writer.add_scalar("learning_rate/weights", self.w_optimizer.param_groups[0]["lr"], epoch)
                self.logger.info("Firstly, start to train weights for epoch %d" % (epoch))
                self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")

            for epoch in range(self.meta_epochs, self.max_epochs):
                self.logger.info("Start to train weight for epoch %d" % (epoch))
                self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")

                structure_sort, structure_accuracy = get_best_structure(model, self.criterion, train_thetas_loader, self.device)
                    
                for index in range(len(self.voting)):
                    self.voting[index].append(structure_accuracy[index])

            self.voting = np.array(self.voting)
            voting = torch.Tensor([0 for i in range(self.structure_num)])
            for index in range(len(self.voting)):
                voting[index] = self.voting[index].max()
            print(voting)
            print(voting.argmax())

            model = model.module if self.ngpu > 1 else model

            #top1_avg = self._validate(model, test_loader, layer)
            #if best_top1 < top1_avg:
            #    best_top1 = top1_avg
            #    self.logger.info("Best top1 acc by now. Save model")
            #    save(model, self.path_to_save_model)
            
            ops_names, macs_list = model.get_candidate_info()
            candidate_info = {
                        "ops_names" : ops_names,
                        "accuracy" : voting.numpy().tolist(),
                        "macs" : macs_list.tolist()
                    }
            self.candidate_table.append(candidate_info)

            layers_info = model.add_stage_module(voting.argmax(), voting)
            self._save_layer_structure(layers_info)

            if layer+1 == CONFIG["train_settings"][self.dataset]["maximum_layer"]:
                save_candidate_table(self.candidate_table, self.path_to_candidate_table)
                break

            model.add_search_module(layers_info)
            
            model.to(self.device)
            if (self.device.type == "cuda") and (self.ngpu > 1):
                model = nn.DataParallel(model, list(range(self.ngpu)))

            self._optimizer_init(model)

    def _save_layer_structure(self, layers_info):
        layer_name, input_depth, out_depth, _ = layers_info

        self.layers_structure.append({
                                        "name" : layer_name,
                                        "input_depth" : input_depth,
                                        "out_depth" : out_depth
                                     })

        save_best_layer(self.layers_structure, self.path_to_save_structure)

    def _split_block_info(self, layer_name):
        block_info = layer_name.split("_")
        print(block_info)
        e = int(block_info[4][1])
        s = int(block_info[2][1])
        k = int(block_info[1][1])

        return e, s, k

    def _optimizer_init(self, model, lr=None):
        del self.w_optimizer
        del self.w_scheduler

        self.w_optimizer = torch.optim.SGD(params=model.parameters(),
                                           lr=CONFIG["optimizer"]["lr"],
                                           momentum=CONFIG["optimizer"]["w_momentum"],
                                           weight_decay=CONFIG["optimizer"]["w_weight_decay"])

        self.w_scheduler = torch.optim.lr_scheduler.StepLR(self.w_optimizer, step_size=3, gamma=0.2, last_epoch=-1)


    def _training_step(self, model, loader, optimizer, epoch, info_for_logger=""):
        model.train()
        start_time = time.time()
        structure_num = model.module.get_structure_nums() if self.ngpu > 1 else model.get_structure_nums()
        structure_order = np.array([i for i in range(structure_num)])
        np.random.shuffle(structure_order)
        
        structure_index = 0

        meta_status = model.module.stages_to_search.meta_learning if self.ngpu > 1 else model.stages_to_search.meta_learning
        
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            N = X.shape[0]

            outs = model(X, structure_order[structure_index])
            structure_index += 1
            loss = self.criterion(outs, y)
            self.losses_ce.update(loss.item(), N)
            loss.backward()
            if step % structure_num == 0 or meta_status:
                optimizer.step()
                optimizer.zero_grad()
                structure_index = 0
                np.random.shuffle(structure_order)

            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")

        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train="train")
        for avg in [self.top1, self.top3, self.losses_ce, self.losses_lat, self.losses]:
            avg.reset()

    def _warmup_training_step(self, model, loader, optimizer, epoch, info_for_logger=""):
        model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            N = X.shape[0]

            optimizer.zero_grad()
            outs = model(X, 0)
            loss = self.criterion(outs, y)
            loss.backward()
            optimizer.step()

            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")

        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train="train")
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()

    def _validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]

                outs = model(X, 0)
                loss = self.criterion(outs, y)
                self.losses_ce.update(loss.item(), N)

                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")

        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train="val")
        for avg in [self.top1, self.top3, self.losses, self.losses_lat, self.losses_ce]:
            avg.reset()
        return top1_avg

    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=""):
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_loss"+info_for_logger, self.losses.get_avg(), epoch)
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_top1"+info_for_logger, self.top1.get_avg(), epoch)
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_top3"+info_for_logger, self.top3.get_avg(), epoch)
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_losses_lat"+info_for_logger, self.losses_lat.get_avg(), epoch)
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_losses_cs"+info_for_logger, self.losses_ce.get_avg(), epoch)

        top1_avg = self.top1.get_avg()
        self.logger.info(info_for_logger+val_or_train+":[{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(epoch+1, self.max_epochs, top1_avg, time.time()-start_time))

    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top3.update(prec3.item(), N)

        if (step > 1 and step % self.print_freq==0) or step == len_loader -1 :
            self._save_acc_record(self.top1.get_avg())
            self.logger.info(val_or_train+
                    ":[{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f}"
                    "Prec@(1, 3) ({:.1%}, {:.1%}), ce_loss {:.3f}, lat_loss {:.3f}".format(
                        epoch+1, self.max_epochs, step, len_loader-1, self.losses.get_avg(),
                        self.top1.get_avg(), self.top3.get_avg(), self.losses_ce.get_avg(), self.losses_lat.get_avg()
                        ))
    def _save_acc_record(self, top1):
        acc_record = {}
        with open(self.path_to_save_acc) as f:
            acc_record = json.load(f)
        
        if str(self.layer) not in acc_record:
            acc_record[str(self.layer)] = [top1]
        else:
            acc_record[str(self.layer)].append(top1)

        with open(self.path_to_save_acc, "w") as f:
            json.dump(acc_record, f)



class TrainerNet:
    def __init__(self, criterion, optimizer, scheduler, logger, writer, device):
        self.top1 = AverageMeter()
        self.top3 = AverageMeter()
        self.losses = AverageMeter()

        self.logger = logger
        self.writer = writer
        self.block_acc = []
        self.device = device

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_epochs = CONFIG["train_settings"]["train_epochs"]
        self.print_freq = CONFIG["train_settings"]["print_freq"]
        self.path_to_save_model = CONFIG["train_settings"]["path_to_save_model"]

    def get_block_acc(self):
        return self.block_acc

    def train_loop(self, train_loader, val_loader, test_loader, model):
        best_top1 = 0.0
        for epoch in range(self.train_epochs):
            self.writer.add_scalar("learning_rate/weights", self.optimizer.param_groups[0]["lr"], epoch)

            self.logger.info("Start to train for epoch %d" % (epoch))
            self._training_step(model, train_loader, self.optimizer, epoch, info_for_logger="_train_step_")
            if val_loader is not None:
                self._training_step(model, val_loader, self.optimizer, epoch, info_for_logger="_train_step_")

            top1_avg = self._validate(model, test_loader, epoch)

            self.block_acc.append(top1_avg)

            if best_top1 < top1_avg:
                best_top1 = top1_avg
                self.logger.info("Best top1 acc by now. Save model")
                save(model, self.path_to_save_model)

    def _training_step(self, model, loader, optimizer, epoch, info_for_logger=""):
        model.train()
        start_time = time.time()

        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            N = X.shape[0]

            optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)
            loss.backward()
            optimizer.step()
            self.scheduler.step()

            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")

        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train="train")
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()

    def _validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]

                outs = model(X)
                loss = self.criterion(outs, y)

                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")

        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train="val")
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()
        return top1_avg

    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=""):
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_loss"+info_for_logger, self.losses.get_avg(), epoch)
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_top1"+info_for_logger, self.top1.get_avg(), epoch)
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_top3"+info_for_logger, self.top3.get_avg(), epoch)
        self.writer.add_scalar("train_vs_val/"+val_or_train+"_losses"+info_for_logger, self.losses.get_avg(), epoch)

        top1_avg = self.top1.get_avg()
        self.logger.info(info_for_logger+val_or_train+":[{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(epoch+1, self.train_epochs, top1_avg, time.time()-start_time))

    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top3.update(prec3.item(), N)

        if (step > 1 and step % self.print_freq==0) or step == len_loader -1 :
            self.logger.info(val_or_train+
                    ":[{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f}"
                    "Prec@(1, 3) ({:.1%}, {:.1%}), loss {:.3f}".format(
                        epoch+1, self.train_epochs, step, len_loader-1, self.losses.get_avg(),
                        self.top1.get_avg(), self.top3.get_avg(), self.losses.get_avg())
                        )
