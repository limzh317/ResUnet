#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 下午8:29
# @Author  : chuyu zhang
# @File    : model.py
# @Software: PyCharm


import os
import sys
import time
import numpy as np
from easydict import EasyDict as edict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose, ColorJitter

import libs.transforms as tf
from libs.utils import evaluate_iou, class_balanced_cross_entropy_loss, \
    adjust_learning_rate, load_checkpoint, save_checkpoint_lite, load_checkpoint_lite

from libs.models import ResUNet2, ResUNet1
from dataloader import ImageLoader
from loss import FocalLoss
import ipdb
import json

sys.path.append(os.path.abspath('./'))


class Model:
    def __init__(self, model_name):
        # parameter
        self.paras = edict()
        self.paras.epoch = 0
        self.paras.lr = 1e-4
        self.paras.momentum = 0.99
        self.paras.weight_decay = 1e-4

        self.paras.num_epochs = 20

        # data
        self.paras.batch_size = 16
        self.train_parent_loader = None
        self.val_parent_loader = None
        self.test_parent_loader = None

        # loss
        self.paras.best_val_dice = 0
        self.model_name = model_name
        # TODO modify while using node>36
        self.data_root_dir = '/root/Project/cellproject'
        self.save_root_dir = '/root/Project/cellproject/ResUnet/'

        self.num_workers = 16
        self.device = torch.device("cuda")
        self.resume = False
        self.checkpoint_dir = None
        self.model = None

        self.opt = None
        self.criterion = FocalLoss()
        self.adjust_lr = adjust_learning_rate

        self.all_label = True
        self.loss_exclude = False
        self.cross_entropy = False
        self.loss_epoch = True

    def init_data_loader(self):
        train_transform = Compose([tf.RandomHorizontalFlip(),
                                   tf.ScaleNRotate(rots=(-15, 15), scales=(.9, 1.1)),
                                   # tf.Gamma(gamma = 0.8),
                                   tf.ToTensor(),
                                   tf.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

        val_transform = Compose([tf.ToTensor(),
                                 tf.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
        """
        train_image_transform = transforms.Compose([transforms.CenterCrop(480),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        train_label_transform = transforms.Compose([transforms.CenterCrop(480),
                                                    transforms.ToTensor()])

        val_image_transform = transforms.Compose([transforms.CenterCrop(480),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        val_label_transform = transforms.Compose([transforms.CenterCrop(480),
                                                  transforms.ToTensor()])
        """

        train_dataset = ImageLoader(mode='train', data_root_dir=self.data_root_dir,
                                    transforms=train_transform)

        self.train_parent_loader = DataLoader(train_dataset, batch_size=self.paras["batch_size"],
                                              shuffle=True, num_workers=self.num_workers)

        val_dataset = ImageLoader(mode='val', data_root_dir=self.data_root_dir,
                                  transforms=val_transform)

        self.val_parent_loader = DataLoader(val_dataset, batch_size=self.paras["batch_size"],
                                            shuffle=True, num_workers=self.num_workers)

        test_dataset = ImageLoader(mode='test', data_root_dir=self.data_root_dir,
                                   transforms=val_transform)

        self.test_parent_loader = DataLoader(test_dataset, batch_size=self.paras["batch_size"],
                                             shuffle=True, num_workers=self.num_workers)

    def record(self, result):
        if os.path.exists("log.json"):
            with open("log.json", "r") as f:
                record = json.load(f)
            record[len(list(record.keys()))] = {**self.paras, **result}
        else:
            record = {1: {**self.paras, **result}}

        with open("log.json", "w") as f:
            json.dump(record, f)

    def init_net(self):
        """
        pretrain model or train model from scratch
        :return:
        """
        self.checkpoint_dir = os.path.join(self.save_root_dir, 'checkpoints', self.model_name)

        # if resume is true and checkpoint_dir exist, retrain from the checkpoint.
        if self.resume and os.path.exists(self.checkpoint_dir):
            print('Resume from {}'.format(self.checkpoint_dir))

            state_dict, optim_dict = load_checkpoint_lite(self.checkpoint_dir)

            self.model = ResUNet2()
            model_state = self.model.state_dict()

            model_pretrained = {}
            for k, v in state_dict.items():
                # ipdb.set_trace()
                k = k.replace('module.', '')
                if k in model_state.keys():
                    print(k)
                    model_pretrained[k] = v
            model_state.update(model_pretrained)

            self.model.load_state_dict(state_dict=model_state)
        else:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            self.model = ResUNet2().to(self.device)

        self.model = torch.nn.DataParallel(self.model)

    @staticmethod
    def class_balance_weight(num_ex, num_mito, num_mem, num_nu, num_gr):
        # rate = (num_ex + num_mem + num_nu) / (num_mito + num_gr)
        weights = [10000/num_ex, 10000/num_mito, 10000/num_mem, 10000/num_nu, 10000/num_gr]
        return weights

    def train(self, epoch):
        loss_epoch = []
        iou_epoch = []
        acc_epoch = []
        dice_coeff_epoch = []
        dice_coeff_all_epoch = []

        num_iter = len(self.train_parent_loader)
        for batch_idx, sample in enumerate(self.train_parent_loader):
            start = time.time()
            img, labels = sample['image'], sample['label']
            img, labels = img.float().to(self.device), labels.float().to(self.device)
            outputs = self.model(img)
            loss_batch = 0
            loss_middle = [0] * len(outputs)

            for idx in range(len(outputs)):
                # currently, to keep the code simple, ignore the class balance
                """
                pixel_num = float(self.paras["batch_size"] * labels.shape[1] * labels.shape[2])
                print(labels.shape)
                print(outputs[idx].shape)
                if self.all_label:
                    weights = self.class_balance_weight(float(torch.sum(labels == 0)),
                                                        float(torch.sum(labels == 1)),
                                                        float(torch.sum(labels == 2)),
                                                        float(torch.sum(labels == 3)),
                                                        float(torch.sum(labels == 4)))
                else:
                    weights = [pixel_num / float(torch.sum(labels == 0)), pixel_num / float(torch.sum(labels == 1)),
                               pixel_num / float(torch.sum(labels == 2))]

                weights_norm = torch.FloatTensor(weights) / torch.FloatTensor(weights).sum()

                class_weights = torch.FloatTensor(weights_norm).to(self.device)
                """

                loss_middle[idx] = self.criterion(F.log_softmax(outputs[idx]), labels.long())

            if self.loss_epoch:
                loss_batch += (1 - epoch / self.paras["num_epochs"]) * sum(loss_middle[:-1]) \
                              / (self.paras["batch_size"] - 1) + loss_middle[-1]
            else:
                loss_batch += loss_middle[-1]

            output = outputs[-1]

            bin_mask = torch.argmax(torch.softmax(output, 1), dim=1)

            iou_all, iou_mean, acc, dice_coeff = evaluate_iou(bin_mask, labels, self.all_label)

            time_cost = time.time() - start

            loss_epoch.append(loss_batch.data.cpu().numpy())
            iou_epoch.append(iou_mean)
            acc_epoch.append(acc)
            dice_coeff_epoch.append(dice_coeff.mean())
            dice_coeff_all_epoch.append(dice_coeff)

            self.opt.zero_grad()
            loss_batch.backward()
            self.opt.step()

            if batch_idx % 8 == 0:
                print("{}! {}:{}/{}, Time:{:.3f}, Loss: {:.4f}, IOU: {:.4f}, Acc: {:.4f}, dice: {:.4f}"
                      .format('Training', epoch, batch_idx, num_iter, time_cost, loss_batch.item(), iou_mean, acc,
                              dice_coeff.mean()))
                print("IOU: {:.4f} {:.4f} {:.4f} {:.4f}".format(iou_all[0], iou_all[1], iou_all[2], iou_all[3]))
                print("DICE: {:.4f} {:.4f} {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1], dice_coeff[2],
                                                                 dice_coeff[3]))

        loss_epoch = np.array(loss_epoch)
        iou_epoch = np.array(iou_epoch)
        acc_epoch = np.array(acc_epoch)
        dice_coeff_epoch = np.array(dice_coeff_epoch)
        dice_coeff_all_epoch = np.array(dice_coeff_all_epoch)

        loss = loss_epoch.mean()
        iou = iou_epoch.mean()
        acc = acc_epoch.mean()
        dice_coeff = dice_coeff_epoch.mean()
        dice_coeff_each = dice_coeff_all_epoch.mean(axis=0)

        return loss, iou, acc, dice_coeff, dice_coeff_each

    def val(self, epoch=None, val_test="val"):
        # TODO: can I merge the val and train to inference to simply the code? model.eva
        if epoch is None:
            epoch = self.paras["batch_size"]
        loss_epoch = []
        iou_epoch = []
        acc_epoch = []
        dice_coeff_epoch = []
        dice_coeff_all_epoch = []

        num_iter = len(self.val_parent_loader)
        if val_test=="val":
            dataloader = self.val_parent_loader
        elif val_test=="test":
            dataloader = self.test_parent_loader
        else:
            raise NotImplementedError("Not implement error")

        with torch.no_grad():
            for batch_idx, sample in enumerate(dataloader):
                start = time.time()
                img, labels = sample['image'], sample['label']
                img, labels = img.float().to(self.device), labels.float().to(self.device)
                loss_batch = 0
                outputs = self.model(img)

                loss_batch_list = [0] * len(outputs)
                for idx in range(len(outputs)):
                    loss_batch_list[idx] = self.criterion(F.log_softmax(outputs[idx]), labels.long())

                if self.loss_epoch:
                    loss_batch += (1 - epoch / self.paras["num_epochs"]) * sum(loss_batch_list[:-1]) / \
                                  (self.paras["batch_size"] - 1) + loss_batch_list[-1]
                else:
                    loss_batch += loss_batch_list[-1]

                time_cost = time.time() - start

                output = outputs[-1]
                bin_mask = torch.argmax(torch.softmax(output, 1), dim=1)

                iou_all, iou_mean, acc, dice_coeff = evaluate_iou(bin_mask, labels, self.all_label)

                loss_epoch.append(loss_batch.data.cpu().numpy())
                iou_epoch.append(iou_mean)
                acc_epoch.append(acc)
                dice_coeff_epoch.append(dice_coeff.mean())
                dice_coeff_all_epoch.append(dice_coeff)

                if batch_idx % 8 == 0:
                    print("{}! {}:{}/{}, Time:{:.3f}, Loss: {:.4f}, IOU: {:.4f}, Acc: {:.4f}, dice: {:.4f}"
                          .format('val', epoch, batch_idx, num_iter, time_cost, loss_batch.item(), iou_mean, acc,
                                  dice_coeff.mean()))
                    print("IOU: {:.4f} {:.4f} {:.4f} {:.4f}".format(iou_all[0], iou_all[1], iou_all[2], iou_all[3]))
                    print("DICE: {:.4f} {:.4f} {:.4f} {:.4f}".format(dice_coeff[0], dice_coeff[1], dice_coeff[2],
                                                                     dice_coeff[3]))

        loss_epoch = np.array(loss_epoch)
        iou_epoch = np.array(iou_epoch)
        acc_epoch = np.array(acc_epoch)
        dice_coeff_epoch = np.array(dice_coeff_epoch)
        dice_coeff_all_epoch = np.array(dice_coeff_all_epoch)

        loss = loss_epoch.mean()
        iou = iou_epoch.mean()
        acc = acc_epoch.mean()
        dice_coeff = dice_coeff_epoch.mean()
        dice_coeff_each = dice_coeff_all_epoch.mean(axis=0)  # Four means

        return loss, iou, acc, dice_coeff, dice_coeff_each

    def main(self):
        print(self.paras)
        self.init_data_loader()
        results = {}
        torch.cuda.synchronize()
        loss_train_epochs = []
        loss_val_epochs = []

        iou_train_epochs = []
        iou_val_epochs = []

        acc_train_epochs = []
        acc_val_epochs = []

        dice_coeff_train_epochs = []
        dice_coeff_val_epochs = []

        dice_coeff_each_train_epochs = []
        dice_coeff_each_val_epochs = []

        self.opt = torch.optim.Adam(self.model.module.encoder.parameters(),
                                    lr=self.paras["lr"],
                                    weight_decay=self.paras["weight_decay"])

        for epoch in range(self.paras["num_epochs"]):
            print("Start training!")
            start = time.time()
            loss_train_epoch, iou_train_epoch, acc_train_epoch, dice_coeff_train_epoch, dice_coeff_each_train_epoch \
                = self.train(epoch=epoch)

            self.adjust_lr(optimizer=self.opt, init_lr=self.paras["lr"], epoch=epoch, step=30)
            results["train"] = [loss_train_epoch, iou_train_epoch, acc_train_epoch, dice_coeff_train_epoch,
                                dice_coeff_each_train_epoch]
            time_cost = time.time() - start
            loss_train_epochs.append(loss_train_epoch)
            iou_train_epochs.append(iou_train_epoch)
            acc_train_epochs.append(acc_train_epoch)
            dice_coeff_train_epochs.append(dice_coeff_train_epoch)
            dice_coeff_each_train_epochs.append(dice_coeff_each_train_epoch)

            print('epoch:{},epoch_time:{:.4f},loss_train:{:.4f},iou_train:{:.4f},acc_train:{:.4f},dice_train:{:.4f}'
                  .format(epoch, time_cost, loss_train_epoch, iou_train_epoch, acc_train_epoch, dice_coeff_train_epoch))

            if (epoch+1)%5==0:
                print("Val:")
                loss_val_epoch, iou_val_epoch, acc_val_epoch, dice_coeff_val_epoch, dice_coeff_each_val_epoch \
                    = self.val(epoch=epoch)
                loss_val_epochs.append(loss_val_epoch)
                iou_val_epochs.append(iou_val_epoch)
                acc_val_epochs.append(acc_val_epoch)
                dice_coeff_val_epochs.append(dice_coeff_val_epoch)
                dice_coeff_each_val_epochs.append(dice_coeff_each_val_epoch)
                results['val'] = [loss_val_epoch, iou_val_epoch, acc_val_epoch, dice_coeff_val_epoch,
                                  dice_coeff_each_val_epoch]
                print('epoch:{}, loss_val:{:.4f}, iou_val:{:.4f}, acc_val:{:.4f}, dice_val:{:.4f}'.format(
                          epoch, loss_val_epoch, iou_val_epoch, acc_val_epoch, dice_coeff_val_epoch))

                if dice_coeff_val_epoch > self.paras.best_val_dice:
                    print("Saving checkpoints.")
                    self.paras.best_val_dice = dice_coeff_val_epoch

                    save_checkpoint_lite(checkpoint_dir=self.checkpoint_dir,
                                         model=self.model,
                                         optim=self.opt,
                                         paras=self.paras)

        print("finish training, start test:")
        loss_test_epoch, iou_test_epoch, acc_test_epoch, dice_coeff_test_epoch, dice_coeff_each_test_epoch \
            = self.val(val_test="test")
        results["test"] = [loss_test_epoch, iou_test_epoch, acc_test_epoch, dice_coeff_test_epoch,
                           dice_coeff_each_test_epoch]
        self.record(results)


if __name__ == '__main__':
    model = Model(model_name='1029_resnet101_lr-4_bs16_resnet101_adam_decay30_ck')
    model.init_net()
    model.main()
