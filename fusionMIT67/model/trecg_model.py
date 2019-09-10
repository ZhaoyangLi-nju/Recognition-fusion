import os
import time
from collections import OrderedDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision

import util.utils as util
from util.average_meter import AverageMeter
from util.confusion_matrix import plot_confusion_matrix
from . import networks
from .base_model import BaseModel
import copy
import math
from torchvision.models.resnet import resnet18
class TRecgNet(BaseModel):

    def __init__(self, cfg, vis=None, writer=None):
        super(TRecgNet, self).__init__(cfg)

        util.mkdir(self.save_dir)
        assert (self.cfg.WHICH_DIRECTION is not None)
        self.AtoB = self.cfg.WHICH_DIRECTION == 'AtoB'
        self.modality = 'rgb' if self.AtoB else 'depth'
        self.sample_model = None
        self.phase = cfg.PHASE
        self.upsample = not cfg.NO_UPSAMPLE
        self.content_model = None
        self.content_layers = []

        self.writer = writer
        self.vis = vis

        # networks
        self.use_noise = cfg.WHICH_DIRECTION == 'BtoA'
        self.net_classification_1= networks.define_TrecgNet(cfg, self.use_noise, device=self.device)
        self.net_classification_2= networks.define_TrecgNet(cfg, self.use_noise, device=self.device)
        networks.print_network(self.net)


    def set_input(self, data, type='pair'):

        self.source_modal = None
        self.target_modal = None

        if type == 'pair':

            input_A = data['A']
            input_B = data['B']
            self.img_names = data['img_name']
            self.imgs_all.extend(data['img_name'])
            self.real_A = input_A.to(self.device)
            self.real_B = input_B.to(self.device)

            AtoB = self.AtoB
            self.source_modal = self.real_A if AtoB else self.real_B
            self.target_modal = self.real_B if AtoB else self.real_A
            self.source_modal_original = self.source_modal

            self.batch_size = input_A.size(0)

            if 'label' in data.keys():
                self._label = data['label']
                self.label = torch.LongTensor(self._label).to(self.device)
        if type=='single':
            input_A = data[0]
            self.real_A = input_A.to(self.device)
            self.real_B = input_B.to(self.device)

            AtoB = self.AtoB
            self.source_modal = self.real_A if AtoB else self.real_B
            self.target_modal = self.real_B if AtoB else self.real_A
            self.source_modal_original = self.source_modal

            self.batch_size = input_A.size(0)

            self._label = data[1]
            self.label = torch.LongTensor(self._label).to(self.device)

    def build_output_keys(self, gen_img=True, cls=True):

        out_keys = []

        if gen_img:
            out_keys.append('gen_img')

        if cls:
            out_keys.append('cls')

        return out_keys

    # def _optimize(self, cfg, epoch):

    #     # self._forward(epoch)

    #     # if cfg.NITER_START_GAN <= epoch <= cfg.NITER_END_GAN and \
    #     #         'GAN' in cfg.LOSS_TYPES:
    #     #     self.optimizer_D.zero_grad()
    #     #     self.backward_D()
    #     #     self.optimizer_D.step()

    #     self.optimizer_ED.zero_grad()
    #     total_loss = self._construct_TRAIN_G_LOSS(epoch)
    #     total_loss.backward()
    #     self.optimizer_ED.step()

    

    # # encoder-decoder branch
    # def _forward(self, epoch=None):

    #     self.gen = None
    #     self.source_modal_show = None
    #     self.target_modal_show = None
    #     self.cls_loss = None

    #     if self.phase == 'train':

    #         # # use fake data to train
    #         if self.sample_model is not None:
    #             with torch.no_grad():
    #                 out_keys = self.build_output_keys(gen_img=True, cls=False)
    #                 [fake_source], self.loss = self.sample_model(source=self.target_modal,
    #                                                   out_keys=out_keys, return_losses=False)
    #             input_num = len(fake_source)
    #             index = [i for i in range(0, input_num) if np.random.uniform() > 1 - self.cfg.FAKE_DATA_RATE]
    #             for j in index:
    #                 self.source_modal[j, :] = fake_source.data[j, :]
    #             self.fake_image_num += len(index)

    #         if 'CLS' not in self.cfg.LOSS_TYPES or self.cfg.UNLABELED:

    #             out_keys = self.build_output_keys(gen_img=True, cls=False)
    #             [self.gen], self.loss = self.net(source=self.source_modal, target=self.target_modal, out_keys=out_keys,
    #                                              phase=self.phase, content_layers=self.content_layers)

    #         elif self.upsample:
    #             out_keys = self.build_output_keys(gen_img=True, cls=True)
    #             [self.gen, self.cls], self.loss = self.net(source=self.source_modal, target=self.target_modal,
    #                                                        label=self.label, out_keys=out_keys, phase=self.phase,
    #                                                        content_layers=self.content_layers)
    #         else:
    #             out_keys = self.build_output_keys(gen_img=False, cls=True)
    #             [self.cls], self.loss = self.net(source=self.source_modal, target=self.target_modal,
    #                                              label=self.label, out_keys=out_keys, phase=self.phase)

    #     else:

    #         if self.upsample:

    #             if self.cfg.EVALUATE:
    #                 out_keys = self.build_output_keys(gen_img=True, cls=True)
    #                 [self.gen, self.cls], self.loss = self.net(self.source_modal, label=self.label, out_keys=out_keys, phase=self.phase)

    #         else:
    #             out_keys = self.build_output_keys(gen_img=False, cls=True)
    #             [self.cls], self.loss = self.net(source=self.source_modal, label=self.label, out_keys=out_keys, phase=self.phase)

    #     self.source_modal_show = self.source_modal
    #     self.target_modal_show = self.target_modal

    # def _construct_TRAIN_G_LOSS(self, epoch=None):

    #     loss_total = torch.zeros(1)
    #     if self.use_gpu:
    #         loss_total = loss_total.cuda()

    #     if self.gen is not None:
    #         assert (self.gen.size(-1) == self.cfg.FINE_SIZE)

    #     if 'CLS' in self.cfg.LOSS_TYPES:
    #         cls_loss = self.loss['cls_loss'].mean()
    #         loss_total = loss_total + cls_loss

    #         cls_loss = round(cls_loss.item(), 4)
    #         self.loss_meters['TRAIN_CLS_LOSS'].update(cls_loss, self.batch_size)

    #         prec1 = util.accuracy(self.cls.data, self.label, topk=(1,))
    #         self.loss_meters['TRAIN_CLS_ACC'].update(prec1[0].item(), self.batch_size)

    #     # ) content supervised
    #     if self.cfg.NITER_START_CONTENT <= epoch <= self.cfg.NITER_END_CONTENT:

    #         if 'SEMANTIC' in self.cfg.LOSS_TYPES:

    #             content_loss = self.loss['content_loss'].mean()
    #             loss_total = loss_total + content_loss

    #             content_loss = round(content_loss.item(), 4)
    #             self.loss_meters['TRAIN_SEMANTIC_LOSS'].update(content_loss, self.batch_size)

    #     if self.cfg.NITER_START_PIX2PIX <= epoch <= self.cfg.NITER_END_PIX2PIX:

    #         if 'PIX2PIX' in self.cfg.LOSS_TYPES:
    #             pix2pix_loss = self.loss['pix2pix_loss'].mean()
    #             loss_total = loss_total + pix2pix_loss

    #             pix2pix_loss = round(pix2pix_loss.item(), 4)
    #             self.loss_meters['TRAIN_PIXEL_LOSS'].update(pix2pix_loss, self.batch_size)

    #     if self.cfg.NITER_START_GAN <= epoch <= self.cfg.NITER_END_GAN:

    #         if 'GAN' in self.cfg.LOSS_TYPES:
    #             self.forward_D(detach=False)

    #             loss_GAN = self.criterion_GAN(self.pred_fake, self._real) * self.cfg.ALPHA_GAN
    #             loss_total += loss_GAN

    #             loss_G_GAN = round(loss_GAN.item(), 4)
    #             self.loss_meters['TRAIN_G_LOSS'].update(loss_G_GAN, self.batch_size)

    #     # total loss
    #     return loss_total

    # def forward_D(self, detach=False):

    #     size = self.cfg.FINE_SIZE // 2 ** self.net_D.module.d_downsample_num
    #     batch_size = self.source_modal.size(0)
    #     self._real = torch.ones(batch_size, 1, size, size).to(self.device)
    #     self._fake = torch.zeros(batch_size, 1, size, size).to(self.device)

    #     fake_input_D = self.gen
    #     real_input_D = self.target_modal

    #     if detach:
    #         fake_input_D = fake_input_D.detach()
    #     else:
    #         fake_input_D = fake_input_D

    #     self.pred_fake = self.net_D(fake_input_D)
    #     self.pred_real = self.net_D(real_input_D)

    # def backward_D(self):

    #     self.forward_D(detach=True)
    #     loss_D = self._construct_D_loss()
    #     loss_D.backward()

    # def _construct_D_loss(self):

    #     loss_D_total = torch.zeros(1).to(self.device)

    #     # GAN loss
    #     loss_D_fake = self.criterion_GAN(self.pred_fake, self._fake) * self.cfg.ALPHA_GAN
    #     loss_D_real = self.criterion_GAN(self.pred_real, self._real) * self.cfg.ALPHA_GAN

    #     loss_D_total += loss_D_real + loss_D_fake

    #     # GAN Fake
    #     loss_D_fake = round(loss_D_fake.item(), 4)
    #     self.loss_meters['TRAIN_D_FAKE'].update(loss_D_fake)

    #     # GAN Real
    #     loss_D_real = round(loss_D_real.item(), 4)
    #     self.loss_meters['TRAIN_D_REAL'].update(loss_D_real)

    #     # Combined loss
    #     return loss_D_total

    def set_log_data(self, cfg):

        self.loss_meters = defaultdict()
        self.log_keys = [
            'TRAIN_G_LOSS',
            'TRAIN_D_REAL',  # GAN
            'TRAIN_D_FAKE',
            'TRAIN_PIXEL_LOSS',  # pixel-wise
            'VAL_PIXEL_LOSS',
            'TRAIN_SEMANTIC_LOSS',  # semantic
            'TRAIN_CLS_ACC',
            'VAL_CLS_ACC',  # classification
            'TRAIN_CLS_LOSS',
            'VAL_CLS_LOSS',
            'TRAIN_CLS_MEAN_ACC',
            'VAL_CLS_MEAN_ACC'
        ]
        for item in self.log_keys:
            self.loss_meters[item] = AverageMeter()

    def get_current_visuals(self):
        source = util.tensor2im(self.source_modal.data[0])
        gen = util.tensor2im(self.gen.data[0])
        target_img = util.tensor2im(self.target_modal.data[0])

        if self.cfg.IN_CONC:
            source1 = util.tensor2im(self.source_modal.data[0][0:3])
            source2 = util.tensor2im(self.source_modal.data[0][3:])
            vis_images = OrderedDict([('source1', source1), ('gen', gen), ('source2', source2)])
            return vis_images

        vis_images = OrderedDict([('source', source), ('gen', gen), ('target_img', target_img)])

        return vis_images

    def save_checkpoint(self, iter, filename=None):

        if filename is None:
            filename = 'TRecg2Net_{0}_{1}.pth'.format(self.cfg.WHICH_DIRECTION, iter)

        net_state_dict = self.net.state_dict()
        save_state_dict = {}
        for k, v in net_state_dict.items():
            if 'content_model' in k:
                continue
            save_state_dict[k] = v

        state = {
            'iter': iter,
            'state_dict': save_state_dict,
            'optimizer_ED': self.optimizer_ED.state_dict(),
        }
        if 'GAN' in self.cfg.LOSS_TYPES:
            state['state_dict_D'] = self.net_D.state_dict()
            state['optimizer_D'] = self.optimizer_D.state_dict()

        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)

    def load_checkpoint(self, net, checkpoint_path, checkpoint, optimizer=None, data_para=True):

        keep_fc = not self.cfg.NO_FC

        if os.path.isfile(checkpoint_path):

            # load from pix2pix net_G, no cls weights, selected update
            state_dict = net.state_dict()
            state_checkpoint = checkpoint['state_dict']
            if data_para:
                new_state_dict = OrderedDict()
                for k, v in state_checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
                state_checkpoint = new_state_dict

            if keep_fc:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict}
            else:
                pretrained_G = {k: v for k, v in state_checkpoint.items() if k in state_dict and 'fc' not in k}

            state_dict.update(pretrained_G)
            net.load_state_dict(state_dict)

            if self.phase == 'train' and not self.cfg.INIT_EPOCH:
                optimizer.load_state_dict(checkpoint['optimizer_ED'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> !!! No checkpoint found at '{}'".format(self.cfg.RESUME))
            return

    def visualize_generated_images(self, save_dir=None, data_loader=None, vis=None, epoch=None, phase='train'):

        visuals = self.get_current_visuals()
        pred = ''
        target = ''
        if self.cfg.EVALUATE:
            pred_index = self.pred_index[0]
            pred = data_loader.dataset.int_to_class[pred_index]
            target_index = self._label[0].item()
            target = data_loader.dataset.int_to_class[target_index]

        vis.display_current_results(save_dir, epoch=epoch, name=self.img_names[0],
                                    visuals=visuals, category=(pred, target), phase=phase)

    def set_optimizer(self, cfg):

        self.optimizers = []
        # self.optimizer_ED = torch.optim.Adam([{'params': self.net.fc.parameters(), 'lr': cfg.LR}], lr=cfg.LR / 10, betas=(0.5, 0.999))

        self.optimizer_ED = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
        print('optimizer G: ', self.optimizer_ED)
        self.optimizers.append(self.optimizer_ED)

        if 'GAN' in self.cfg.LOSS_TYPES:
            self.optimizer_D = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net_D.parameters()), cfg.LR,
                                               momentum=cfg.MOMENTUM,
                                               weight_decay=cfg.WEIGHT_DECAY)
            print('optimizer D: ', self.optimizer_D)
            self.optimizers.append(self.optimizer_D)

    def evaluate(self, cfg, epoch=None):

        self.phase = 'test'

        # switch to evaluate mode
        self.net.eval()

        self.imgs_all = []
        self.pred_index_all = []
        self.target_index_all = []

        with torch.no_grad():

            print('# Cls val images num = {0}'.format(self.val_image_num))
            # batch_index = int(self.val_image_num / cfg.BATCH_SIZE)
            # random_id = random.randint(0, batch_index)

            for i, data in enumerate(self.val_loader):
                self.set_input(data, self.cfg.DATA_TYPE)

                self._forward()
                self._process_fc()

                if not cfg.INFERENCE:
                    # loss
                    if self.loss['cls_loss'] is not None:
                        cls_loss = self.loss['cls_loss'].mean()
                        self.loss_meters['VAL_CLS_LOSS'].update(round(cls_loss.item(), 4), self.batch_size)

                # accuracy
                prec1 = util.accuracy(self.cls.data, self.label, topk=(1,))
                self.loss_meters['VAL_CLS_ACC'].update(prec1[0].item(), self.batch_size)

        # Mean ACC
        mean_acc = self._cal_mean_acc(cfg=cfg, data_loader=self.val_loader)
        print('mean_acc:', mean_acc)
        return mean_acc

    def _process_fc(self):

        pred, self.pred_index = util.process_output(self.cls.data)

        self.pred_index_all.extend(list(self.pred_index))
        self.target_index_all.extend(list(self._label.numpy()))

    def _cal_mean_acc(self, cfg, data_loader):

        mean_acc = util.mean_acc(np.array(self.target_index_all), np.array(self.pred_index_all),
                                 cfg.NUM_CLASSES,
                                 data_loader.dataset.classes)
        return mean_acc

    def _write_loss(self, phase, global_step):

        loss_types = self.cfg.LOSS_TYPES

        if phase == 'train':

            self.writer.add_scalar('LR', self.optimizer_ED.param_groups[0]['lr'], global_step=global_step)

            if 'CLS' in loss_types:
                self.writer.add_scalar('TRAIN_CLS_LOSS', self.loss_meters['TRAIN_CLS_LOSS'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('TRAIN_CLS_MEAN_ACC', self.loss_meters['TRAIN_CLS_MEAN_ACC'].avg,
                                       global_step=global_step)

            if 'PIX2PIX' in loss_types:
                self.writer.add_scalar('TRAIN_PIXEL_LOSS', self.loss_meters['TRAIN_PIXEL_LOSS'].avg,
                                       global_step=global_step)

            if 'SEMANTIC' in loss_types:
                self.writer.add_scalar('TRAIN_SEMANTIC_LOSS', self.loss_meters['TRAIN_SEMANTIC_LOSS'].avg,
                                       global_step=global_step)

            if 'GAN' in loss_types:
                self.writer.add_scalar('TRAIN_G_LOSS', self.loss_meters['TRAIN_G_LOSS'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('TRAIN_D_REAL', self.loss_meters['TRAIN_D_REAL'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('TRAIN_D_FAKE', self.loss_meters['TRAIN_D_FAKE'].avg,
                                       global_step=global_step)

            self.writer.add_image('Train_1_Source',
                                  torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)
            if self.upsample and self.gen is not None and not self.cfg.NO_VIS:
                self.writer.add_image('Train_2_Gen', torchvision.utils.make_grid(self.gen[:6].clone().cpu().data, 3,
                                                                                 normalize=True),
                                      global_step=global_step)
                self.writer.add_image('Train_3_Target',
                                      torchvision.utils.make_grid(self.target_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)

        if phase == 'test':

            self.writer.add_image('Val_1_Source',
                                  torchvision.utils.make_grid(self.source_modal_show[:6].clone().cpu().data, 3,
                                                              normalize=True), global_step=global_step)

            if self.cfg.EVALUATE and self.cfg.CAL_LOSS:
                self.writer.add_scalar('VAL_CLS_LOSS', self.loss_meters['VAL_CLS_LOSS'].avg,
                                       global_step=global_step)
                self.writer.add_scalar('VAL_CLS_ACC', self.loss_meters['VAL_CLS_ACC'].avg,
                                       global_step=global_step)

                if self.AtoB:
                    mean_acc_win = 'VAL_CLS_MEAN_ACC_A'
                else:
                    mean_acc_win = 'VAL_CLS_MEAN_ACC_B'

                self.writer.add_scalar(mean_acc_win, self.loss_meters['VAL_CLS_MEAN_ACC'].avg,
                                       global_step=global_step)

            if self.upsample and self.gen is not None and not self.cfg.NO_VIS:

                    self.writer.add_image('Val_2_Gen', torchvision.utils.make_grid(self.gen[:6].clone().cpu().data, 3,
                                                                                   normalize=True), global_step=global_step)
                    self.writer.add_image('Val_3_Target',
                                          torchvision.utils.make_grid(self.target_modal_show[:6].clone().cpu().data, 3,
                                                                  normalize=True), global_step=global_step)
