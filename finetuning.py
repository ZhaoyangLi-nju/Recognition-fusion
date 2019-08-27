import os
import random
from collections import Counter
from functools import reduce
from collections import OrderedDict
from collections import defaultdict
import torch
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.parallel
import data.single_dataset as dataset
import util.utils as util
from config.default_config import DefaultConfig
from config.resnet18_sunrgbd_config import RESNET18_SUNRGBD_CONFIG
from data import DataProvider
from model.trecg_model import TRecgNet
from model.networks import define_TrecgNet
from torchvision.models.resnet import *
import copy
import numpy as np
from redefineModel_fusion import ReD_Model
# from redefineModel_test import ReD_Model


def main():
    global cfg
    cfg = DefaultConfig()
    args = {
        'resnet18': RESNET18_SUNRGBD_CONFIG().args(),
    }

    # args for different backbones
    cfg.parse(args['resnet18'])
    cfg.LR=1e-3
    cfg.EPOCHS=200
    # print('cfg.EPOCHS:',cfg.EPOCHS)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    # dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # data
    train_dataset = dataset.SingleDataset(cfg, data_dir=cfg.DATA_DIR_TRAIN, transform=transforms.Compose([
        dataset.Resize((cfg.LOAD_SIZE,cfg.LOAD_SIZE)),
        dataset.RandomCrop((cfg.FINE_SIZE,cfg.FINE_SIZE)),
        dataset.RandomHorizontalFlip(),
        dataset.ToTensor(),
        dataset.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]))

    val_dataset = dataset.SingleDataset(cfg, data_dir=cfg.DATA_DIR_VAL, transform=transforms.Compose([
        dataset.Resize((cfg.LOAD_SIZE,cfg.LOAD_SIZE)),
        dataset.CenterCrop((cfg.FINE_SIZE,cfg.FINE_SIZE)),
        dataset.ToTensor(),
        dataset.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]))
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
    #     num_workers=4, pin_memory=True, sampler=None)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,batch_size=cfg.BATCH_SIZE, shuffle=False,
    #     num_workers=4, pin_memory=True)
    train_loader = DataProvider(cfg, dataset=train_dataset,batch_size=20, shuffle=True)
    val_loader = DataProvider(cfg, dataset=val_dataset, batch_size=5, shuffle=False)

    run_id = random.randint(1, 100000)
    summary_dir='/home/lzy/summary/generateDepth/'+'finetuning_'+str(run_id)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    writer = SummaryWriter(summary_dir)

    if cfg.GENERATE_Depth_DATA:
        print('GENERATE_Depth_DATA model set')
        cfg_generate = copy.deepcopy(cfg)

        cfg_generate.CHECKPOINTS_DIR='/home/lzy/generateDepth/checkpoints/best_AtoB/trecg_AtoB_best.pth'
        cfg_generate.GENERATE_Depth_DATA = False
        cfg_generate.NO_UPSAMPLE = False
        checkpoint = torch.load(cfg_generate.CHECKPOINTS_DIR)
        model = define_TrecgNet(cfg_generate, upsample=True,generate=True)
        load_checkpoint_depth(model,cfg_generate.CHECKPOINTS_DIR, checkpoint, data_para=True)
        generate_model = torch.nn.DataParallel(model).cuda()
        generate_model.eval()

    model=ReD_Model(cfg)
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    best_mean=0
    # optimizer= torch.optim.Adam(policies, cfg.LR,
    #                         weight_decay=cfg.WEIGHT_DECAY)
    optimizer= torch.optim.SGD(policies, cfg.LR,
                            momentum=cfg.MOMENTUM,
                            weight_decay=cfg.WEIGHT_DECAY)

    for epoch in range(cfg.START_EPOCH,cfg.EPOCHS+1 ):
        adjust_learning_rate(optimizer, epoch)
        # mean_acc=validate(val_loader,model,criterion,generate_model,epoch)

        train(train_loader,model,criterion,generate_model,optimizer,epoch,writer)
        mean_acc=validate(val_loader,model,criterion,generate_model,epoch)
        if mean_acc>best_mean:
            best_mean=mean_acc
            print('best mean accuracy:',best_mean)
        else:
        	print('best mean accuracy:',best_mean)
        writer.add_scalar('mean_acc_color', mean_acc, global_step=epoch)
        writer.add_scalar('best_meanacc_color', best_mean, global_step=epoch)
    writer.close()
def train(train_loader,model,criterion,generate_model,optimizer,epoch,writer):
    losses = AverageMeter()
    losses = AverageMeter()
    model.train()
    for i, (input,target) in enumerate(train_loader):
        out_keys=[]
        out_keys.append('gen_img')
        depth_image=generate_model(source=input,out_keys=out_keys)
        depth_image=depth_image.detach()

        target = target.cuda(async=True)
        
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()

        feature=model(input_var,depth_image)

        loss = criterion(feature,target_var)
        # measure accuracy and record loss

            # compute gradient and do SGD step
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if i % 40 == 0:
            print('Epoch: [{0}][{1}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
               epoch, i, loss=losses))

    writer.add_scalar('LR', cfg.LR, global_step=epoch)
    writer.add_scalar('train_loss_color', losses.avg, global_step=epoch)




def validate(val_loader,model,criterion,generate_model,epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to evaluate mode
    model.eval()


    pred_mean=[]
    target_mean=[]

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        out_keys=[]
        out_keys.append('gen_img')
        depth_image=generate_model(source=input,out_keys=out_keys)
        depth_image=depth_image.detach()
        # out_keys = build_output_keys(gen_img=True)
        # # self.set_input(data, self.cfg.DATA_TYPE)
        # depth_image=generate_model(source=input,out_keys=out_keys)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target).cuda()

        # compute output
        feature=model(input_var,depth_image)

        # feature = torch.cat((output_1,output_2), 1)
        loss = criterion(feature,target_var)
        prec1, prec5,pred_mean_ac,target_mean_ac= accuracy(feature.data, target, topk=(1,5) )

        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        pred_mean.extend(pred_mean_ac)
        target_mean.extend(target_mean_ac)

        # measure elapsed time

        if i % 40 == 0:
            print(('Test: [{0}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'top5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                   i, loss=losses, top1=top1,top5=top5)))


    mean_acc_all=mean_acc(np.array(target_mean),np.array(pred_mean),67)



    print(('Testing Results_color: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))
    print('mean_acc: ',str(mean_acc_all))
    writer.add_scalar('val_loss_color', losses.avg, global_step=epoch)

    return mean_acc_all


# def build_output_keys(gen_img=True):

#         out_keys = []

#         if gen_img:
#             out_keys.append('gen_img')
#         return out_keys


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    # lr = cfg.LR
    if epoch <=10:
        lr = cfg.LR
    else:
        lr = cfg.LR * 0.9
        cfg.LR=lr
    print('-----------------------lr:',str(lr),'---------------------------')

    #lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def accuracy(output, target,topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    pred_mean=[]
    target_mean=[]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    pred_mean=pred[0].cpu().numpy()
    target_mean=target.cpu().numpy()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0],res[1],pred_mean,target_mean

def mean_acc(target_indice, pred_indice, num_classes, classes=None):
    acc = 0.
    result=np.zeros(num_classes)
    result_all=np.zeros(num_classes)
    for i in range(len(target_indice)):
        if target_indice[i]==pred_indice[i]:
            result[target_indice[i]]+=1
        result_all[target_indice[i]]+=1
    for i in range(num_classes):
        acc+=result[i]*1.0/result_all[i]
    return (acc / num_classes) * 100

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    # if cfg.RESUME:
    #     checkpoint_path = os.path.join(cfg.CHECKPOINTS_DIR, cfg.RESUME_PATH)
    #     checkpoint = torch.load(checkpoint_path)
    #     load_epoch = checkpoint['epoch']
    #     model.load_checkpoint(model.net, checkpoint_path, checkpoint, data_para=True)
    #     cfg.START_EPOCH = load_epoch
    # train

    # print('save model ...')
    # model_filename = '{0}_{1}_{2}.pth'.format(cfg.MODEL, cfg.WHICH_DIRECTION, cfg.NITER_TOTAL)
    # model.save_checkpoint(cfg.NITER_TOTAL, model_filename)

    # if writer is not None:
    #     writer.close()
def load_checkpoint_depth(net, checkpoint_path, checkpoint, optimizer=None, data_para=True):

        keep_fc = False

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

            # if self.phase == 'train' and not self.cfg.INIT_EPOCH:
            #     optimizer.load_state_dict(checkpoint['optimizer_ED'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['iter']))
        else:
            print("=> !!! No checkpoint found at '{}'".format(cfg.RESUME))
            return
def save_checkpoint(state, CorD):
    if CorD:
        filename='Color_model_best.pth.tar'
    else:
        filename='Depth_model_best.pth.tar'
    torch.save(state, filename)

if __name__ == '__main__':
    main()
