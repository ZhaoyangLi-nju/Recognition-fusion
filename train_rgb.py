import os
import random
from collections import Counter
from functools import reduce
from collections import OrderedDict
from collections import defaultdict
import torch
import torchvision
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
from torchvision import models

import copy
import numpy as np
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

def main():
    global cfg
    cfg = DefaultConfig()
    args = {
        'resnet18': RESNET18_SUNRGBD_CONFIG().args(),
    }

    # Setting random seed
    # if cfg.MANUAL_SEED is None:
    #     cfg.MANUAL_SEED = random.randint(1, 10000)
    # random.seed(cfg.MANUAL_SEED)
    # torch.manual_seed(cfg.MANUAL_SEED)

    # args for different backbones
    cfg.parse(args['resnet18'])
    run_id = random.randint(1, 100000)
    summary_dir='/home/lzy/summary/generateDepth/'+'train_rgb_'+str(run_id)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    writer = SummaryWriter(summary_dir)
    cfg.LR=0.0001
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'
    device_ids = torch.cuda.device_count()
    print('device_ids:', device_ids)
    # project_name = reduce(lambda x, y: str(x) + '/' + str(y), os.path.realpath(__file__).split(os.sep)[:-1])
    # util.mkdir('logs')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    # normalize=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
    #     val_dataset,batch_size=16, shuffle=False,
    #     num_workers=4, pin_memory=True)
    train_loader = DataProvider(cfg, dataset=train_dataset,batch_size=40, shuffle=True)
    val_loader = DataProvider(cfg, dataset=val_dataset, batch_size=10,shuffle=False)

    # class weights
    # num_classes_train = list(Counter([i[1] for i in train_loader.dataset.imgs]).values())
    # cfg.CLASS_WEIGHTS_TRAIN = torch.FloatTensor(num_classes_train)

    # writer = SummaryWriter(log_dir=cfg.LOG_PATH)  # tensorboard

    # net_classification_1=resnet50(pretrained=True)
    # net_classification_2=resnet50(pretrained=True)

    # net_classification_1 = models.__dict__['resnet18'](num_classes=365)
    # net_classification_2 = models.__dict__['resnet18'](num_classes=365)
    net_classification_1=torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
    # net_classification_2=torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
    # for param in net_classification_1.parameters():
    #     param.requires_grad = False
    # for param in net_classification_2.parameters():
    #     param.requires_grad = True
    net_classification_1.fc = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(2048, 1024),nn.LeakyReLU(inplace=True),nn.Linear(1024,67))
    # net_classification_2.fc = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(2048, 1024),nn.LeakyReLU(inplace=True),nn.Linear(1024,67))

    net_classification_1.load_state_dict(torch.load("./bestmodel/best_model_resnext_16d_2048_1024_dropout_0.5_b.pkl"))
    # net_classification_2.load_state_dict(torch.load("./bestmodel/best_model_resnext_16d_2048_1024_dropout_0.5_b.pkl"))
    # net_classification_2
    # load_path = "/home/dudapeng/workspace/pretrained/place/resnet18_places365.pth"
    # checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
    # state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    # net_classification_1.load_state_dict(state_dict)
    # net_classification_2.load_state_dict(state_dict)
    print(net_classification_1)




    # num_ftrs = net_classification_1.fc.in_features
    # net_classification_1.fc = nn.Linear(num_ftrs, cfg.NUM_CLASSES)
    # num_ftrs = net_classification_2.fc.in_features
    # net_classification_2.fc = nn.Linear(num_ftrs, cfg.NUM_CLASSES)


    net_classification_1.cuda()
    # net_classification_2.cuda()
    cudnn.benchmark = True

    # if cfg.GENERATE_Depth_DATA:
    #     print('GENERATE_Depth_DATA model set')
    #     cfg_generate = copy.deepcopy(cfg)
    #     cfg_generate.CHECKPOINTS_DIR='/home/lzy/generateDepth/checkpoints/best_AtoB/trecg_AtoB_best.pth'
    #     cfg_generate.GENERATE_Depth_DATA = False
    #     cfg_generate.NO_UPSAMPLE = False
    #     checkpoint = torch.load(cfg_generate.CHECKPOINTS_DIR)
    #     model = define_TrecgNet(cfg_generate, upsample=True,generate=True)
    #     load_checkpoint_depth(model,cfg_generate.CHECKPOINTS_DIR, checkpoint, data_para=True)
    #     generate_model = torch.nn.DataParallel(model).cuda()
    #     generate_model.eval()

    net_classification_1 = torch.nn.DataParallel(net_classification_1).cuda()
    # net_classification_2 = torch.nn.DataParallel(net_classification_2).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    best_mean_1=0
    # best_mean_2=0
    # optimizer = optim.SGD(model_ft.parameters(), lr=0.05,momentum=0.9)#,weight_decay=0.00005)
    optimizer_1 = torch.optim.SGD(net_classification_1.parameters(),lr=cfg.LR,momentum=cfg.MOMENTUM,weight_decay=cfg.WEIGHT_DECAY)
    # optimizer_2 = torch.optim.SGD(net_classification_2.parameters(),lr=cfg.LR,momentum=cfg.MOMENTUM,weight_decay=cfg.WEIGHT_DECAY)
    for epoch in range(0,100):
        adjust_learning_rate(optimizer_1,epoch)
        # meanacc_1,meanacc_2=validate(val_loader, net_classification_1,net_classification_2,generate_model,criterion,epoch)
# 
        train(train_loader, net_classification_1,criterion,optimizer_1,epoch,writer)
        meanacc_1=validate(val_loader, net_classification_1,criterion,epoch,writer)
        # meanacc_2=validate(val_loader,net_classification_2,generate_model,criterion,epoch,writer)

        # train(train_loader,net_classification_2,generate_model,criterion,optimizer_2,epoch,writer)
        # meanacc_2=validate(val_loader,net_classification_2,generate_model,criterion,epoch,writer)





        # writer.add_image(depth_image[0])
        # save best
        if meanacc_1>best_mean_1:
            best_mean_1=meanacc_1
            print('best_mean_color:',str(best_mean_1))
            save_checkpoint({
                'epoch': epoch,
                'arch': cfg.ARCH,
                'state_dict': net_classification_1.state_dict(),
                'best_mean_1': best_mean_1,
                'optimizer' : optimizer_1.state_dict(),
            },CorD=True)

        # if meanacc_2>best_mean_2:
        #     best_mean_2=meanacc_2
        #     print('best_mean_depth:',str(best_mean_2))
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'arch': cfg.ARCH,
        #         'state_dict': net_classification_2.state_dict(),
        #         'best_mean_2': best_mean_2,
        #         'optimizer' : optimizer_2.state_dict(),
        #     },CorD=False)
        print('best_mean_color:',str(best_mean_1))
        writer.add_scalar('mean_acc_color', meanacc_1, global_step=epoch)
        # writer.add_scalar('mean_acc_depth', meanacc_2, global_step=epoch)
        writer.add_scalar('best_meanacc_color', best_mean_1, global_step=epoch)
        # writer.add_scalar('best_meanacc_depth', best_mean_2, global_step=epoch)


    writer.close()





def train(train_loader,net_classification_1,criterion,optimizer_1,epoch,writer):
    losses_1 = AverageMeter()
    # losses_2 = AverageMeter()
    net_classification_1.train()
    # net_classification_2.train()
    # generate_model.eval()
    for i, (input,target) in enumerate(train_loader):

 
        # out_keys = build_output_keys(gen_img=True)
        # # self.set_input(data, self.cfg.DATA_TYPE)
        # depth_image=generate_model(source=input,out_keys=out_keys)
        # depth_image=depth_image.detach()
        # writer.add_image('depth_image'+str(i),torchvision.utils.make_grid(depth_image[:6].clone().cpu().data, 3,normalize=True), global_step=i*48)
        # writer.add_image('rgb_image'+str(i),torchvision.utils.make_grid(input[:6].clone().cpu().data, 3,normalize=True), global_step=i*48)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        # depth_var = torch.autograd.Variable(depth_image)
        
        #classification_1
        feature_1 = net_classification_1(input_var)
        loss_1 = criterion(feature_1, target)
        losses_1.update(loss_1.item(), input.size(0))
        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()
        # # classification_2
        # feature_2 = net_classification_2(depth_var)
        # loss_2 = criterion(feature_2,target)
        # losses_2.update(loss_2.item(), input.size(0))
        # optimizer_2.zero_grad()
        # loss_2.backward()
        # optimizer_2.step()
        if i % 40 == 0:
            print('Epoch: [{0}][{1}]\t'
            'Loss_color {loss_1.val:.4f} ({loss_1.avg:.4f})\t'
            # 'Loss_depth {loss_2.val:.4f} ({loss_2.avg:.4f})\t'
            .format(
               epoch, i,loss_1=losses_1))
    writer.add_scalar('train_loss_color', losses_1.avg, global_step=epoch)
    # writer.add_scalar('train_loss_depth', losses_2.avg, global_step=epoch)
    writer.add_scalar('LR',cfg.LR,global_step=epoch)

def validate(val_loader,net_classification_1,criterion,epoch,writer):
    losses_1 = AverageMeter()
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()
    # losses_2 = AverageMeter()
    # top1_2 = AverageMeter()
    # top5_2 = AverageMeter()
    # switch to evaluate mode
    net_classification_1.eval()
    # net_classification_2.eval()
    # generate_model.eval()

    pred_mean_1=[]
    target_mean_1=[]
    # pred_mean_2=[]
    # target_mean_2=[]
    for i, (input,target) in enumerate(val_loader):

        target = target.cuda(async=True)
        # out_keys = build_output_keys(gen_img=True)
        # # self.set_input(data, self.cfg.DATA_TYPE)
        # depth_image=generate_model(source=input,out_keys=out_keys)
        # depth_image=depth_image.detach()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            # depth_var = torch.autograd.Variable(depth_image)
        # compute output


        output_1 = net_classification_1(input_var)
        # output_2 = net_classification_2(depth_var)
        loss_1 = criterion(output_1,target)
        # loss_2 = criterion(output_2,target)

        # measure accuracy and record loss
        prec1_1, prec5_1,pred_mean_ac_1,target_mean_ac_1= accuracy(output_1.data, target, topk=(1,5) )
        # prec1_2, prec5_2,pred_mean_ac_2,target_mean_ac_2= accuracy(output_2.data, target, topk=(1,5) )

        losses_1.update(loss_1.data.item(), input.size(0))
        top1_1.update(prec1_1.item(), input.size(0))
        top5_1.update(prec5_1.item(), input.size(0))

        # losses_2.update(loss_2.data.item(), input.size(0))
        # top1_2.update(prec1_2.item(), input.size(0))
        # top5_2.update(prec5_2.item(), input.size(0))

        pred_mean_1.extend(pred_mean_ac_1)
        target_mean_1.extend(target_mean_ac_1)
        # pred_mean_2.extend(pred_mean_ac_2)
        # target_mean_2.extend(target_mean_ac_2)

        if i % 40 == 0:
            print(('Test: [{0}]\t'
                  'Loss_color {loss_1.val:.4f} ({loss_1.avg:.4f})\t'
                  'acc_color {top1_1.val:.3f} ({top1_1.avg:.3f})\t'.format(
                   i, 
                   loss_1=losses_1, top1_1=top1_1)))

    mean_acc_all_1=mean_acc(np.array(target_mean_1),np.array(pred_mean_1),67)
    # mean_acc_all_2=mean_acc(np.array(target_mean_2),np.array(pred_mean_2),67)

    writer.add_scalar('val_loss_color', losses_1.avg, global_step=epoch)
    # writer.add_scalar('val_loss_depth', losses_2.avg, global_step=epoch)
    print(('Testing Results_color: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1_1, top5=top5_1, loss=losses_1)))
    # print(('Testing Results_depth: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
    #       .format(top1=top1_2, top5=top5_2, loss=losses_2)))

    print('mean_acc_color: ',str(mean_acc_all_1))
    # print('------------mean_acc_depth: ',str(mean_acc_all_2))

    return mean_acc_all_1

def build_output_keys(gen_img=True):

        out_keys = []

        if gen_img:
            out_keys.append('gen_img')
        return out_keys


def adjust_learning_rate(optimizer1,epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    if epoch < 20:
        lr = cfg.LR
    elif epoch >= 20 and epoch < 100:
        lr = cfg.LR * 0.9
        cfg.LR=lr
    print('-----------------------lr:',str(lr),'---------------------------')
    # lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    # for param_group in optimizer2.param_groups:
    #     param_group['lr'] = lr*1.5
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
def save_checkpoint(state, CorD):
    if CorD:
        filename='_resnext101_Color___best.pth.tar'
    else:
        filename='_resnext101_Depth___best.pth.tar'
    torch.save(state, filename)

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

if __name__ == '__main__':
    main()
