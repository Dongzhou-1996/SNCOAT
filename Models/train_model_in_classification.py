import argparse
import numpy as np
import os
import sys
import cv2
import time
import shutil
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
import matplotlib.pyplot as plt

from Utils.ImageNet_dataloader import data_loader
from Models.attention_module import MultiHeadAttention
from critic_models import ConvNet
from torch.utils.tensorboard import SummaryWriter
from ptflops import get_model_complexity_info


class SENet(nn.Module):
    def __init__(self, input_channels=3, class_num=1000, with_MHA=True, vis=False, vis_dir='',
                 with_BN=True, with_layer4=True, with_maxpool=True, with_SE=True):
        super(SENet, self).__init__()
        self.backbone = ConvNet(input_channels, with_BN, with_layer4,
                                with_maxpool, with_SE, vis=vis)

        if with_MHA:
            self.attention = MultiHeadAttention(in_features=128, head_num=8)

        self.fc1 = nn.Linear(128 * 5 * 5, 2048)
        self.drop_out = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(2048, class_num)

        self.with_MHA = with_MHA
        self.with_SE = with_SE
        self.vis = vis
        self.vis_dir = os.path.join(vis_dir, 'visualization')
        if os.path.exists(self.vis_dir):
            print('=> visualization directory is existed, it will be re-created soon ...')
            shutil.rmtree(self.vis_dir)
            os.makedirs(self.vis_dir)
        else:
            print('=> visualization directory is not existed, it will be created soon ...')
            os.makedirs(self.vis_dir)

        self.attention_visualization = {}

    def forward(self, x):
        self.attention_visualization.update({'input': x})
        out = self.backbone(x)
        self.attention_visualization.update({'embedding_feature': out})
        if self.with_MHA:
            res = self.attention(out).transpose(-1, -2).view(*out.shape)
            self.attention_visualization.update({'context_res': out})
            out += res
            self.attention_visualization.update({'context': out})

        out = torch.flatten(out, start_dim=1)
        out = F.relu(self.fc1(out))
        out = self.drop_out(out)
        out = self.fc2(out)

        if not self.vis:
            self.visualization_clear()

        return out

    def visualization_clear(self):
        self.attention_visualization.clear()
        self.backbone.attention_visualization.clear()

    def visualization(self):

        vis_time = time.time()
        # visualize overall network
        if len(self.attention_visualization) > 0:
            for k, tensor in self.attention_visualization.items():
                sample = tensor[0]
                # sample: CxHxW
                sample = sample.permute(1, 2, 0)  # HxWxC
                file_name = os.path.join(self.vis_dir, '{}_{}.jpg'.format(vis_time, k))
                if k == 'input':
                    plt.imshow(sample[..., :3].cpu().numpy().astype(np.float))
                    plt.savefig(file_name)
                else:
                    attention_map = sample.pow(2).mean(2).detach().cpu().numpy()  # HxW
                    attention_map = cv2.resize(src=attention_map, dsize=(255, 255), interpolation=cv2.INTER_CUBIC)
                    plt.imshow(attention_map.astype(np.float))
                    plt.savefig(file_name)

        self.backbone.visualization(self.vis_dir, vis_time)

        self.visualization_clear()


def model_params(eval_net, model_name='ConvQNet', input_shape=(3, 255, 255)):
    flops, params = get_model_complexity_info(eval_net, input_shape,
                                              as_strings=True, print_per_layer_stat=True)
    print('Model name: ' + model_name)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    print('=========================================================')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluation(val_loader, model, loss_func, log_dir=''):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = loss_func(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % 100 == 0:
            model.visualization()
        else:
            model.visualization_clear()

        print('\rEval: [{0}/{1}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader), loss=losses, top1=top1, top5=top5), end='')

    print('\n * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(train_loader, model, loss_func, optimizer, epoch, log_dir=''):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = loss_func(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        model.visualization_clear()

        print('\rEpoch: [{0}][{1}/{2}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format( epoch, i, len(train_loader), loss=losses, top1=top1, top5=top5), end='')
    print('\n')
    return losses.avg, top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser(description='SENet with ImageNet')
parser.add_argument('--data_root', default='/home/group1/dzhou/dataset/ImageNet', type=str,
                    metavar='DIR', help='path to dataset')
parser.add_argument('--log_dir', default='train/', type=str, help='path to save results')
parser.add_argument('--gpu_idx', default=0, type=int, help='the index of gpu to use')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--restore', default='./train/ConvNet_MHA/model_ep_91.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--with_SE', action='store_true',
                    help='whether to adopt SE layers in ConvNet')
parser.add_argument('--with_MHA', action='store_true',
                    help='whether to adopt MHA layers in ConvNet')
parser.add_argument('--visualize', action='store_true',
                    help='whether to visualize network attention')

args = parser.parse_args()
device = torch.device('cuda', args.gpu_idx) if torch.cuda.device_count() > args.gpu_idx else \
    torch.device('cpu', args.gpu_idx)


def main():
    model_name = 'ConvNet'
    with_SE = args.with_SE
    with_MHA = args.with_MHA
    visualize = args.visualize
    pretrained = args.pretrained
    evl = args.evaluate
    # with_SE = False
    # with_MHA = True
    # visualize = True
    # pretrained = True
    # evl = True

    if with_SE:
        model_name += '_SE'

    if with_MHA:
        model_name += '_MHA'

    log_dir = os.path.join(args.log_dir, model_name)
    if not os.path.exists(log_dir):
        print('=> log directory ({}) is not existed! It will be created soon ...'.format(log_dir))
        os.makedirs(log_dir)

    best_prec1 = 0
    # create model
    lr = args.lr
    model = SENet(3, 1000, with_SE=with_SE, with_MHA=with_MHA, vis=visualize, vis_dir=log_dir)
    model_params(model, model_name=model_name, input_shape=(3, 255, 255))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    start_epoch = args.start_epoch
    if pretrained:
        print("=> using pre-trained model!")
        if os.path.exists(args.restore):
            print('=> loading checkpoint: {}'.format(args.restore))
            checkpoint = torch.load(args.restore, map_location=lambda storage, loc: storage.cpu())
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no model file is found in {}'.format(args.restore))

    model.to(device)

    train_loader, val_loader = data_loader(args.data_root, batch_size=args.batch_size,
                                           image_size=255, workers=args.workers, pin_memory=True)

    if evl:
        evaluation(val_loader, model, loss_func, log_dir)
        return
    else:
        summary_writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        train_loss, train_prec1, train_prec5 = train(train_loader, model, loss_func, optimizer, epoch, log_dir)
        summary_writer.add_scalar('loss/train', train_loss, epoch)
        summary_writer.add_scalar('prec1/train', train_prec1, epoch)
        summary_writer.add_scalar('prec5/train', train_prec5, epoch)

        test_loss, test_prec1, test_prec5 = evaluation(val_loader, model, loss_func)
        summary_writer.add_scalar('loss/test', train_loss, epoch)
        summary_writer.add_scalar('prec1/test', test_prec1, epoch)
        summary_writer.add_scalar('prec5/test', test_prec5, epoch)

        best_prec1 = max(test_prec1, best_prec1)

        if epoch % 10 == 0:
            model_path = os.path.join(log_dir, 'model_ep_{:02d}.pth'.format(epoch + 1))
            print('=> saving network to {} ...'.format(model_path))
            checkpoint = {'epoch': epoch,
                          'best_prec1': best_prec1,
                          'state_dict': model.state_dict(),
                          'backbone_params': model.backbone.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
            print('=> model params is saved!')


if __name__ == '__main__':
    main()