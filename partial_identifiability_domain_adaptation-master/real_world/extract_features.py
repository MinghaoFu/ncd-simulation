import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

import os
sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    train_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    if args.data.lower() == 'domainnet':
        backbone = utils.get_model('resnet101', pretrain=True).to(device)
        backbone = nn.Sequential(backbone, nn.AdaptiveAvgPool2d(output_size=(1,1)), nn.Flatten())
        backbone.eval()
        all_dataset = ['i', 'p', 'q', 'r', 's', 'c']
        for dataset in all_dataset:
            source_dataset = ['i', 'p', 'q', 'r', 's', 'c']
            source_dataset.remove(dataset)
            dataset = [dataset]
            _, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
                utils.get_dataset(args.data, args.root, source_dataset, dataset, train_transform, val_transform)
            train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            print('converting training data!')
            features = []
            labels = []
            for i,data in enumerate(train_target_loader):
                if i % 100 == 0:
                    print('[%d]' % (i+1))
                img, target, domain_index, path = data
                img = img.to(device)
                with torch.no_grad():
                    #feats = backbone(img).detach()
                    #assert len(feats.size()) == 2
                    #features.append(feats)
                    labels.append(target)
            labels = torch.cat(labels, 0)
            torch.save(labels.detach().cpu(), osp.join(args.root, 'domainnet_label_%s_train.pt' % dataset[0]))
            #features = torch.cat(features, 0)
            #torch.save(features.detach().cpu(), osp.join(args.root, 'domainnet_%s_train.pt' % dataset[0]))
            print('converting training data finished! %d samples in domain %s' % (i+1, dataset[0]))

            print('converting validation data!')
            features = []
            labels = []
            for i,data in enumerate(val_loader):
                if i % 100 == 0:
                    print('[%d]' % (i+1))
                img, target, domain_index, path = data
                img = img.to(device)
                with torch.no_grad():
                    #feats = backbone(img).detach()
                    #features.append(feats)
                    labels.append(target)
            labels = torch.cat(labels, 0)
            torch.save(labels.detach().cpu(), osp.join(args.root, 'domainnet_label_%s_val.pt' % dataset[0]))
            #features = torch.cat(features, 0)
            #torch.save(features.detach().cpu(), osp.join(args.root, 'domainnet_%s_val.pt' % dataset[0]))
            print('converting validation data finished! %d samples in domain %s' % (i+1, dataset[0]))


            print('converting testing data!')
            features = []
            labels = []
            for i,data in enumerate(test_loader):
                if i % 100 == 0:
                    print('[%d]' % (i+1))
                img, target, domain_index, path = data
                img = img.to(device)
                with torch.no_grad():
                    #feats = backbone(img).detach()
                    #features.append(feats)
                    labels.append(target)
            labels = torch.cat(labels, 0)
            torch.save(labels.detach().cpu(), osp.join(args.root, 'domainnet_label_%s_test.pt' % dataset[0]))
            #features = torch.cat(features, 0)
            torch.save(features.detach().cpu(), osp.join(args.root,'domainnet_%s_test.pt' % dataset[0]))
            print('converting testing data finished! %d samples in domain %s' % (i+1, dataset[0]))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('--root', type=str, default='../da_datasets/domainnet',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='DomainNet', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', default='i,p,q,r,s')
    parser.add_argument('-t', '--target', help='target domain(s)', default='c')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=2048, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--beta', default=1., type=float)
    parser.add_argument('--lambda_vae', default=.1, type=float)
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay',default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--style_dim', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument("--mode", type=str, default='vae')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-e', '--eval-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    model_id = '%s-%s-%s-%s-lam%s-beta_%s-D%d' % (args.data, args.target[0], args.name, args.mode, args.lambda_vae, args.beta,
                                                  args.style_dim)
    args.log = os.path.join(args.log, model_id)
    args.source = [i for i in args.source.split(',')]
    args.target = [i for i in args.target.split(',')]
    main(args)



