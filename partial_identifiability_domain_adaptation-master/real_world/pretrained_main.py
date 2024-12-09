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
from common.modules.networks import iVAE
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
import os
sys.path.append('.')
import utils
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
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
    dataset_list = []
    domain_ids = []
    all_domains = args.source+args.target
    args.num_classes = 345
    for source in args.source:
        labels = torch.load(osp.join(args.root, 'domainnet_label_%s_train.pt' % source))
        data = torch.load(osp.join(args.root, 'domainnet_%s_train.pt' % source))
        ds = torch.cat([torch.tensor([all_domains.index(source)])] * len(data),0)
        dataset = TensorDataset(data, labels, ds)
        dataset_list.append(dataset)
        domain_ids += ds
    train_source_dataset = torch.utils.data.ConcatDataset(dataset_list)
    train_source_dataset.domain_ids = domain_ids
    labels = torch.load(osp.join(args.root, 'domainnet_label_%s_train.pt' % args.target[0]))
    data = torch.load(osp.join(args.root, 'domainnet_%s_train.pt' % args.target[0]))
    dt = torch.cat([torch.tensor([all_domains.index(args.target[0])])] * len(data),0)
    train_target_dataset = TensorDataset(data, labels, dt)
    train_source_loader = DataLoader(train_source_dataset, batch_size=(args.n_domains-1)*args.batch_size,
                                     num_workers=args.workers, drop_last=True,
                                     sampler=_make_balanced_sampler(train_source_dataset.domain_ids))
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True,
                                     )
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)


    labels = torch.load(osp.join(args.root, 'domainnet_label_%s_val.pt' % args.target[0]))
    data = torch.load(osp.join(args.root, 'domainnet_%s_val.pt' % args.target[0]))
    val_dataset = TensorDataset(data, labels)

    labels = torch.load(osp.join(args.root, 'domainnet_label_%s_test.pt' % args.target[0]))
    data = torch.load(osp.join(args.root, 'domainnet_%s_test.pt' % args.target[0]))
    test_dataset = TensorDataset(data, labels)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)



    # create model
    print("=> using model '{}'".format(args.arch))
    classifier = iVAE(args).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(),
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    #optimizer = torch.optim.Adam(classifier.get_parameters(), lr=2e-4, weight_decay=args.weight_decay)
    print(optimizer.param_groups[0]['lr'], ' *** lr')
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    print(optimizer.param_groups[0]['lr'], ' *** lr')


    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    total_iter = 0
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr(), optimizer.param_groups[0]['lr'])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, optimizer,
              lr_scheduler, epoch, args, total_iter)
        total_iter += args.iters_per_epoch
        # evaluate on validation set
        acc1 = utils.validate(train_source_loader, classifier, args, device)
        print(' * Src Acc@1 %.3f' % (acc1))
        acc1 = utils.validate(val_loader, classifier, args, device)
        print(' * Val Acc@1 %.3f' % (acc1))
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(' * Test Acc@1 %.3f' % (acc1))

        source_feature = collect_feature(train_source_loader, classifier.extract_feature, device)
        target_feature = collect_feature(train_target_loader, classifier.extract_feature, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()

def l1_loss(x):
    return torch.mean(torch.abs(x))



def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model,  optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace, total_iter):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':4.2f')
    recon_losses = AverageMeter('Rec', ':4.2f')
    vae_losses = AverageMeter('VAE', ':4.2f')
    kl_style_losses = AverageMeter('KL_S', ':4.2f')
    cauinf_losses = AverageMeter('CauInf', ':4.2f')
    kl_content_losses = AverageMeter('KL_C', ':4.2f')
    cls_losses = AverageMeter('Cls', ':4.2f')
    ent_losses = AverageMeter('Ent', ':4.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    val_accs = AverageMeter('Val Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, cls_losses, ent_losses, vae_losses, recon_losses, kl_style_losses, kl_content_losses, cauinf_losses, cls_accs, val_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        total_iter += 1
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)

        x_s, labels_s, d_s = next(train_source_iter)
        x_t, labels_t, d_t = next(train_target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        x_all = torch.cat([x_s, x_t], 0)
        d_all = torch.cat([d_s, d_t], 0).to(device)
        label_all = torch.cat([labels_s, labels_t], 0)
        losses_cls = []
        losses_kl_style = []
        losses_kl_content = []
        img_all = []
        tilde_z_all = []
        y_t = None
        y_s = []
        labels_s = []
        for id in range(args.n_domains):
            domain_id = id
            is_target = domain_id == args.n_domains-1
            x_dom = x_all[d_all==id][:args.batch_size]
            label_dom = label_all[d_all==id][:args.batch_size]
            d_dom = d_all[d_all==id][:args.batch_size]
            z, tilde_z, f, mu, log_var, zstyle, flow_zstyle, tilde_z_target, h1 = model.encode(x_dom, d_dom, track_bn=is_target)
            logit = model.predict(f, track_bn=is_target)
            # vae loss
            cont_dim = args.z_dim - args.style_dim
            kl_content = (-0.5 * (1 + log_var[:, :cont_dim] - mu[:, :cont_dim].pow(2) - log_var[:, :cont_dim].exp())).sum(
                -1).mean()
            C_max = torch.tensor(args.C_max_content)
            C_content = torch.clamp(C_max / args.C_stop_iter * total_iter, 0, C_max.item())
            loss_kl_content = 10 * (kl_content - C_content).abs()
            if total_iter % 100 == 0:
                print(kl_content.item(), C_content, total_iter, log_var[:,:cont_dim].mean(), mu[:,:cont_dim].mean())

            kl_style = (-0.5 * (1 + log_var[:, cont_dim:] - mu[:, cont_dim:].pow(2) - log_var[:, cont_dim:].exp())).sum(
                -1).mean()
            C_max = torch.tensor(args.C_max_style)
            C_style = torch.clamp(C_max / args.C_stop_iter * total_iter, 0, C_max.item())
            loss_kl_style = args.beta * (kl_style - C_style).abs()

            if not is_target:  # only source
                losses_cls.append(F.cross_entropy(logit, label_dom))
                y_s.append(logit)
                labels_s.append(label_dom)
            else:
                y_t = logit
            losses_kl_content.append(loss_kl_content)
            losses_kl_style.append(loss_kl_style)
            img_all.append(x_dom)
            tilde_z_all.append(tilde_z)

        img_all = torch.cat(img_all, 0)
        tilde_z_all = torch.cat(tilde_z_all, 0)
        img_all_hat = model.decode(tilde_z_all)

        # vae loss
        mean_loss_recon = F.mse_loss(img_all, img_all_hat, reduction='sum') / len(img_all)
        mean_loss_kl_style = torch.stack(losses_kl_style, 0).mean()
        mean_loss_kl_content = torch.stack(losses_kl_content, 0).mean()
        mean_loss_vae = mean_loss_recon + mean_loss_kl_content + mean_loss_kl_style

        # source classification
        mean_loss_cls = torch.stack(losses_cls, 0).mean()

        # sparse loss
        loss_causal_influence = l1_loss(model.causal_influence)
        cont_dim = args.z_dim - args.style_dim
        loss_cls_content_influence = l1_loss(model.cls_influence[:, :cont_dim])
        loss_cls_style_influence = l1_loss(model.cls_influence[:, cont_dim:])

        # entropy loss
        loss_ent = torch.tensor(0.)
        if args.lambda_ent > 0:
            output_t = y_t
            entropy = F.cross_entropy(output_t, torch.softmax(output_t, dim=1), reduction='none').detach()
            index = torch.nonzero((entropy < args.entropy_thr).float()).squeeze(-1)
            select_output_t = output_t[index]
            if len(select_output_t) > 0:
                loss_ent = F.cross_entropy(select_output_t, torch.softmax(select_output_t, dim=1))


        loss = mean_loss_cls + args.lambda_vae * mean_loss_vae + args.lambda_cauinf * loss_causal_influence + \
               + args.lambda_ent * loss_ent + \
               args.lambda_clscont * loss_cls_content_influence + args.lambda_clsstyle * loss_cls_style_influence

        y_s = torch.cat(y_s, 0)
        labels_s = torch.cat(labels_s, 0)
        cls_acc = accuracy(y_s, labels_s)[0]
        cls_losses.update(mean_loss_cls.item(), y_s.size(0))
        recon_losses.update(mean_loss_recon.item(), x_all.size(0))
        cls_accs.update(cls_acc.item(), y_s.size(0))
        vae_losses.update(mean_loss_vae.item(), x_all.size(0))
        ent_losses.update(loss_ent.item(), y_t.size(0))
        kl_style_losses.update(mean_loss_kl_style.item(), x_all.size(0))
        kl_content_losses.update(mean_loss_kl_content.item(), x_all.size(0))
        cauinf_losses.update(loss_causal_influence.item(), 1)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            # not used in training
            model.eval()
            x_t = x_all[d_all==args.n_domains-1]
            labels_t = label_all[d_all==args.n_domains-1]
            with torch.no_grad():
                y = model(x_t, d_all[d_all==args.n_domains-1])
                cls_acc = accuracy(y, labels_t)[0]
                val_accs.update(cls_acc.item(), x_t.size(0))
            model.train()

            progress.display(i)

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
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=128, type=int,
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
    parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int,
                        help='Number of iterations per epoch')
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
    parser.add_argument('--z_dim', type=int, default=512, metavar='N')
    parser.add_argument('--style_dim', type=int, default=64, metavar='N')
    parser.add_argument('--beta', type=float, default=10., metavar='N')
    parser.add_argument('--name', type=str, default='experiments', metavar='N')
    parser.add_argument('--mode', type=str, default='vae', metavar='N')
    parser.add_argument('--cls_mode', type=str, default='style', metavar='N')
    parser.add_argument('--flow', type=str, default='ddsf', metavar='N')
    parser.add_argument('--flow_dim', type=int, default=16, metavar='N')
    parser.add_argument('--flow_nlayer', type=int, default=2, metavar='N')
    parser.add_argument('--norm_id', type=int, default=5, metavar='N')
    parser.add_argument('--init_value', type=float, default=0.0, metavar='N')
    parser.add_argument('--flow_bound', type=int, default=5, metavar='N')
    parser.add_argument('--flow_bins', type=int, default=8, metavar='N')
    parser.add_argument('--flow_order', type=str, default='linear', metavar='N')
    parser.add_argument('--net', type=str, default='dirt', metavar='N')
    parser.add_argument('--n_flow', type=int, default=2, metavar='N')
    parser.add_argument('--cls_dim', type=int, default=1024, metavar='N')
    parser.add_argument('--cls_nlayer', type=int, default=2, metavar='N')
    parser.add_argument('--lambda_vae', type=float, default=0.001, metavar='N')
    parser.add_argument('--lambda_cls', type=float, default=1., metavar='N')
    parser.add_argument('--lambda_cauinf', type=float, default=0.1, metavar='N')
    parser.add_argument('--lambda_clscont', type=float, default=.0, metavar='N')
    parser.add_argument('--lambda_clsstyle', type=float, default=.0, metavar='N')
    parser.add_argument('--lambda_ent', type=float, default=.1, metavar='N')
    parser.add_argument('--lambda_vat_s', type=float, default=0, metavar='N')
    parser.add_argument('--lambda_vat_t', type=float, default=0, metavar='N')
    parser.add_argument('--entropy_thr', type=float, default=0.5, metavar='N')
    parser.add_argument('--C_max', type=float, default=15, metavar='N')
    parser.add_argument('--C_max_content', type=float, default=120., metavar='N')
    parser.add_argument('--C_max_style', type=float, default=20., metavar='N')
    parser.add_argument('--max_iter', type=int, default=50000, metavar='N')
    parser.add_argument('--C_stop_iter', type=int, default=10000, metavar='N')
    parser.add_argument('--max_epoch', type=int, default=1000, metavar='N')
    args = parser.parse_args()
    model_id = '%s-%s-%s-%s-lam%s-beta_%s-D%d' % (args.data, args.target[0], args.name, args.mode, args.lambda_vae, args.beta,
                                                         args.style_dim)
    args.log = os.path.join(args.log, model_id)

    args.source = [i for i in args.source.split(',')]
    args.target = [i for i in args.target.split(',')]
    args.n_domains = len(args.source) + len(args.target)
    main(args)
